import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.asyncio import tqdm
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader

from .samplers import CondSampler
from .train import SAVE_PATH
from .dataset import SQGLeadTimeDataset, SQGDataset, DATA_PATH, DATA_TEST, DATA_test
from .utils import plot_metrics, print_metrics, MetricAccumulator


def get_latent_chain(ens, n, sample_shape, alpha=1.0, device='cpu'):
    """One AR(1) noise chain of length n per ensemble member.

    get_latents() is called once per AR window, so it RESETS the noise at every
    window edge: correlation drops to 0 across the 24 / 48 / 72 / 96 boundaries.
    This builds a single chain spanning the whole rollout instead, so
    corr(z_i, z_j) = alpha**|i-j| all the way out with no discontinuity — which
    is what the latent SDE we are fitting actually claims.

    Variance preserving: every z[:, i] is standard normal.
    alpha=1.0 is fixed noise, alpha=0.0 is independent noise.

    Returns:
        z : (ens, n, C, H, W)
    """
    C, H, W = sample_shape

    z = torch.zeros((ens, n, C, H, W), device=device)
    z[:, 0] = torch.randn((ens, C, H, W), device=device)
    a = torch.tensor(float(alpha), device=device)

    for i in range(1, n):
        noise = torch.randn((ens, C, H, W), device=device)
        z[:, i] = a * z[:, i - 1] + (1 - a ** 2).sqrt() * noise

    return z


def arci(
    sampler,
    traj,
    anchor=0,
    max_lead=24,
    alpha=0.5, 
    ens=20,
    latent_fn=None,
    device='cpu',
    n_micro=1,
    micro_batch=None,
):
    """AR + Continuous forecasting of ONE trajectory, from `anchor` to its end.

    The model only knows direct leads 1..max_lead, so reaching the end of a
    100-frame trajectory means re-anchoring on its own forecast every max_lead
    steps. There is no n_ar / max_ar knob: the number of windows is whatever it
    takes to cover the trajectory, ceil(n / max_lead). The old code derived
    n_ar with a floor division and silently dropped the tail — with anchor=0 it
    stopped at state 96 and never forecast 97, 98, 99.

    Args:
        sampler   : CondSampler, already loaded and on `device`
        traj      : (T, C, H, W) ONE normalised trajectory
        anchor    : frame the forecast starts from. State `anchor` is the
                    initial condition and is NOT forecast.
        max_lead  : largest direct lead the model was trained on (24)
        ens       : ensemble members
        latent_fn : optional replacement for get_latent_chain, called as
                    latent_fn(ens, n, (C, H, W), device) -> (ens, n, C, H, W)
        n_micro   : how many pieces to cut a full window into. 1 (default)
                    sends the whole window in one sampler call; 2 halves peak
                    VRAM, 4 quarters it, and so on. This is the knob to tune —
                    it is relative, so it keeps working when you change `ens`
                    and you never have to know a window is ens*max_lead rows.
                    Exactly equivalent to micro_batch=ens*max_lead/n_micro.
        micro_batch : absolute cap on rows per call, overriding n_micro when
                    set. Prefer n_micro; reach for this only to pin VRAM to a
                    card regardless of ens, since a fixed n_micro lets memory
                    grow with ens (double ens, double peak).

                    Equivalent in exact arithmetic: the rows of a window are
                    independent conditional samples and the network has no
                    cross-batch operation (73 GroupNorm, zero BatchNorm). In
                    fp32 it is NOT bit-identical — measured max|diff| ~9e-6,
                    because the batch size feeds into the reduction order inside
                    the conv/matmul kernels. torch.use_deterministic_algorithms
                    does not remove this; it only fixes run-to-run variation at
                    a FIXED shape. The gap is ~3000x below the forecast RMSE at
                    lead 1 (2.6e-2), so it cannot move any metric; a genuine row
                    misalignment would show up at O(1) instead (swapping two
                    ensemble members gives 1.2).

                    It does not touch the AR structure — windows stay
                    sequential, because each one re-anchors on the previous
                    window's output.

    Returns:
        predicts : (n, ens, C, H, W)  flat over lead; index 0 is anchor+1
        truth    : (n, C, H, W)       the same frames from `traj`
        with n = T - 1 - anchor  (99 for a 100-frame trajectory from anchor 0)
    """
    assert n_micro >= 1, f'n_micro must be at least 1, got {n_micro}'
    assert micro_batch is None or micro_batch >= 1, \
        f'micro_batch must be at least 1 or None, got {micro_batch}'

    T, C, H, W = traj.shape
    n = T - 1 - anchor
    if n <= 0:
        raise ValueError(
            f'anchor={anchor} leaves nothing to forecast in a {T}-frame trajectory'
        )

    # ONE chain over the whole rollout, not one per window.

    if latent_fn is None:
        z = get_latent_chain(ens, n, (C, H, W), alpha=alpha, device=device)
    else:
        z = latent_fn(ens, n, (C, H, W), device)
    assert z.shape == (ens, n, C, H, W), f'latents shape wrong: {tuple(z.shape)}'

    # Peak memory is set by the LARGEST window (ens*max_lead rows), so size the
    # chunk from that once, before the loop. Recomputing it per window would cut
    # the short final window into even smaller pieces for no memory benefit, and
    # would stop n_micro=k agreeing with micro_batch=ens*max_lead/k.
    cap = micro_batch if micro_batch is not None else -(-(ens * max_lead) // n_micro)

    truth    = traj[anchor + 1: anchor + 1 + n]
    predicts = torch.zeros(n, ens, C, H, W)

    # every member starts from the same (single, true) initial condition
    x = traj[anchor].unsqueeze(0).expand(ens, C, H, W).to(device)

    done = 0
    while done < n:
        L = min(max_lead, n - done)          # 24, 24, 24, 24, 3 for n=99

        # The label is the lead WITHIN this window and is always divided by
        # max_lead, never by L. The model has no notion of absolute time: the
        # first step of the last window is 1/24 even though it is +97 h.
        labels = torch.arange(1, L + 1, dtype=torch.float32, device=device) / max_lead

        # batch-major / lead-minor, the order get_latents has always produced:
        #   [m0 l0..l(L-1), m1 l0..l(L-1), ...]
        # z0, x_t and lead_times must all agree on it.
        z0   = z[:, done:done + L].reshape(ens * L, C, H, W)
        x_t  = x.repeat_interleave(L, dim=0)
        lead = labels.repeat(ens)

        # Split the window's rows if asked. All three tensors are built in the
        # same order and sliced identically, so each chunk stays self-consistent
        # and torch.cat restores the original row order exactly.
        #
        rows = ens * L
        if cap >= rows:
            pred = sampler.sample(z0=z0, x_t=x_t, lead_times=lead)
        else:
            pred = torch.cat([
                sampler.sample(z0=z0[s:s + cap],
                               x_t=x_t[s:s + cap],
                               lead_times=lead[s:s + cap])
                for s in range(0, rows, cap)
            ], dim=0)

        pred = pred.reshape(ens, L, C, H, W)                    # sample() returns cpu
        predicts[done:done + L] = pred.transpose(0, 1)

        # each member re-anchors on ITS OWN last frame
        x = pred[:, -1].to(device)
        done += L

    return predicts, truth


def arci_dataset(sampler, ds, n_traj=None, ens=20, alpha=0.5,
                 latent_fn=None, device='cpu', n_micro=1, micro_batch=None):
    """Run arci over the first n_traj trajectories of an SQGDataset.

    Returns:
        results  : list of (predicts, truth), one per trajectory
        traj_ids : list of int, the file id each result came from
    """
    n_traj = len(ds) if n_traj is None else min(n_traj, len(ds))

    results, traj_ids = [], []
    for i in tqdm(range(n_traj), total=n_traj):
        traj, traj_id = ds[i]
        results.append(
            arci(sampler, traj, ens=ens, alpha=alpha,
                 latent_fn=latent_fn, device=device,
                 n_micro=n_micro, micro_batch=micro_batch)
        )
        traj_ids.append(traj_id)

    return results, traj_ids


# this forecasting function will return truth value for each prediction
# I can actually do that separately
# So I still need a function performing prediction only
def forecast_analysis(
    sampler,
    data_dir=DATA_test,
    seed=42,
    latents = None,
    n_traj = None,
    n_ens = None
):
    '''The forecasting function for evaluation learned latent space
    which means no AR step

    Inputs:
        latents: numpy array (N, ens, 24, 2, 64, 64)
    Outputs:
        preds: (N, ens, 24, 2, 64, 64)
        truths: (N, 24, 2, 64, 64)

    '''
    # the number of trajectories forecast and ensemble configuration depends on latents
    # assume that latents (N, ens, 24, 2, 64, 64)
    N, ens, leads, levels, H, W = latents.shape
    assert n_traj == N, "Check if trajectories number matches with latent space data."
    assert n_ens == ens, "Check if ensemble number mathces with latent space data."
    assert leads == 24, "Check if leads in latent space data is 24."
    assert latents.is_cuda, "Run on your GPU."
    ds = SQGLeadTimeDataset(
        data_dir         = data_dir,
        split            = 'test',
        random_lead_time = False,
        eval_traj_num = n_traj,
        seed=seed
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    i = 0
    preds = torch.zeros((n_traj, ens, leads, levels, H, W))
    truths = torch.zeros((n_traj, leads, levels, H, W))
    for initial, truth, lead_time, traj_idx, anchor in tqdm(loader, total=len(loader)):
        truths[i] = truth.squeeze(0)
        latent = latents[i]  # latent [leads, levels, H , W]
        # traj_idx      = traj_idx.item()
        # anchor      = anchor.item()


        time_labels = lead_time.squeeze(0).repeat(ens)
        assert time_labels.shape == (ens * leads,), f"time_labels shape wrong: {time_labels.shape}"

        initials = initial.repeat(ens * leads, 1, 1, 1)
        assert initials.shape == (ens * leads, levels, H, W), f"initials shape wrong: {initials.shape}"

        # reshape latent into (ens * leads, levels, H, W)
        latent = latent.reshape(ens * leads, levels, H, W)
        pred = sampler.sample(z0=latent, x_t=initials, lead_times=time_labels)
        pred = pred.reshape(ens, leads, levels, H, W).contiguous()
        preds[i] = pred

        i = i + 1

    return preds.cpu().numpy(), truths.cpu().numpy()

def just_forecasting(latents, x0, sampler):
    # this should also support round trip
    # whose latent space data only has 4 dimensions
    # but because the code in get_latents only allows to have dimension N
    dim = len(latents.shape)
    assert dim == 4 or dim == 6, f"This function only support single step forecasting yours be like:{latents.shape}."
    if dim == 4:
        ens = 1
        N = 1
        L, C, H, W = latents.shape
    elif dim == 6:
        N, ens, L, C, H, W = latents.shape
    sample_shape = (C, H, W)

    latents = latents.reshape(N * ens * L, *sample_shape)
    if x0.dim() == 3:
        # one trajectory
        x0 = x0.unsqueeze(0)
    x0 = x0.repeat_interleave(ens * L, dim=0)
    time_labels = (torch.arange(1, L+1, dtype=torch.float32) / L).repeat(N*ens)
    pred = sampler.sample(z0=latents, x_t=x0, lead_times=time_labels)
    pred = pred.reshape(N, ens, L, *sample_shape).contiguous()
    return pred

def get_latents(latent_shape, n_direct, alpha=1.0, device='cpu'):
    """
    The noise correlation done in Algorithm 2 reparameterized with alpha instead of rho.
    Note that this will only affect the direct forecasting, not the iterative timesteps.
    Variance preserving function for the noise z.
    alpha=1.0 means fixed noise,
    alpha=0.0 means uncorrelated noise
    """

    # about alpha, fixed noise or uncorrelated noise is neither what I want
    B, C, H, W = latent_shape # latent_shape: (n_samples * n_ens, num_variables, dx, dy)

    z = torch.zeros((n_direct, B, C, H, W), device=device)
    z[0] = torch.randn((B, C, H, W), device=device)
    alpha = torch.tensor(alpha, device=device)

    for t in range(1, n_direct):
        noise = torch.randn((B, C, H, W), device=device)
        z[t] = alpha * z[t - 1] + (1 - alpha**2).sqrt() * noise

    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W) # Transposing makes sure the order is preserved.

    return z

# The pooled lag-1 correlation of the inverted val latents, measured over all
# 1200 windows by Analysis/utils.compute_scalar_alpha (raw, not centered):
#     alpha = S01 / sqrt(S00 * S11) = 0.7133258747
# Deliberately the CORRELATION and not the cubic MLE (0.7459660702): the cubic
# assumes unit-variance latents, and the measured per-lead variance runs
# 0.577 (lead 1) to 0.943 (lead 23), so that assumption does not hold.
# get_latent_chain is variance-preserving by construction, so the quantity it
# has to reproduce is the correlation.
ALPHA_CONST = 0.7133258747


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def perform_forecasting(eval_traj_num=50, ens=20, alpha=ALPHA_CONST, n_micro=4,
                        micro_batch=None, anchor=0, steps=100,
                        model_path=SAVE_PATH, dirname='predict_constant_alpha',
                        device=None, test_pl=False, resume=True):
    '''AR+continuous forecast of whole test trajectories with a constant-alpha
    latent chain, writing results and metrics as it goes.

    One trajectory at a time, saved and then dropped. The previous version
    collected every result in a list before doing anything with them, which at
    50 trajectories is 3.2 GiB of float32 held at once -- more than this
    machine has free. Peak is now one trajectory, about 65 MiB, so the run
    size is limited by disk rather than RAM.

    GIFs are not produced. Rendering 99 frames x 50 trajectories through
    matplotlib needed every result in memory at the end, which is exactly what
    this restructure removes; the saved .npy files can be animated afterwards.

    Args:
        eval_traj_num : test trajectories to forecast, from the start of the split
        ens           : ensemble members per trajectory
        alpha         : lag-1 correlation of the latent chain (see ALPHA_CONST)
        n_micro       : pieces to cut each sampler window into. A window is
                        ens*24 rows; 480 rows OOMs on a 6 GB card, so ens=20
                        needs 4 (120 rows, 2.2 GiB). Throughput is flat in
                        batch size, so this costs nothing.
        micro_batch   : absolute rows-per-call cap, overriding n_micro
        anchor        : frame each forecast starts from. 0 means the whole
                        trajectory: 99 leads, re-anchoring every 24.
        test_pl       : 3-trajectory / ens-5 smoke run into a _smoketest
                        directory, so it cannot touch real output
        resume        : skip trajectories already on disk and rebuild their
                        contribution to the metrics by reading them back.
                        A 15 h run should not restart from zero, and the
                        metrics are plain sums over trajectories, so a
                        resumed run gives the same numbers as one pass.

    Writes to data/{dirname}/:
        sample{NN}_predicts.npy  (n_lead, ens, C, H, W) float32
        sample{NN}_truth.npy     (n_lead,      C, H, W) float32
        traj_ids.npy             which trajectory each sample came from
        metrics.npz              rmse / crps / ssr / spread / counts / lead_times
        metrics.png              the three-panel plot
        meta.json                alpha, model hash, settings, timings
    '''
    if test_pl:
        eval_traj_num, ens, dirname = 3, 5, dirname + '_smoketest'

    device   = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = DATA_PATH / dirname
    save_dir.mkdir(parents=True, exist_ok=True)

    # limit= matters: SQGDataset loads eagerly, so asking for all 150 test
    # trajectories to forecast 50 of them would cost 330 MB for nothing.
    ds      = SQGDataset(DATA_TEST, limit=eval_traj_num)
    n_traj  = min(eval_traj_num, len(ds))
    sampler = CondSampler(model_path=model_path, device=device, steps=steps)

    print(f'Forecasting {n_traj} trajectories from {DATA_TEST}')
    print(f'  alpha={alpha!r} ens={ens} anchor={anchor} n_micro={n_micro} '
          f'steps={steps} device={device}')
    print(f'  -> {save_dir}')

    # Provenance guard, same reasoning as the invert stage: resume ADDS to
    # what is already there, so a folder written with a different alpha or a
    # different checkpoint must not be silently extended.
    prev = save_dir / 'meta.json'
    if resume and prev.exists():
        pm = json.loads(prev.read_text())
        mismatch = []
        if pm.get('alpha') != alpha:
            mismatch.append(f'alpha {pm.get("alpha")} != {alpha}')
        if pm.get('model', {}).get('sha256') not in (None, _sha256(model_path)):
            mismatch.append('model checkpoint differs')
        if pm.get('ens') not in (None, ens):
            mismatch.append(f'ens {pm.get("ens")} != {ens}')
        if pm.get('anchor') not in (None, anchor):
            mismatch.append(f'anchor {pm.get("anchor")} != {anchor}')
        if mismatch:
            raise SystemExit(
                f'{save_dir} was written with different settings:\n'
                + ''.join(f'  - {m}\n' for m in mismatch)
                + 'Move the directory aside, or pass resume=False to overwrite it.'
            )

    acc, traj_ids = MetricAccumulator(), []
    t0 = time.time()
    done = 0
    for i in tqdm(range(n_traj), total=n_traj, desc='forecast'):
        pred_path  = save_dir / f'sample{i:02d}_predicts.npy'
        truth_path = save_dir / f'sample{i:02d}_truth.npy'

        # Already forecast: read it back into the accumulator instead of
        # spending 18 minutes of GPU redoing it. mmap so the 62 MiB file is
        # paged in rather than held twice.
        if resume and pred_path.exists() and truth_path.exists():
            acc.add(np.load(pred_path, mmap_mode='r'),
                    np.load(truth_path, mmap_mode='r'))
            traj_ids.append(int(ds.traj_ids[i]))
            done += 1
            continue

        traj, traj_id = ds[i]
        predicts, truth = arci(sampler, traj, anchor=anchor, ens=ens,
                               alpha=alpha, device=device,
                               n_micro=n_micro, micro_batch=micro_batch)

        np.save(save_dir / f'sample{i:02d}_predicts.npy', predicts.numpy())
        np.save(save_dir / f'sample{i:02d}_truth.npy',    truth.numpy())
        acc.add(predicts, truth)
        traj_ids.append(int(traj_id))

        # Drop the big tensors before the next trajectory allocates its own.
        del predicts, truth
        np.save(save_dir / 'traj_ids.npy', np.asarray(traj_ids))

    elapsed = time.time() - t0
    if done:
        print(f'resumed: {done} of {n_traj} trajectories were already on disk')
    metrics = acc.finalize()

    np.savez(save_dir / 'metrics.npz', **metrics)
    plot_metrics(metrics, out_path=save_dir / 'metrics.png')
    print_metrics(metrics['lead_times'], metrics['rmse'],
                  metrics['crps'], metrics['ssr'])

    (save_dir / 'meta.json').write_text(json.dumps({
        'created':       time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'alpha':         alpha,
        'alpha_source':  'pooled lag-1 correlation of data/invert/from_val '
                         '(1200 windows, raw, compute_scalar_alpha)',
        'latent_chain':  'get_latent_chain: one AR(1) chain over all leads, '
                         'variance preserving, not reset at AR boundaries',
        'n_traj':        n_traj,
        'ens':           ens,
        'anchor':        anchor,
        'steps':         steps,
        'n_micro':       n_micro,
        'micro_batch':   micro_batch,
        'data_dir':      str(DATA_TEST),
        'traj_ids':      traj_ids,
        'elapsed_seconds': round(elapsed, 1),
        'model': {
            'path':   str(model_path),
            'bytes':  model_path.stat().st_size,
            'sha256': _sha256(model_path),
        },
        'torch':  torch.__version__,
        'device': str(device),
    }, indent=2, sort_keys=True))

    print(f'\nSaved {n_traj} samples to {save_dir} in {elapsed/3600:.2f} h')
    print(f'  metrics.png / metrics.npz / meta.json written')
    return save_dir


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description='AR+continuous forecasting over the test split, with a '
                    'constant-alpha latent chain.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--eval-traj-num', type=int, default=50,
                   help='test trajectories to forecast')
    p.add_argument('--ens', type=int, default=20,
                   help='ensemble members per trajectory')
    p.add_argument('--alpha', type=float, default=ALPHA_CONST,
                   help='lag-1 correlation of the latent chain; the default is '
                        'the value measured on the inverted val latents')
    p.add_argument('--anchor', type=int, default=0,
                   help='frame each forecast starts from; 0 = whole trajectory')
    p.add_argument('--n-micro', type=int, default=4,
                   help='split each sampler window into this many calls. A window '
                        'is ens*24 rows and 480 rows OOMs on a 6 GB card, so '
                        'ens=20 needs 4 (120 rows, 2.2 GiB).')
    p.add_argument('--micro-batch', type=int, default=None,
                   help='absolute rows-per-call cap, overriding --n-micro')
    p.add_argument('--steps', type=int, default=100,
                   help='Euler steps per sample; must match the inversion')
    p.add_argument('--dirname', default='predict_constant_alpha',
                   help='output directory under data/')
    p.add_argument('--test-pl', action='store_true',
                   help='3-trajectory / ens-5 smoke run into a _smoketest dir')
    p.add_argument('--no-resume', action='store_true',
                   help='recompute trajectories even if their files exist')
    a = p.parse_args()

    perform_forecasting(
        eval_traj_num = a.eval_traj_num,
        ens           = a.ens,
        alpha         = a.alpha,
        anchor        = a.anchor,
        n_micro       = a.n_micro,
        micro_batch   = a.micro_batch,
        steps         = a.steps,
        dirname       = a.dirname,
        test_pl       = a.test_pl,
        resume        = not a.no_resume,
    )
