"""Round trip: data -> latent -> data, one arm per (steps, order).

The backward half is NOT done here. generate_invert_data writes the latents; this
script reads one of its directories, pushes every window forward through the SAME
integrator at the SAME step count, and pairs the result with the truth it came from.
Matching the two halves is the point: a mixed pair (invert with RK4, return with
Euler) measures neither integrator.

Nothing is chained. Each lead is its own map Phi(x0, tau) from the SAME anchor, so
lead 24 is not built from lead 23. A rising curve is therefore error accumulating
along one ODE solve, NOT error feeding back on itself.

Everything is in NORMALISED model units: the latents already are, and dividing the
truth by NORM_STD puts it on the same axis.

WHAT THIS DOES AND DOES NOT SHOW. Backward-then-forward with one solver at one step
count partially cancels its own local error, so a small residual here does not prove
the inversion is faithful -- it proves the composition is self-consistent. Whether
the latents are right is a different question, answered by their per-lead variance.

Run from the repository root:

    PYTHONPATH=. python Analysis/round_trip.py --split test --steps 100 --order 1
    PYTHONPATH=. python Analysis/round_trip.py --split test --steps 500 --order 4
    PYTHONPATH=. python Analysis/round_trip.py --plot          # figure, after the runs
"""
import argparse
import hashlib
import json
import re
import time

import numpy as np
import torch

from Condition_model.samplers import CondSampler
from Condition_model.train import SAVE_PATH
from Condition_model.dataset import (NORM_STD, DATA_PATH, DATA_TRAIN, DATA_VAL,
                                     DATA_TEST)
from Condition_model.utils import compute_rmse
# Imported, not copied. The old version kept its own INVERT_DIRS dict keyed by
# steps alone, which could not name an arm (100 Euler and 500 RK4 are different
# directories) and drifted out of step with the writer the moment the naming
# convention changed.
from Condition_model.generate_invert_data import out_dirs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

L      = 24                                              # direct leads in a window
LEADS  = np.arange(1, L + 1)                              # 1..24, lead 0 is not modelled
LABELS = torch.arange(1, L + 1, dtype=torch.float32) / L  # what the model is conditioned on

SPLIT_DIRS = {'train': DATA_TRAIN, 'val': DATA_VAL, 'test': DATA_TEST}

RESULT_DIR = DATA_PATH.parent / 'Analysis' / 'Result'
RT_ROOT    = DATA_PATH / 'round_trip'

ORDER_NAME = {1: 'euler', 2: 'heun', 4: 'rk4'}


def arm_dirname(steps, order):
    """data/round_trip/<this>. Explicit, unlike the invert convention where the
    baseline (100, 1) carries no suffix: here every arm names its integrator."""
    return f'{ORDER_NAME[order]}_steps={steps}'


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def build_traj_index():
    """Map global trajectory id -> (split name, file path), over all three splits.

    An invert filename carries a GLOBAL trajectory id, but which folder that id lives
    in depends on where the split boundaries currently sit (train 0..1199,
    val 1200..1349, test 1350..1499). Resolving the id is what kills the old class of
    bug: indexing a file LIST by an id silently hands back whatever trajectory happens
    to sit at that POSITION, which is a different trajectory as soon as the folder
    does not start at id 0.

    Raises if one id appears in two splits -- that would make every pairing ambiguous.
    """
    index = {}
    for split, d in SPLIT_DIRS.items():
        if not d.is_dir():
            continue
        for f in sorted(d.glob('sqg_N64_*hrly_*.npy')):
            m = re.search(r'_(\d+)\.npy$', f.name)
            if m is None:
                continue
            tid = int(m.group(1))
            if tid in index:
                raise RuntimeError(f'trajectory id {tid} exists in two splits: '
                                   f'{index[tid][1]} and {f}')
            index[tid] = (split, f)
    assert index, f'no trajectory files under any of {list(SPLIT_DIRS.values())}'
    return index


class TruthLoader:
    """Anchor frame and truth window for a (trajectory id, anchor) pair.

    Files are opened memmapped and cached, so a trajectory that contributes several
    windows is opened once and only the ~25 frames actually needed are paged in.
    """

    def __init__(self, index, std=NORM_STD):
        self.index = index
        self.std   = float(std)
        self._mm   = {}

    def _traj(self, tid):
        mm = self._mm.get(tid)
        if mm is None:
            if tid not in self.index:
                raise KeyError(f'trajectory {tid} is in no split folder -- the latents '
                               f'in data/invert refer to a trajectory not on disk')
            mm = np.load(self.index[tid][1], mmap_mode='r')
            self._mm[tid] = mm
        return mm

    def split_of(self, tid):
        return self.index[tid][0] if tid in self.index else None

    def window(self, tid, anchor, n_lead=L):
        """Returns (x0, truth) = (C,H,W) and (n_lead,C,H,W), both normalised.

        Lead k pairs with frame anchor + k, so the window is frames
        anchor+1 .. anchor+n_lead. The anchor frame itself is the conditioning input.
        """
        t = self._traj(tid)
        if anchor + n_lead >= t.shape[0]:
            raise IndexError(f'traj {tid} has {t.shape[0]} frames; anchor {anchor} + '
                             f'{n_lead} leads runs past the end')
        x0    = np.asarray(t[anchor], dtype=np.float32) / self.std
        truth = np.asarray(t[anchor + 1: anchor + 1 + n_lead], dtype=np.float32) / self.std
        return torch.from_numpy(x0), torch.from_numpy(truth)


def run_arm(split='test', steps=100, order=1, limit=None, resume=True,
            model_path=SAVE_PATH, device=DEVICE, outdir=None):
    """Forward-integrate one invert directory and write the pairs + metrics.

    The window set is whatever is IN the invert directory -- the invert stage is
    the single knob for how many windows an arm has. Add trajectories there,
    re-run this, and only the new windows cost anything.

    Writes into data/round_trip/<arm>/:
        traj{T}initial{A}.npy   stack([pred, truth]) -> (2, L, C, H, W)
        metrics.npz             per-window per-lead RMSE, tids, anchors, leads
        meta.json               model, integrator, source directory, provenance

    Returns the output directory.
    """
    assert order in ORDER_NAME, f'order must be one of {sorted(ORDER_NAME)}'
    assert split in SPLIT_DIRS, f'split must be one of {sorted(SPLIT_DIRS)}'

    src = out_dirs(split, steps, order)[0]
    if not src.is_dir():
        raise SystemExit(
            f'no latents at {src}\n'
            f'Invert that arm first:\n'
            f'  python -m Condition_model.generate_invert_data '
            f'--split {split} --steps {steps} --order {order} --anchors 1 --n-traj N'
        )
    names = sorted(p.name for p in src.glob('invert_traj*initial*.npy'))
    if not names:
        raise SystemExit(f'{src} exists but holds no invert_traj*initial*.npy')
    if limit:
        names = names[:limit]

    save_dir = RT_ROOT / (outdir or arm_dirname(steps, order))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Provenance guard, same contract as the invert stage: resume ADDS to what is
    # already here, so refuse to mix a second model or a second integrator into one
    # folder. Without this the arm directory would silently hold two generations
    # and metrics.npz would average across them.
    prev = save_dir / 'meta.json'
    if resume and prev.exists():
        pm  = json.loads(prev.read_text())
        bad = []
        old = pm.get('model', {}).get('sha256')
        new = _sha256(model_path)
        if old and old != new:
            bad.append(f'model {old[:16]}... != {new[:16]}...')
        for k, v in (('steps', steps), ('order', order), ('split', split)):
            if pm.get(k) not in (None, v):
                bad.append(f'{k} {pm.get(k)} != {v}')
        if bad:
            raise SystemExit(f'{save_dir} already holds a different round trip:\n'
                             + ''.join(f'  - {b}\n' for b in bad)
                             + 'Move it aside, or pass --no-resume.')

    loader = TruthLoader(build_traj_index())

    # Say out loud where the truth comes from. Latents inverted under an older split
    # can name ids that now sit in data/train, which would make this not held out.
    tids_all = sorted({int(re.search(r'invert_traj(\d+)', n).group(1)) for n in names})
    where = {}
    for tid in tids_all:
        where.setdefault(loader.split_of(tid), []).append(tid)
    print(f'{arm_dirname(steps, order)}: {len(names)} windows from {src}')
    print(f'  integrator {ORDER_NAME[order]} steps={steps}  device={device}')
    print('  truth from: ' + ', '.join(f'{k or "MISSING"}={len(v)} traj'
                                       for k, v in sorted(where.items(), key=lambda x: str(x[0]))))
    if 'train' in where:
        print(f'  NOTE {len(where["train"])} of these trajectories are in data/train '
              f'under the current split, so this is NOT held out.')

    # Resume by file, and accumulate metrics as we go rather than holding every
    # window: 150 windows of (2, 24, 2, 64, 64) float32 is 236 MB, and the only
    # thing needed from a window is its per-lead RMSE.
    todo = [n for n in names
            if not (resume and (save_dir / _pair_name(n)).exists())]
    print(f'  {len(todo)} to compute, {len(names) - len(todo)} already on disk')

    smp = invert_fn = None
    if todo:
        smp = CondSampler(model_path=model_path, device=device, steps=steps)
        invert_fn = {1: smp.sample, 2: smp.heunSample, 4: smp.rk4Sample}[order]

    rows, t0 = [], time.time()
    for i, name in enumerate(names, 1):
        m = re.search(r'invert_traj(\d+)initial(\d+)', name)
        tid, anchor = int(m.group(1)), int(m.group(2))
        out = save_dir / _pair_name(name)

        if out.exists() and resume:
            pair  = np.load(out, mmap_mode='r')
            pred, truth = pair[0], pair[1]
        else:
            z = torch.from_numpy(np.load(src / name).astype(np.float32))   # (L,C,H,W)
            assert z.shape[0] == L, f'{name} holds {z.shape[0]} leads, expected {L}'
            x0, truth_t = loader.window(tid, anchor)
            # Forward, with the integrator this arm inverted with. invert=False.
            pred_t = invert_fn(z0=z,
                               x_t=x0.unsqueeze(0).expand(L, -1, -1, -1),
                               lead_times=LABELS)
            pred, truth = pred_t.numpy(), truth_t.numpy()
            np.save(out, np.stack([pred, truth], axis=0))
            print(f'  [{i}/{len(names)}] traj{tid} anchor{anchor}  '
                  f'rmse t=1 {compute_rmse(pred[:1], truth[:1])[0]:.6f}', flush=True)

        rows.append((tid, anchor, compute_rmse(np.asarray(pred), np.asarray(truth))))

    tids    = np.array([r[0] for r in rows], dtype=np.int64)
    anchors = np.array([r[1] for r in rows], dtype=np.int64)
    rmse    = np.stack([r[2] for r in rows]).astype(np.float64)   # (n_windows, L)

    np.savez(save_dir / 'metrics.npz', rmse=rmse, tids=tids, anchors=anchors,
             leads=LEADS, steps=steps, order=order, split=split)

    meta = {
        'created':   time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'split':     split,
        'steps':     steps,
        'order':     order,
        'integrator': ORDER_NAME[order],
        'arm':       arm_dirname(steps, order),
        'n_windows': len(names),
        'n_computed_this_run': len(todo),
        'source_invert_dir': str(src),
        'source_invert_meta': json.loads((src / 'meta.json').read_text())
                              if (src / 'meta.json').exists() else None,
        'units':     'normalised (truth / NORM_STD=%s)' % NORM_STD,
        'norm_std':  NORM_STD,
        'elapsed_seconds': round(time.time() - t0, 1),
        'model': {'path': str(model_path), 'sha256': _sha256(model_path)},
        'torch': torch.__version__,
        'device': str(device),
    }
    (save_dir / 'meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True))

    mean = rmse.mean(axis=0)
    print(f'  RMSE  t=1 {mean[0]:.6f}   t=24 {mean[L-1]:.6f}   '
          f'growth {(rmse[:, L-1] / rmse[:, 0]).mean():.2f}x')
    print(f'  -> {save_dir}  ({len(names)} windows, '
          f'{meta["elapsed_seconds"]:.0f}s this run)')
    return save_dir


def _pair_name(invert_name):
    """invert_traj1350initial29.npy -> traj1350initial29.npy"""
    return invert_name.replace('invert_', '', 1)


def plot(arms=None):
    """One axis, every arm that has a metrics.npz. Figure only -- no data is made.

    No forecast reference. The old forecast_reference() read data/predict_constant_alpha,
    which does not exist (forecasting has not run), and comparison arms are out of
    scope for this test.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    found = []
    for d in sorted(RT_ROOT.glob('*/metrics.npz')) if RT_ROOT.is_dir() else []:
        if arms and d.parent.name not in arms:
            continue
        found.append((d.parent.name, np.load(d)))
    if not found:
        raise SystemExit(f'no metrics.npz under {RT_ROOT} -- run the arms first')

    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    print(f"{'arm':<20}{'n':>5}{'RMSE t=1':>12}{'RMSE t=24':>12}{'growth':>10}")
    for i, (name, z) in enumerate(found):
        r = z['rmse']
        ax.plot(z['leads'], r.mean(0), 'o-', ms=4, lw=1.8,
                color=cmap(0.05 + 0.7 * i / max(len(found) - 1, 1)), label=name)
        print(f'{name:<20}{r.shape[0]:>5}{r.mean(0)[0]:>12.6f}'
              f'{r.mean(0)[-1]:>12.6f}{(r[:, -1] / r[:, 0]).mean():>9.2f}x')
    ax.set_yscale('log')
    ax.set_xlabel('lead time [h]')
    ax.set_ylabel('RMSE [normalised units]')
    ax.set_title('round-trip error, first 24 h', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / 'round_trip_rmse.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {path}')
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='test', choices=sorted(SPLIT_DIRS))
    p.add_argument('--steps', type=int, default=100,
                   help='MUST match the invert directory this arm reads')
    p.add_argument('--order', type=int, default=1, choices=sorted(ORDER_NAME),
                   help='1 Euler, 2 Heun, 4 RK4. Also must match.')
    p.add_argument('--limit', type=int, default=None,
                   help='use at most N windows (default: every one in the invert dir)')
    p.add_argument('--outdir', default=None,
                   help='directory name under data/round_trip/ (default: '
                        '<integrator>_steps=<steps>)')
    p.add_argument('--no-resume', action='store_true',
                   help='recompute windows even if their pair file exists')
    p.add_argument('--plot', action='store_true',
                   help='only draw the figure from existing metrics.npz, compute nothing')
    a = p.parse_args()
    if a.plot:
        plot()
    else:
        run_arm(split=a.split, steps=a.steps, order=a.order, limit=a.limit,
                resume=not a.no_resume, outdir=a.outdir)
