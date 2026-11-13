"""
Generate inverted latents: run the ODE backwards from data to noise.

For a trajectory and an anchor a, take the 24 true future frames
traj[a+1 .. a+24] and invert each one back through the conditional flow,
conditioned on x_a = traj[a] and its own lead time. The result is the latent
that would reproduce that frame exactly if pushed forward again.

Output: one file per (trajectory, anchor), shape (max_lead, C, H, W), under

    data/invert/from_{split}/invert_traj{id}initial{a}.npy

plus a copy of the FIRST anchor of every trajectory under

    data/invert/distribution_check_{split}/

That second set holds exactly one window per trajectory, so no two of its
windows share a frame. Within `from_{split}` the 8 windows of a trajectory
overlap heavily (anchors 30 and 35 share 19 of their 24 target frames), which
is fine for pooling a correlation but wrong for any check that assumes
independent draws -- hence the separate, genuinely independent set.

Both directories also get a meta.json recording the model, seed and settings
that produced them. Without it nothing on disk says which model the latents
came from, which is how the previous generation of invert files (ids 400+,
from the retired data_500traj layout) became unusable.
"""

import argparse
import hashlib
import json
import re
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from .samplers import CondSampler
from .train import SAVE_PATH
from .dataset import SQGDataset, DATA_PATH, DATA_TRAIN, DATA_VAL, DATA_TEST


MAX_LEAD = 24

OUT_ROOT = DATA_PATH / 'invert'

SPLIT_DIRS = {'train': DATA_TRAIN, 'val': DATA_VAL, 'test': DATA_TEST}

# One constant so the name lives in exactly one place.
CHECK_DIRNAME = 'distribution_check_{split}'


# The (steps, order) that the plain, unsuffixed directory names belong to.
# data/invert/from_val already holds 1200 windows made with these, and the
# alpha in use (0.7133258747) was measured on them, so that path must keep
# meaning exactly one thing.
BASELINE = (100, 1)


def _suffix(steps, order):
    """'' for the baseline integrator, '_steps=S_order=O' for anything else.

    Without this every (steps, order) wrote to the same folder under the same
    invert_traj{id}initial{a}.npy names, so a 300-step Heun run would land on
    top of the 100-step Euler set -- and the resume check would SKIP any
    window whose name was already there, leaving a silent mixture of two
    integrators that load_invert() would stack into one array.
    """
    return '' if (steps, order) == BASELINE else f'_steps={steps}_order={order}'


def out_dirs(split, steps=100, order=1, tag=None, outdir=None):
    """(main, distribution-check) output directories for a run.

    `tag` puts a run in its own namespace, always carrying steps and order.
    Use it for anything that is a side experiment rather than THE set for a
    split: an integrator ladder run at steps=100 order=1 would otherwise be
    written straight into data/invert/from_{split}, adding its windows to the
    1200 already there and overwriting the meta.json that records how those
    1200 were made. The provenance guard would not catch it either -- model,
    steps and order all match; only the anchor plan differs.
    """
    if outdir:
        # Verbatim name, for a run you want to find by a name you chose.
        # It carries no steps/order, so the ONLY record of how these latents
        # were made is meta.json -- which is also what the resume guard reads,
        # so a folder cannot be extended with a different integrator.
        return OUT_ROOT / outdir, OUT_ROOT / f'{outdir}_check'
    if tag:
        stem = f'{tag}_{split}_steps={steps}_order={order}'
        return OUT_ROOT / stem, OUT_ROOT / f'{tag}_check_{split}_steps={steps}_order={order}'
    sfx = _suffix(steps, order)
    return (OUT_ROOT / f'from_{split}{sfx}',
            OUT_ROOT / (CHECK_DIRNAME.format(split=split) + sfx))


def existing_anchors(save_dir, traj_id):
    """Anchors already on disk for this trajectory, read off the filenames.

    The filename is the only record of a window's anchor that survives
    independently of meta.json, and that matters because these folders get
    populated by hand: data/invert/from_val_rk4 was seeded by copying 30
    windows out of data/invert/RK4_distribution_check and has no meta.json at
    all. Reading the directory means a folder is described by what is in it,
    not by what some earlier RNG would have drawn.
    """
    rx = re.compile(rf'^invert_traj{int(traj_id)}initial(\d+)\.npy$')
    return sorted(int(m.group(1))
                  for f in save_dir.glob(f'invert_traj{int(traj_id)}initial*.npy')
                  for m in [rx.match(f.name)] if m)


def choose_anchors(T, k, max_lead, seed, traj_id, existing=()):
    """The k anchors for one trajectory: what is already on disk, then more.

    Anchor a is valid iff a + max_lead <= T - 1, so the candidates are
    0 .. T-1-max_lead -- 76 of them for a 100-frame trajectory at lead 24.

    NESTED IN k, which is the whole point. The draw is a permutation of the
    candidates from an RNG seeded by (seed, traj_id), and the k anchors are its
    first k. So k=1's set is a subset of k=2's, which is a subset of k=3's, and
    raising --anchors on a finished folder ADDS windows instead of replacing
    them.

    The previous version drew rng.choice(size=k) from ONE RNG shared across all
    trajectories, which is not nested and not even stable in k: at seed 0,
    trajectory 1200 drew anchor 64 at k=1 but [58, 5, 1, ...] at k=8. Going
    from --anchors 1 to --anchors 2 would therefore have computed 300 brand-new
    windows and left the 150 already on disk orphaned under names that nothing
    would ever ask for again. Seeding per trajectory also makes the draw
    independent of --n-traj and of iteration order.

    `existing` comes first and is never dropped, so a folder filled under the
    old scheme -- or by hand -- keeps every file it has. Those anchors were
    still uniform draws over the same candidate range, just from a different
    source, so the pooled set stays uniform.

    Distinct by construction: a permutation cannot repeat. A repeat would not
    be harmless, because two identical windows map to the SAME output filename,
    so the second save would silently overwrite the first and the run would end
    with fewer files than it reported.

    Deliberately NOT sorted: element 0 is a uniform draw over the whole range,
    which is what the distribution-check copy wants. Taking min() instead would
    bias that set towards early, spin-up frames (the expected minimum of 8
    draws from 76 is about 7.6).
    """
    last = T - 1 - max_lead
    assert last >= 0, f'trajectory of {T} frames is too short for lead {max_lead}'
    keep  = [int(a) for a in existing if 0 <= int(a) <= last]
    pool  = np.random.default_rng([int(seed), int(traj_id)]).permutation(last + 1)
    have  = set(keep)
    fresh = [int(a) for a in pool if int(a) not in have]
    return keep + fresh[:max(0, min(k, last + 1) - len(keep))]


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def _write_meta(path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def perform_invert(
    split='val',
    n_traj=None,
    anchors_per_traj=8,
    batch_pairs=5,
    steps=100,
    order=1,
    max_lead=MAX_LEAD,
    model_path=SAVE_PATH,
    seed=0,
    device=None,
    use_compile=False,
    resume=True,
    tag=None,
    outdir=None,
):
    """
    Args:
        split            : 'val' / 'test' / 'train'. Picks both the input
                           directory and the output name (from_{split}).
        n_traj           : trajectories to use; None means all of them
        anchors_per_traj : anchors drawn per trajectory
        batch_pairs      : (traj, anchor) windows per forward pass. Each one
                           costs max_lead samples, so the real batch is
                           batch_pairs * max_lead. 5 -> 120 rows -> 2.2 GiB.
                           Do not raise this for speed: throughput is flat
                           (measured 1.83 rows/s at 120, 1.76 at 240) because
                           the GPU is already saturated, and 480 rows OOMs on
                           a 6 GB card.
        steps            : Euler steps for the backward ODE
        order            : 1 Euler, 2 Heun, 4 RK4
        seed             : anchor draw is reproducible given this
        use_compile      : torch.compile the network. OFF because it was
                           MEASURED SLOWER here: 1.67 vs 1.85 rows/s (0.90x)
                           and 2716 vs 2179 MiB. Training's 1.71x came from
                           fusing forward+backward+optimizer into one graph;
                           a single Euler step is 650 ms of pure compute for
                           120 rows, so there is no overhead left to fuse out.
                           Numerically it is safe (max|diff| 2.9e-4, pooled
                           alpha identical to 8 dp) -- just not faster.
        resume           : skip windows whose output file already exists
        tag              : namespace this run's output directories, so a side
                           experiment cannot be written into the main set for
                           a split. See out_dirs().
        outdir           : exact directory name under data/invert/, overriding
                           both the default naming and `tag`.

    Returns the directory written to.
    """
    assert order in (1, 2, 4), 'order must be 1 (Euler), 2 (Heun) or 4 (RK4)'
    assert split in SPLIT_DIRS, f'split must be one of {sorted(SPLIT_DIRS)}'

    device   = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = SPLIT_DIRS[split]
    ds       = SQGDataset(data_dir)
    n_traj   = len(ds) if n_traj is None else min(n_traj, len(ds))

    save_dir, check_dir = out_dirs(split, steps, order, tag, outdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # With one anchor per trajectory the main set ALREADY has exactly one
    # window per trajectory, so no two of its windows share a frame and it is
    # itself the independent set. Copying it to a second directory would just
    # duplicate every file.
    dup_check = anchors_per_traj > 1
    if dup_check:
        check_dir.mkdir(parents=True, exist_ok=True)

    def paths_for(i, a, is_check):
        name = f'invert_traj{ds.traj_ids[i]}initial{a}.npy'
        return save_dir / name, (check_dir / name if (is_check and dup_check)
                                 else None)

    # Provenance guard. Resume means we ADD to whatever is already in the
    # folder, so if those files came from a different checkpoint the two
    # generations would be silently pooled -- the exact failure that made the
    # old invert_traj400* set useless. meta.json makes it detectable, so check
    # it before writing anything.
    prev = save_dir / 'meta.json'
    if resume and prev.exists():
        pm = json.loads(prev.read_text())
        new_sha = _sha256(model_path)
        bad = []
        old_sha = pm.get('model', {}).get('sha256')
        if old_sha and old_sha != new_sha:
            bad.append(f'model {old_sha[:16]}... != {new_sha[:16]}...')
        # steps/order are in the directory name for non-baseline runs, but
        # check anyway: the baseline name carries neither.
        if pm.get('steps') not in (None, steps):
            bad.append(f'steps {pm.get("steps")} != {steps}')
        if pm.get('order') not in (None, order):
            bad.append(f'order {pm.get("order")} != {order}')
        if bad:
            raise SystemExit(
                f'{save_dir} already holds latents made differently:\n'
                + ''.join(f'  - {b}\n' for b in bad)
                + 'Move the directory aside, or pass --no-resume to overwrite it.'
            )

    # Windows but no meta.json: the folder was filled by hand (from_val_rk4 was
    # seeded with 30 files copied out of RK4_distribution_check). Nothing on
    # disk says which model or integrator made them, so the guard above had
    # nothing to compare. Say so rather than adopting them silently -- the
    # meta.json written at the end will claim THIS run's settings for all of
    # them.
    n_on_disk = len(list(save_dir.glob('invert_traj*initial*.npy')))
    if resume and n_on_disk and not prev.exists():
        print(f'  NOTE: {save_dir} holds {n_on_disk} windows but no meta.json.')
        print(f'        Unverifiable provenance; adopting them as steps={steps} '
              f'order={order}.')

    # Flatten to (position, anchor, is_check) so batching is uniform. is_check
    # marks the first anchor of each trajectory; that window is saved to BOTH
    # directories -- it costs no extra ODE solves, and keeping it in the main
    # set too means the pooled correlation still uses every window.
    #
    # Each trajectory's anchor list starts from the files already in save_dir,
    # so raising --anchors extends the set instead of redrawing it. See
    # choose_anchors().
    plan, adopted = [], 0
    for i in range(n_traj):
        T    = ds[i][0].shape[0]
        tid  = int(ds.traj_ids[i])
        have = existing_anchors(save_dir, tid) if resume else []
        adopted += len(have)
        for j, a in enumerate(choose_anchors(T, anchors_per_traj, max_lead,
                                             seed, tid, have)):
            plan.append((i, int(a), j == 0))

    # Resume: each window is an independent file, so a restart only has to do
    # what is missing. If the main file survived but its check copy did not,
    # copy it across rather than recomputing -- the bytes are identical.
    todo = []
    for i, a, is_check in plan:
        main, chk = paths_for(i, a, is_check)
        if resume and main.exists():
            if chk is not None and not chk.exists():
                shutil.copy2(main, chk)
            continue
        todo.append((i, a, is_check))

    meta = {
        'created':          time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'split':            split,
        'data_dir':         str(data_dir),
        'n_traj':           n_traj,
        'anchors_per_traj': anchors_per_traj,
        'seed':             seed,
        'steps':            steps,
        'order':            order,
        'max_lead':         max_lead,
        'batch_pairs':      batch_pairs,
        'tag':              tag,
        'outdir':           outdir,
        'precision':        'fp32',
        'torch_compile':    bool(use_compile),
        'windows_planned':  len(plan),
        'windows_todo':     len(todo),
        'windows_adopted':  adopted,   # already on disk when this run started
        'model': {
            'path':   str(model_path),
            'bytes':  model_path.stat().st_size,
            'mtime':  time.strftime('%Y-%m-%dT%H:%M:%S',
                                    time.localtime(model_path.stat().st_mtime)),
            'sha256': _sha256(model_path),
        },
        'torch':  torch.__version__,
        'cuda':   torch.version.cuda,
        'device': str(device),
        # trajectory id -> the anchors drawn for it, in draw order
        # (element 0 is the one copied to the distribution-check set)
        'anchors': {str(ds.traj_ids[i]): [a for j, a, _ in plan if j == i]
                    for i in range(n_traj)},
    }

    print(f'Inverting {len(todo)} of {len(plan)} windows from {data_dir}')
    print(f'  steps={steps} order={order} compile={use_compile} device={device}')
    print(f'  -> {save_dir}')
    if dup_check:
        print(f'  -> {check_dir}  ({sum(1 for _, _, c in plan if c)} independent windows)')
    else:
        print(f'  -> (no separate check dir: anchors=1, so all {len(plan)} '
              f'windows are already independent)')
    if len(todo) < len(plan):
        print(f'  resuming: {len(plan) - len(todo)} already on disk')

    if not todo:
        print('nothing to do')
        _write_meta(save_dir / 'meta.json', meta)
        if dup_check:
            _write_meta(check_dir / 'meta.json', meta)
        return save_dir

    sampler   = CondSampler(model_path=model_path, device=device, steps=steps,
                            use_compile=use_compile)
    invert_fn = {1: sampler.sample, 2: sampler.heunSample, 4: sampler.rk4Sample}[order]

    # Lead labels for one window: 1..max_lead, normalised exactly as training
    # normalised them (lead / max_lead).
    labels = torch.arange(1, max_lead + 1, dtype=torch.float32) / max_lead

    t0 = time.time()
    for s in tqdm(range(0, len(todo), batch_pairs), desc='invert'):
        chunk = todo[s:s + batch_pairs]
        B     = len(chunk)

        x0     = torch.stack([ds[i][0][a]                      for i, a, _ in chunk])
        target = torch.stack([ds[i][0][a + 1:a + 1 + max_lead] for i, a, _ in chunk])

        # Batch-major, lead-minor -- the same ordering contract arci uses:
        # x_a repeats each anchor max_lead times, so the labels tile 1..max_lead.
        z = invert_fn(
            z0         = target.reshape(B * max_lead, *target.shape[2:]),
            x_t        = x0.repeat_interleave(max_lead, dim=0),
            lead_times = labels.repeat(B),
            invert     = True,
        )
        z = z.reshape(B, max_lead, *target.shape[2:]).numpy().astype(np.float32)

        for j, (i, a, is_check) in enumerate(chunk):
            main, chk = paths_for(i, a, is_check)
            np.save(main, z[j])
            if chk is not None:
                np.save(chk, z[j])

    meta['elapsed_seconds'] = round(time.time() - t0, 1)
    _write_meta(save_dir / 'meta.json', meta)
    if dup_check:
        _write_meta(check_dir / 'meta.json', meta)

    print(f'Done in {meta["elapsed_seconds"]:.0f}s')
    print(f'  {save_dir}: '
          f'{len(list(save_dir.glob("invert_traj*initial*.npy")))} files')
    if dup_check:
        print(f'  {check_dir}: '
              f'{len(list(check_dir.glob("invert_traj*initial*.npy")))} files')
    return save_dir


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='val', choices=sorted(SPLIT_DIRS),
                   help='which split to invert; also names the output directory')
    p.add_argument('--steps', type=int, nargs='+', default=[100],
                   help='step counts to run')
    p.add_argument('--order', type=int, nargs='+', default=[1], choices=[1, 2, 4])
    p.add_argument('--n-traj', type=int, default=0,
                   help='trajectories to invert (0 = all)')
    p.add_argument('--anchors', type=int, default=8, help='anchors per trajectory')
    p.add_argument('--batch-pairs', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--compile', action='store_true',
                   help='torch.compile the network. Measured 0.90x (SLOWER) on '
                        'this card -- off by default; here to re-measure, not to use.')
    p.add_argument('--tag', default=None,
                   help='namespace the output dirs (e.g. --tag ladder gives '
                        'data/invert/ladder_val_steps=300_order=2). Use it for '
                        'side experiments so they cannot be written into the '
                        'main from_<split> set.')
    p.add_argument('--outdir', default=None,
                   help='exact directory name under data/invert/, overriding '
                        '--tag and the default naming (e.g. RK4_distribution_check)')
    p.add_argument('--no-resume', action='store_true',
                   help='recompute windows even if their file already exists')
    args = p.parse_args()

    for order in args.order:
        for steps in args.steps:
            perform_invert(
                split            = args.split,
                n_traj           = args.n_traj or None,
                anchors_per_traj = args.anchors,
                batch_pairs      = args.batch_pairs,
                steps            = steps,
                order            = order,
                seed             = args.seed,
                use_compile      = args.compile,
                resume           = not args.no_resume,
                tag              = args.tag,
                outdir           = args.outdir,
            )
