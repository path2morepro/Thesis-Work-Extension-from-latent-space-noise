import numpy as np
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# Output locations, anchored to the repo instead of the working directory.
# These used to be '../Visual/' and '../Performance/metrics.png', which resolve
# against cwd: launched from the repo root they wrote one level ABOVE the repo,
# so the figures ended up outside it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISUAL_DIR   = PROJECT_ROOT / 'Visual'
PERF_DIR     = PROJECT_ROOT / 'Performance'


# ============================================================================
# Metric Computation
# ============================================================================

def compute_rmse(preds_mean: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """
    Compute RMSE of ensemble mean vs ground truth.

    Args:
        preds_mean : shape (rollout_steps, C, H, W) or (C, H, W)
        truth      : same shape as preds_mean

    Returns:
        rmse : shape (rollout_steps,) or scalar
    """
    se = (preds_mean - truth) ** 2

    # If input is 4D, average over last 3 dims (C, H, W)
    if se.ndim == 4:
        return np.sqrt(se.mean(axis=(1, 2, 3)))
    else:
        return np.sqrt(se.mean())


def compute_crps(
    preds: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """
    Compute Continuous Ranked Probability Score (energy form).

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]

    where X, X' are ensemble members and y is the truth.

    Args:
        preds : shape (rollout_steps, n_ensemble, C, H, W) or similar
        truth : shape (rollout_steps, C, H, W) or similar

    Returns:
        crps : shape (rollout_steps,) or scalar
    """
    # E[|X - y|]
    mae = np.abs(preds - truth[:, np.newaxis]).mean(axis=1)  # ensemble dim removed

    # E[|X - X'|] via sort trick
    n = preds.shape[1]  # ensemble size
    s = np.sort(preds, axis=1)
    w = 2 * np.arange(1, n + 1) - n - 1  # (n,)

    # Reshape w for broadcasting depending on preds dimensionality
    if preds.ndim == 5:  # (steps, ensemble, C, H, W)
        w = w.reshape(1, n, 1, 1, 1)
    elif preds.ndim == 4:  # (ensemble, C, H, W)
        w = w.reshape(n, 1, 1, 1)
    elif preds.ndim == 3:  # (ensemble, H, W)
        w = w.reshape(n, 1, 1)
    else:
        w = w.reshape(n)

    spread = (w * s).sum(axis=1) / (n * (n - 1))
    crps_field = mae - spread

    # Average over spatial dims depending on input shape
    if crps_field.ndim == 4:
        return crps_field.mean(axis=(1, 2, 3))  # (steps,)
    elif crps_field.ndim == 3:
        return crps_field.mean(axis=(1, 2))  # scalar or per-step
    else:
        return np.mean(crps_field)


def compute_ssr(preds: np.ndarray, rmse: np.ndarray) -> np.ndarray:
    N = preds.shape[1]  # ensemble size
    spread = np.sqrt(preds.var(axis=1, ddof=1).mean(axis=(1,2,3)))  # unbiased
    return np.sqrt((N+1)/N) * spread / (rmse + 1e-8)        # with correction



def print_metrics(
    lead_times: np.ndarray,
    rmse: np.ndarray,
    crps: np.ndarray,
    ssr: np.ndarray,
) -> None:
    """Pretty-print evaluation results."""
    print(f'\n{"Lead":>5}  {"RMSE":>8}  {"CRPS":>8}  {"SSR":>8}')
    for t, r, c, s in zip(lead_times, rmse, crps, ssr):
        print(f'{int(t):5d}  {r:8.4f}  {c:8.4f}  {s:8.4f}')




def visualize_results(results, out_dir=VISUAL_DIR):
    """
    Save one GIF per sample, one frame per forecast hour.

    arci already returns its output flat over lead — frame i is +(i+1) h, with
    the AR re-anchorings (every 24 h) just showing up as kinks in the sequence.
    There is no window axis to unfold any more.

    Args:
        results : list of (predicts, truth) — one tuple per trajectory
                  predicts : Tensor (n_lead, ens, levels, H, W)
                  truth    : Tensor (n_lead,      levels, H, W)
        out_dir : directory for output GIFs (created if absent)

    Output:  ../Visual/sample00.gif,  sample01.gif, ...
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for sample_idx, (predicts, truth) in enumerate(results):
        n_frames = predicts.shape[0]

        # ens 0, level 0 -> (n_frames, H, W)
        pred_frames  = predicts[:, 0, 0]
        truth_frames = truth[:, 0]

        vmax = max(pred_frames.abs().max(), truth_frames.abs().max()).item()
        kw   = dict(cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].set_title('Truth')
        axes[1].set_title('Pred (ens 0, level 0)')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])

        im_t = axes[0].imshow(truth_frames[0].numpy(), **kw)
        im_p = axes[1].imshow(pred_frames[0].numpy(),  **kw)
        plt.colorbar(im_t, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im_p, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()

        def update(i, _im_t=im_t, _im_p=im_p,
                   _tf=truth_frames, _pf=pred_frames):
            _im_t.set_data(_tf[i].numpy())
            _im_p.set_data(_pf[i].numpy())
            fig.suptitle(
                f'Sample {sample_idx}  |  +{i + 1}h',
                fontsize=10, y=1.02,
            )
            return [_im_t, _im_p]

        anim = FuncAnimation(fig, update, frames=n_frames, interval=200)

        path = out / f'sample{sample_idx:02d}.gif'
        anim.save(str(path), writer='pillow', fps=5)
        plt.close(fig)
        print(f'Saved → {path}')



class MetricAccumulator:
    """compute_metrics(), but fed one trajectory at a time.

    compute_metrics takes the whole `results` list, which means holding every
    forecast in RAM at once: at 50 trajectories x (99, 20, 2, 64, 64) float32
    that is 3.2 GiB, and this laptop has about 4 GiB free. Every quantity it
    computes is a plain sum over trajectories, so accumulating as each one
    finishes gives bit-identical numbers for 65 MiB of peak instead.

    Usage:
        acc = MetricAccumulator()
        for predicts, truth in ...:
            acc.add(predicts, truth)
        metrics = acc.finalize()        # same dict compute_metrics returns
    """

    def __init__(self):
        self.n_ens = None
        self._rmse = np.zeros(0)
        self._spread = np.zeros(0)
        self._crps = np.zeros(0)
        self._counts = np.zeros(0)

    def _grow(self, n):
        if n <= self._rmse.size:
            return
        pad = n - self._rmse.size
        for name in ('_rmse', '_spread', '_crps', '_counts'):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(pad)]))

    def add(self, predicts, truth):
        """predicts (n_lead, ens, C, H, W), truth (n_lead, C, H, W)."""
        p = predicts.numpy() if hasattr(predicts, 'numpy') else np.asarray(predicts)
        t = truth.numpy() if hasattr(truth, 'numpy') else np.asarray(truth)

        if self.n_ens is None:
            self.n_ens = p.shape[1]
        elif self.n_ens != p.shape[1]:
            raise ValueError(
                f'ensemble size changed mid-run: {self.n_ens} -> {p.shape[1]}'
            )

        T = p.shape[0]
        self._grow(T)
        # Identical expressions to compute_metrics, so the two agree exactly.
        self._rmse[:T]   += compute_rmse(p.mean(axis=1), t)
        self._spread[:T] += np.sqrt(p.var(axis=1, ddof=1).mean(axis=(1, 2, 3)))
        self._crps[:T]   += compute_crps(p, t)
        self._counts[:T] += 1

    def finalize(self):
        if self.n_ens is None:
            raise ValueError('no trajectories were added')
        counts = self._counts
        rmse   = self._rmse / counts
        spread = self._spread / counts
        crps   = self._crps / counts
        n      = self.n_ens
        return {
            'rmse':       rmse,
            'crps':       crps,
            'ssr':        np.sqrt((n + 1) / n) * spread / (rmse + 1e-8),
            'spread':     spread,
            'counts':     counts,
            'lead_times': np.arange(1, counts.size + 1),
        }


def compute_metrics(results):
    """Average RMSE / CRPS / SSR over trajectories, per forecast hour.

    Args:
        results : list of (predicts, truth) from arci
                  predicts : Tensor (n_lead, ens, levels, H, W)
                  truth    : Tensor (n_lead,      levels, H, W)

    arci returns its output already flat over lead, so there is no window axis
    to collapse here. `counts` is kept because a shorter trajectory (or a later
    anchor) gives fewer leads, and those entries must not drag the mean down.
    """
    max_frames = max(p.shape[0] for p, t in results)
    n_ens      = results[0][0].shape[1]

    rmse_sum   = np.zeros(max_frames)
    spread_sum = np.zeros(max_frames)
    crps_sum   = np.zeros(max_frames)
    counts     = np.zeros(max_frames)

    for predicts, truth in results:
        T = predicts.shape[0]
        p = predicts.numpy()
        t = truth.numpy()

        rmse_sum[:T]   += compute_rmse(p.mean(axis=1), t)
        spread_sum[:T] += np.sqrt(p.var(axis=1, ddof=1).mean(axis=(1,2,3)))
        crps_sum[:T]   += compute_crps(p, t)
        counts[:T]     += 1

    rmse_mean   = rmse_sum   / counts
    spread_mean = spread_sum / counts
    crps_mean   = crps_sum   / counts
    ssr         = np.sqrt((n_ens + 1) / n_ens) * spread_mean / (rmse_mean + 1e-8)

    return {
        'rmse':       rmse_mean,
        'crps':       crps_mean,
        'ssr':        ssr,
        'counts':     counts,
        'lead_times': np.arange(1, max_frames + 1),
    }


def plot_metrics(metrics, out_path=None):
    out_path = Path(out_path) if out_path is not None else PERF_DIR / 'metrics.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lead_times = metrics['lead_times']
    counts     = metrics['counts']
    valid      = counts > 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(lead_times[valid], metrics['rmse'][valid], color='steelblue')
    axes[0].set_title('RMSE'); axes[0].set_xlabel('Lead time (h)'); axes[0].grid(alpha=0.3)

    axes[1].plot(lead_times[valid], metrics['crps'][valid], color='darkorange')
    axes[1].set_title('CRPS'); axes[1].set_xlabel('Lead time (h)'); axes[1].grid(alpha=0.3)

    axes[2].plot(lead_times[valid], metrics['ssr'][valid], color='seagreen')
    axes[2].axhline(1.0, color='red', linestyle='--', lw=1, label='ideal')
    axes[2].set_title('SSR'); axes[2].set_xlabel('Lead time (h)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle(f'Metrics over lead time — {int(counts[0])} samples', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved metrics plot -> {out_path}')
    plt.close(fig)


def compute_scalar_alpha(Z):
    """
    Lag-1 correlation corr(z_{t-1}, z_t).

    Args:
        Z: array (N, T, 2, 64, 64)

    Returns:
        alpha in [-1, 1]
    """
    Z = np.asarray(Z, dtype=np.float64).reshape(Z.shape[0], Z.shape[1], -1)
    Zp, Zc = Z[:, :-1], Z[:, 1:]                   # (N, T-1, D)

    cov = np.einsum("ntx,ntx->nt", Zp, Zc)
    vp  = np.einsum("ntx,ntx->nt", Zp, Zp)
    vc  = np.einsum("ntx,ntx->nt", Zc, Zc)

    covs = np.sum(cov, axis=(0,1))
    vps = np.sum(vp, axis=(0,1))
    vcs = np.sum(vc, axis=(0,1))

    return covs / np.sqrt(vps * vcs)              # scalar


def load_invert(dir):
    """Load every inverted-latent file in one directory.

    Args:
        dir_ : directory of invert_traj{T}initial{A}.npy files. Each file is ONE window
               = one (trajectory T, anchor A) pair holding leads tau = 1..24.

    Returns:
        Z     : (N, L, C, H, W) float32 — latents, already in normalised model units.
                The dataset divided by NORM_STD before the inversion; never divide again.
        traj  : (N,) int — trajectory id per window. LOCAL to the val split (files[400:]),
                so index SQGLeadTimeDataset(split='val').data with it, never a 500-file set.
        anch  : (N,) int — anchor frame per window; lead k pairs with frame anch + 1 + k.
    """
    files = natsorted(Path(dir).glob('invert_traj*initial*.npy'))
    assert files, f'no invert_traj*initial*.npy in {dir}'
    Z    = np.stack([np.load(f).astype(np.float32) for f in files])
    meta = np.array([[int(m.group(1)), int(m.group(2))] for f in files
                     for m in [re.search(r'invert_traj(\d+)initial(\d+)', f.name)]])
    return Z, meta[:, 0], meta[:, 1]