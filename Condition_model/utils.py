import numpy as np
from pathlib import Path
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


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




def visualize_results(results, out_dir='../Visual/'):
    """
    Save one GIF per sample. All AR steps are concatenated as sequential frames.

    For a sample with 3 AR steps and 24 direct leads:
        frames 0-23  → AR step 0, hours  1-24
        frames 24-47 → AR step 1, hours 25-48
        frames 48-71 → AR step 2, hours 49-72

    Args:
        results : list of (predicts, truth) — one tuple per data sample
                  predicts : Tensor (n_ar+1, n_lead, ens, levels, H, W)
                  truth    : Tensor (n_ar+1, n_lead,      levels, H, W)
        out_dir : directory for output GIFs (created if absent)

    Output:  ../Visual/sample00.gif,  sample01.gif, ...
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for sample_idx, (predicts, truth) in enumerate(results):
        n_steps, n_lead = predicts.shape[:2]
        n_frames        = n_steps * n_lead
        ar_stride       = n_lead          # last direct lead = AR update stride

        # flatten all AR steps: (n_steps*n_lead, H, W) — ens 0, level 0
        pred_frames  = predicts[:, :, 0, 0].reshape(n_frames, *predicts.shape[-2:])
        truth_frames = truth[:,   :, 0   ].reshape(n_frames, *truth.shape[-2:])

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
            ar_step  = i // n_lead
            lead_idx = i  % n_lead
            abs_hour = ar_step * ar_stride + (lead_idx + 1)
            _im_t.set_data(_tf[i].numpy())
            _im_p.set_data(_pf[i].numpy())
            fig.suptitle(
                f'Sample {sample_idx}  |  AR step {ar_step}  |  +{abs_hour}h',
                fontsize=10, y=1.02,
            )
            return [_im_t, _im_p]

        anim = FuncAnimation(fig, update, frames=n_frames, interval=200)

        path = out / f'sample{sample_idx:02d}.gif'
        anim.save(str(path), writer='pillow', fps=5)
        plt.close(fig)
        print(f'Saved → {path}')



def compute_metrics(results):
    max_frames = max(p.shape[0] * p.shape[1] for p, t in results)
    n_ens      = results[0][0].shape[2]

    rmse_sum   = np.zeros(max_frames)
    spread_sum = np.zeros(max_frames)
    crps_sum   = np.zeros(max_frames)
    counts     = np.zeros(max_frames)

    for predicts, truth in results:
        n_steps, n_lead = predicts.shape[:2]
        T = n_steps * n_lead
        p = predicts.reshape(T, *predicts.shape[2:]).numpy()
        t = truth.reshape(T, *truth.shape[2:]).numpy()

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


def plot_metrics(metrics, out_path='../Performance/metrics.png'):
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
    plt.show()