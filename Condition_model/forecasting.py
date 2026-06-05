"""
Direct (non-autoregressive) lead-time forecasting with the conditional
flow-matching model.

Pipeline
--------
1. Load eval data with `SQGLeadTimeDataset` in eval mode: each item gives one
   initial condition x0, the ground-truth target at every requested lead, and
   the matching (normalised) lead-time labels.
2. Load the trained model through `CondSampler` (deterministic ODE, eps = 0).
3. For every initial condition and every lead, draw an `n_ensemble` ensemble by
   sampling independent latent noise and integrating the conditional ODE.
4. Score each case with RMSE / CRPS / SSR (utils.py) and average over the
   initial conditions, reporting one row per lead.
"""

import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.asyncio import tqdm

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cond_sampler import CondSampler
from dataset import SQGLeadTimeDataset, DATA_1H, DATA_3H
from utils import compute_rmse, compute_crps, compute_ssr, print_metrics

DEFAULT_MODEL_PATH = THIS_DIR.parent / 'models' / 'best_model_conditional.pth'


# ============================================================================
# Sampling Noise From OU
# ============================================================================
def get_latent(
    z_prev: Optional[torch.Tensor],
    shape: Tuple[int, ...],
    rho: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample latent noise from a discrete-time Ornstein-Uhlenbeck process.

    z_next = rho * z_prev + sqrt(1 - rho^2) * xi,   xi ~ N(0, I)

    This preserves N(0, I) marginals for all rho in [0, 1).
    rho=0  -> independent noise, identical to torch.randn (original behaviour).
    rho->1 -> strongly correlated noise across rollout steps.

    Args:
        z_prev : previous latent tensor, shape `shape`, on any device; None on first step
        shape  : desired output shape, e.g. (n_ensemble, C, H, W)
        rho    : temporal correlation coefficient in [0, 1)
        device : target device for the output tensor

    Returns:
        z_next : shape `shape`, on `device`
    """
    xi = torch.randn(shape, device=device)
    if z_prev is None or rho == 0.0:
        return xi
    return rho * z_prev.to(device) + math.sqrt(1.0 - rho * rho) * xi


def _lead_vector(lead_time, n: int, device: torch.device) -> torch.Tensor:
    """Broadcast a scalar / tensor lead-time condition to a (n,) float tensor."""
    if not torch.is_tensor(lead_time):
        lead_time = torch.as_tensor(lead_time, dtype=torch.float32)
    lead_time = lead_time.to(device=device, dtype=torch.float32).reshape(-1)
    if lead_time.numel() == 1:
        lead_time = lead_time.expand(n).contiguous()
    assert lead_time.numel() == n, (
        f"lead_time has {lead_time.numel()} elements, expected 1 or {n}"
    )
    return lead_time


# ============================================================================
# Sampling Functions
# ============================================================================

@torch.no_grad()
def forecasting(
    x0: torch.Tensor,
    lead_times,
    n_ensemble: int,
    sampler: CondSampler,
    device: Optional[torch.device] = None,
    rho: float = 0.0,
    z_prev: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Produce one ensemble forecast of x0 at a single lead time.

    The same initial state x0 is broadcast across `n_ensemble` members; the
    ensemble spread comes entirely from the independent latent noise (the ODE
    itself is deterministic when the sampler's eps = 0).

    Args:
        x0         : initial state, shape (C, H, W)
        lead_times : scalar (normalised) lead-time condition, broadcast to all
                     ensemble members, or an (n_ensemble,) tensor
        n_ensemble : number of ensemble members
        sampler    : a constructed CondSampler
        device     : torch device (defaults to cuda if available)
        rho        : OU correlation for `get_latent` (only relevant if `z_prev`
                     is supplied; independent noise otherwise)
        z_prev     : previous latent for OU-correlated noise, shape
                     (n_ensemble, C, H, W); None for independent noise

    Returns:
        ensemble predictions, shape (n_ensemble, C, H, W) as a numpy array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Broadcast x0 across ensemble members: (n_ensemble, C, H, W)
    x_current = x0.unsqueeze(0).expand(n_ensemble, -1, -1, -1).clone().to(device)

    z_latent = get_latent(z_prev, x_current.shape, rho, device)
    lead = _lead_vector(lead_times, n_ensemble, device)

    preds = sampler.sample(z0=z_latent, x_t=x_current, lead_times=lead)
    return preds.cpu().numpy()  # (n_ensemble, C, H, W)


@torch.no_grad()
def predict(
    model_path=DEFAULT_MODEL_PATH,
    data_dir=DATA_1H,
    split: str = 'val',
    forecasting_leads: Optional[Sequence[int]] = (1, 5, 10, 20, 50, 75, 99),
    n_init: int = 20,
    n_ensemble: int = 20,
    steps: int = 100,
    eps=None,
    rho: float = 0.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
):
    """
    Full evaluation pipeline.

    For each seeded initial condition (eval-mode dataset) and each lead, draw an
    `n_ensemble` ensemble and score it with RMSE / CRPS / SSR. Metrics are
    computed per initial condition and averaged across them, giving one row per
    lead. Returns a dict of per-lead arrays and prints the table.

    Notes
    -----
    - `eps=None` -> deterministic ODE; ensemble spread comes from the latent
      noise only, which is exactly what we want to calibrate via SSR.
    - The dataset is built with default normalisation (hours_per_frame=1,
      max_horizon=99) so the lead-time labels match those seen in training.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- data: eval mode (returns x0, targets at each lead, lead labels) ----
    leads = list(forecasting_leads) if forecasting_leads is not None else None
    dataset = SQGLeadTimeDataset(
        data_dir=data_dir,
        split=split,
        random_lead_time=False,
        forecasting_leads=leads,
        n_init=n_init,
        seed=seed,
    )
    leads = dataset.forecasting_leads          # list[int] of lead frames
    n_lead = len(leads)

    # ---- model: deterministic conditional ODE sampler ----
    sampler = CondSampler(model_path=model_path, device=device, steps=steps, eps=eps)

    print(
        f'predict | split={split}  inits={len(dataset)}  ensemble={n_ensemble}  '
        f'leads(frames)={leads}  device={device}'
    )

    # accumulate per-lead metrics, averaged over initial conditions
    rmse_sum = np.zeros(n_lead)
    crps_sum = np.zeros(n_lead)
    ssr_sum = np.zeros(n_lead)

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        x0, truth, time_labels = dataset[i]
        # x0: (C,H,W); truth: (n_lead,C,H,W); time_labels: (n_lead,)
        truth_np = truth.numpy()

        # one ensemble per lead -> (n_lead, n_ensemble, C, H, W)
        preds = np.stack(
            [
                forecasting(
                    x0=x0,
                    lead_times=float(time_labels[j]),
                    n_ensemble=n_ensemble,
                    sampler=sampler,
                    device=device,
                    rho=rho,
                )
                for j in range(n_lead)
            ],
            axis=0,
        )
  

        rmse = compute_rmse(preds.mean(axis=1), truth_np)   # (n_lead,)
        crps = compute_crps(preds, truth_np)                # (n_lead,)
        ssr = compute_ssr(preds, rmse)                      # (n_lead,)

        rmse_sum += rmse
        crps_sum += crps
        ssr_sum += ssr

    n = len(dataset)
    rmse_mean = rmse_sum / n
    crps_mean = crps_sum / n
    ssr_mean = ssr_sum / n

    leads_arr = np.asarray(leads)
    print_metrics(leads_arr, rmse_mean, crps_mean, ssr_mean)

    # I need the plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(leads_arr, rmse_mean, marker='o', color='steelblue')
    axes[0].set_title('RMSE (ensemble mean vs truth)')
    axes[0].set_xlabel('Lead time (steps)')
    axes[0].set_ylabel('RMSE (normalised units)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(leads_arr, crps_mean, marker='o', color='darkorange')
    axes[1].set_title('CRPS (lower is better)')
    axes[1].set_xlabel('Lead time (steps)')
    axes[1].set_ylabel('CRPS (normalised units)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(leads_arr, ssr_mean, marker='o', color='seagreen')
    axes[2].axhline(1.0, color='red', linestyle='--', label='ideal SSR = 1')
    axes[2].set_title('Spread/Skill Ratio (ideal = 1)')
    axes[2].set_xlabel('Lead time (steps)')
    axes[2].set_ylabel('SSR')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Autoregressive Rollout Evaluation |  ensemble={n_ensemble}, inits={n_init}', y=1.02)
    plt.tight_layout()
    plt.savefig('results/metrics_46epoch_64traj.png', bbox_inches='tight')
    plt.show()
    print('Plot saved to results/metrics_46epoch_64traj.png')


    return {
    'leads': leads_arr,
    'rmse': rmse_mean,
    'crps': crps_mean,
    'ssr': ssr_mean,
}
if __name__ == '__main__':
    predict()
