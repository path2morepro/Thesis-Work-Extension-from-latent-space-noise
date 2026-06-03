"""
Autoregressive Forecasting with Conditional Flow-Matching Model

Provides functions for:
- Loading the trained model
- Single-step ODE sampling
- Ensemble rollout (autoregressive generation)
- Metric computation (RMSE, CRPS, SSR)
- Full evaluation pipeline
"""

import math
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from cond_sampler import CondSampler

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
    return rho * z_prev + math.sqrt(1.0 - rho * rho) * xi



# ============================================================================
# Sampling Functions
# ============================================================================

@torch.no_grad()
def autoregressive_ensemble_rollout(
    x0: torch.Tensor,
    rollout_steps: int,
    n_ensemble: int,
    device: Optional[torch.device] = None,
    sampler: CondSampler = None,
    rho: float = 0.0,
) -> np.ndarray:
    """
    Roll out an ensemble of trajectories autoregressively.

    Generates n_ensemble independent trajectories by sampling at each step.
    All ensemble members start from the same x0 but with different random noise.
    Latent noise is drawn from a discrete-time OU process with correlation rho;
    rho=0 recovers independent noise (original behaviour).

    Args:
        x0            : initial condition, shape (C, H, W)
        rollout_steps : number of autoregressive steps
        n_ensemble    : number of ensemble members
        device        : torch device (inferred from model if not provided)
        sampler       : CondSampler instance
        rho           : OU temporal correlation coefficient in [0, 1)

    Returns:
        preds : np.ndarray, shape (rollout_steps, n_ensemble, C, H, W)
                All values in normalized space (same units as input).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Broadcast x0 across ensemble members: (n_ensemble, C, H, W)
    # In AR, there is no explicit modeling about leading time
    # I don't know whether it would be a problem
    x_current = x0.unsqueeze(0).expand(n_ensemble, -1, -1, -1).clone().to(device)
    preds = []
    z_latent = None
    for _ in range(rollout_steps):
        z_latent = get_latent(z_latent, x_current.shape, rho, device)
        x_next = sampler.sample(z0=z_latent, x_t=x_current)
        preds.append(x_next.numpy())
        x_current = x_next.to(device)

    return np.stack(preds, axis=0)  # (rollout_steps, n_ensemble, C, H, W)


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


def compute_ssr(
    preds: np.ndarray,
    rmse: np.ndarray,
) -> np.ndarray:
    """
    Compute Spread/Skill Ratio (calibration metric).

    SSR = ensemble_spread / RMSE

    Well-calibrated when SSR ≈ 1.0 (spread matches skill).

    Args:
        preds : shape (rollout_steps, n_ensemble, C, H, W) or similar
        rmse  : shape (rollout_steps,) or scalar

    Returns:
        ssr : same shape as rmse
    """
    # Ensemble std averaged over spatial dims
    if preds.ndim == 5:
        spread = preds.std(axis=1).mean(axis=(1, 2, 3))  # (steps,)
    elif preds.ndim == 4:
        spread = preds.std(axis=0).mean(axis=(1, 2))
    else:
        spread = preds.std(axis=0)

    return spread / (rmse + 1e-8)



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
