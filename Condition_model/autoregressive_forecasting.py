"""
Autoregressive Forecasting with Conditional Flow-Matching Model

Provides functions for:
- Loading the trained model
- Single-step ODE sampling
- Ensemble rollout (autoregressive generation)
- Metric computation (RMSE, CRPS, SSR)
- Full evaluation pipeline
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Import SongUNet
SQG_DIR = Path(__file__).parent / 'SQG'
if str(SQG_DIR) not in sys.path:
    sys.path.insert(0, str(SQG_DIR))
from diffusion_networks import SongUNet


# ============================================================================
# Config & Model Loading
# ============================================================================

class AutoregressiveConfig:
    """Configuration for autoregressive evaluation."""

    def __init__(
        self,
        model_path: str = '../models/results/best_model_conditional.pth',
        data_std: float = 2660.0,
        img_channels: int = 2,
        img_resolution: int = 64,
        filters: int = 32,
        label_dropout: float = 0.1,
        n_ensemble: int = 10,
        ode_steps: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.model_path = Path(model_path)
        self.data_std = data_std
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.filters = filters
        self.label_dropout = label_dropout
        self.n_ensemble = n_ensemble
        self.ode_steps = ode_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(config: AutoregressiveConfig) -> SongUNet:
    """
    Load the trained conditional flow-matching model.

    Args:
        config: AutoregressiveConfig instance

    Returns:
        Loaded SongUNet model on the specified device
    """
    model = SongUNet(
        img_resolution=config.img_resolution,
        in_channels=config.img_channels * 2,  # noisy z + condition
        out_channels=config.img_channels,
        embedding_type='fourier',
        encoder_type='residual',
        decoder_type='standard',
        channel_mult_noise=2,
        resample_filter=[1, 3, 3, 1],
        model_channels=config.filters,
        channel_mult=[2, 2, 2],
        attn_resolutions=[32],
        label_dropout=config.label_dropout,
    ).to(config.device)

    model.load_state_dict(torch.load(config.model_path, map_location=config.device))
    model.eval()

    return model


# ============================================================================
# Sampling Functions
# ============================================================================

@torch.no_grad()
def sample_one_step(
    model: SongUNet,
    x_t: torch.Tensor,
    steps: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    One conditional flow ODE solve: noise → x_{t+1}.

    Uses Euler integration from s=0 (noise) to s=1 (data).

    Args:
        model  : trained SongUNet
        x_t    : conditioning state, shape (B, C, H, W)
        steps  : number of Euler steps
        device : torch device (inferred from model if not provided)

    Returns:
        x_t1_pred, shape (B, C, H, W)
    """
    if device is None:
        device = next(model.parameters()).device

    B = x_t.shape[0]
    dt = 1.0 / steps
    z = torch.randn_like(x_t)  # fresh noise each call

    for i in range(steps):
        s = torch.full((B,), i * dt, device=device)
        b = model(z, s, class_labels=x_t)  # predicted velocity
        z = z + b * dt  # Euler step

    return z


@torch.no_grad()
def autoregressive_ensemble_rollout(
    model: SongUNet,
    x0: torch.Tensor,
    rollout_steps: int,
    n_ensemble: int,
    ode_steps: int = 100,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Roll out an ensemble of trajectories autoregressively.

    Generates n_ensemble independent trajectories by sampling at each step.
    All ensemble members start from the same x0 but with different random noise.

    Args:
        model         : trained SongUNet
        x0            : initial condition, shape (C, H, W)
        rollout_steps : number of autoregressive steps
        n_ensemble    : number of ensemble members
        ode_steps     : Euler steps per ODE solve
        device        : torch device (inferred from model if not provided)

    Returns:
        preds : np.ndarray, shape (rollout_steps, n_ensemble, C, H, W)
                All values in normalized space (same units as input).
    """
    if device is None:
        device = next(model.parameters()).device

    # Broadcast x0 across ensemble members: (n_ensemble, C, H, W)
    x_current = x0.unsqueeze(0).expand(n_ensemble, -1, -1, -1).clone().to(device)

    preds = []
    for _ in range(rollout_steps):
        x_next = sample_one_step(model, x_current, steps=ode_steps, device=device)
        preds.append(x_next.cpu().numpy())
        x_current = x_next  # feed prediction forward

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


# ============================================================================
# Full Evaluation Pipeline
# ============================================================================

def evaluate_autoregressive(
    model: SongUNet,
    val_trajs: list,
    init_points: list,
    rollout_steps: int = 20,
    n_ensemble: int = 10,
    ode_steps: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full autoregressive evaluation over multiple init points.

    Args:
        model         : trained SongUNet
        val_trajs     : list of torch tensors, each shape (T, C, H, W)
        init_points   : list of (traj_idx, time_idx) tuples
        rollout_steps : number of autoregressive steps
        n_ensemble    : ensemble size
        ode_steps     : ODE integration steps
        device        : torch device

    Returns:
        (lead_times, mean_rmse, mean_crps, mean_ssr)
        All arrays have shape (rollout_steps,)
    """
    if device is None:
        device = next(model.parameters()).device

    all_rmse = []
    all_crps = []
    all_ssr = []

    for ti, t in tqdm(init_points, desc='Evaluating init points'):
        x0 = val_trajs[ti][t]  # (C, H, W)
        truth = val_trajs[ti][t + 1 : t + 1 + rollout_steps].numpy()  # (steps, C, H, W)

        # Ensemble rollout
        preds = autoregressive_ensemble_rollout(
            model, x0, rollout_steps, n_ensemble, ode_steps, device=device
        )  # (steps, n_ens, C, H, W)

        preds_mean = preds.mean(axis=1)  # (steps, C, H, W)

        rmse = compute_rmse(preds_mean, truth)  # (steps,)
        crps = compute_crps(preds, truth)  # (steps,)
        ssr = compute_ssr(preds, rmse)  # (steps,)

        all_rmse.append(rmse)
        all_crps.append(crps)
        all_ssr.append(ssr)

    # Average over init points
    mean_rmse = np.stack(all_rmse).mean(axis=0)  # (steps,)
    mean_crps = np.stack(all_crps).mean(axis=0)
    mean_ssr = np.stack(all_ssr).mean(axis=0)

    lead_times = np.arange(1, rollout_steps + 1)

    return lead_times, mean_rmse, mean_crps, mean_ssr


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
