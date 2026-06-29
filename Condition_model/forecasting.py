import math
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.asyncio import tqdm
import gc

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cond_sampler import CondSampler
from train import SAVE_PATH
from dataset import SQGLeadTimeDataset, DATA_100, DATA_500
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from utils import visualize_results, compute_metrics, plot_metrics


# capsulate the forecasting as a function
# here i input 5 data samples, but here only one forecasting result returns
def forecast(
    sampler,
    data_dir=DATA_100,
    split='val',
    eval_traj_num=5,            # controls eval dataset size
    ens=5,
    max_lead=24,
    frames_per_traj=100,
    alpha=0.5,
    seed=42,
    device = 'cpu'
):
    """
    Autoregressive ensemble forecasting.

    Initialises its own eval-mode dataset internally — caller only needs
    to pass a ready CondSampler.

    Args:
        sampler         : CondSampler (already loaded and on device)
        data_dir        : path to data directory
        split           : 'val' or 'train'
        n_init          : number of eval anchor points (dataset size)
        ens             : ensemble size
        max_lead        : max direct lead in hours  (24)
        frames_per_traj : fixed trajectory length   (100)
        alpha           : noise correlation for get_latents (0=independent, 1=fixed)
        seed            : RNG seed for eval anchor sampling

    Returns:
        predicts : torch.Tensor  (n_ar+1, n_direct_lead, ens, levels, H, W)
        truth    : torch.Tensor  (n_ar+1, n_direct_lead,      levels, H, W)
        ds       : SQGLeadTimeDataset  (kept in scope for forecasting_leads etc.)
    """
    bs, levels, H, W = 1, 2, 64, 64

    # ── dataset and loader ───────────────────────────────────────────────────
    ds = SQGLeadTimeDataset(
        data_dir         = data_dir,
        split            = split,
        random_lead_time = False,
        eval_traj_num    = eval_traj_num,
        max_lead         = max_lead,
        max_frames       = frames_per_traj,
        seed             = seed,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    results = []

    # ── forecasting loop ─────────────────────────────────────────────────────
    for initial, target, lead_time, traj_idx, anchor_t in tqdm(loader, total=len(loader)):

        n_direct_lead = lead_time.shape[1]
        traj_idx      = traj_idx.item()
        traj          = ds.data[traj_idx]
        anchor_t      = anchor_t.item()

        n_ar = (frames_per_traj - 1 - anchor_t - max_lead) // max_lead

        predicts = torch.zeros(n_ar + 1, n_direct_lead, ens, levels, H, W)
        truth    = torch.zeros(n_ar + 1, n_direct_lead,      levels, H, W)

        time_labels = lead_time.squeeze(0).repeat(ens)
        assert time_labels.shape == (ens * bs * n_direct_lead,), \
            f"time_labels shape wrong: {time_labels.shape}"

        initials = initial.repeat(ens * bs * n_direct_lead, 1, 1, 1)
        assert initials.shape == (ens * bs * n_direct_lead, levels, H, W), \
            f"initials shape wrong: {initials.shape}"

        truth[0] = target.squeeze(0)

        for i in range(n_ar+1):
            latent = get_latents((ens * bs, levels, H, W), n_direct_lead, alpha=alpha, device=device)
            pred   = sampler.sample(z0=latent, x_t=initials, lead_times=time_labels)
            # seemingly reshape itself cannot revert pred properly
            # but I don't know what this for either
            pred = pred.reshape(ens, n_direct_lead, levels, H, W).transpose(0, 1).contiguous()
            predicts[i] = pred
   

            if i > 0:
                truth[i] = torch.stack([
                    traj[anchor_t + i * max_lead + k]
                    for k in range(1, max_lead+1)
                ])  # (n_direct_lead, levels, H, W)

            initials = pred[-1].repeat_interleave(n_direct_lead, dim=0)
        results.append((predicts, truth))

    return results



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
        z[t] = (alpha).sqrt() * z[t - 1] + (1 - alpha).sqrt() * noise

    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W) # Transposing makes sure the order is preserved.

    return z


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = CondSampler(model_path=SAVE_PATH, device=device, steps=100, eps=None)
    # test code
    results = forecast(sampler, eval_traj_num=3, ens=5)

    # results = forecast(sampler, eval_traj_num=50, ens=20)
    visualize_results(results)
    metrics = compute_metrics(results)
    plot_metrics(metrics)

    # Comparsion results using uncorrelated noise and fixed noise as the input
    # uncorrelated = forecast(sampler, eval_traj_num=3, alpha=0, eval_traj_num=50, ens=20)
    # fixed = forecast(sampler, eval_traj_num=3, alpha=1, eval_traj_num=50, ens=20)
    # visualize_results(uncorrelated, out_dir='../Visual/uncorrelated')
    # visualize_results(fixed, out_dir='../Visual/fixed')



