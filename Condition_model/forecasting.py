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
from .dataset import SQGLeadTimeDataset, DATA_100, DATA_500, DATA_PATH, DATA_test
from .utils import visualize_results, compute_metrics, plot_metrics


# capsulate the forecasting as a function
# here i input 5 data samples, but here only one forecasting result returns
def forecast(
    sampler,
    data_dir=DATA_500,
    eval_traj_num=5,            # controls eval dataset size
    ens=5,
    max_lead=24,
    frames_per_traj=100,
    alpha=0.5,
    seed=42,
    device = 'cpu',
    latent_fn=None,
    max_ar=None,
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
        latent_fn       : optional callable replacing get_latents, called as
                          latent_fn(shape, n_direct, device) -> (n_direct*B, C, H, W).
                          Must use the same batch-major / lead-minor flattening as
                          get_latents, since time_labels is built in that order.
                          None (default) keeps the AR(1) get_latents behaviour.
        max_ar          : optional cap on the number of AR re-anchorings. The loop
                          runs min(n_ar, max_ar) + 1 windows, so max_ar=0 forecasts
                          only the first max_lead window and latent_fn is called
                          exactly once per sample. None (default) uses the full
                          chain that fits in the trajectory.

    Returns:
        predicts : torch.Tensor  (n_ar+1, n_direct_lead, ens, levels, H, W)
        truth    : torch.Tensor  (n_ar+1, n_direct_lead,      levels, H, W)
        ds       : SQGLeadTimeDataset  (kept in scope for forecasting_leads etc.)
    """
    bs, levels, H, W = 1, 2, 64, 64

    # ── dataset and loader ───────────────────────────────────────────────────
    ds = SQGLeadTimeDataset(
        data_dir         = data_dir,
        split            = 'val',
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
        if max_ar is not None:
            n_ar = min(n_ar, max_ar)

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
            if latent_fn is None:
                latent = get_latents((ens * bs, levels, H, W), n_direct_lead, alpha=alpha, device=device)
            else:
                latent = latent_fn((ens * bs, levels, H, W), n_direct_lead, device)
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





def perform_forecasting(test_pl=False, eval_traj_num=100, ens=20, comparison=False, alpha=0.5):
    '''
    test_pl: Valid the pipeline on the local machine, NOTE this parameter is default as true
    eval_traj_num: The number of predicted trajectories 
    ens: The number of ensemble for each trajectory
    comparison: Forecast using fixed noise and uncorrelated noise as well
    alpha: The default parameter controling the correlation over noise
    '''

    # TODO: It would be better if this line can be reused
    save_dir = DATA_PATH / f"predit_alpha={alpha}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = CondSampler(model_path=SAVE_PATH, device=device, steps=100)
    if test_pl:
        eval_traj_num = 3
        ens = 5

    results = forecast(sampler, eval_traj_num=eval_traj_num, ens=ens, alpha=alpha)
    # store the trajectories
    save_dir.mkdir(parents=True, exist_ok=True)
    # .npy only — these are forecast outputs, not raw SQG generation data,
    # so there's no need for the .nc intermediate format used in sqg_generate.py.
    for i, (predicts, truth) in enumerate(results):
        np.save(save_dir / f"sample{i:02d}_predicts.npy", predicts.numpy())
        np.save(save_dir / f"sample{i:02d}_truth.npy",    truth.numpy())

    print(f"Saved {len(results)} samples to {save_dir}")
    visualize_results(results)
    metrics = compute_metrics(results)
    plot_metrics(metrics)

    if comparison:
        # Comparsion results using uncorrelated noise and fixed noise as the input
        uncorrelated = forecast(sampler, eval_traj_num=eval_traj_num, ens=ens, alpha=0)
        fixed = forecast(sampler, eval_traj_num=eval_traj_num, ens=ens, alpha=1)

        visualize_results(uncorrelated, out_dir='../Visual/uncorrelated')
        visualize_results(fixed, out_dir='../Visual/fixed')

        # store the trajectories

if __name__ == '__main__':
    # TODO: Pass the parameters calling in terminal
    perform_forecasting()






