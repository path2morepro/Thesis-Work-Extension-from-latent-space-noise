from pathlib import Path
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.animation import FuncAnimation 
import math

# ── paths ───────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
DATA_PATH    = PROJECT_ROOT / 'data'
RESULT_DIR   = ANALYSIS_DIR / 'Result'

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Inverted-latent sets, keyed by the number of Euler steps used for the inversion.
# The key MUST be carried through to the forward integration in any round trip: a
# mismatch produces error that grows with lead and is easily misread as a finding.
INVERT_DIRS = {100:  DATA_PATH / 'invert' /'invert_steps=100',
               300:  DATA_PATH / 'invert' /'invert_steps=300',
               1000: DATA_PATH / 'invert' /'invert_steps=1000'}


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


def shared_windows(dirs=None):
    """Filenames present in EVERY given invert directory.

    Any comparison across step counts must run on the same windows, otherwise the
    integrator is confounded with which trajectories and anchors happened to be picked.

    Args:
        dirs : iterable of directories; defaults to all of INVERT_DIRS.

    Returns:
        sorted list of filenames common to all of them.
    """
    dirs = list(INVERT_DIRS.values()) if dirs is None else list(dirs)
    common = set.intersection(*[{f.name for f in Path(d).glob('invert_traj*initial*.npy')}
                                for d in dirs])
    assert common, f'no window is present in every directory: {dirs}'
    return sorted(common)



def temporal_difference(x):
    '''
    The temporal difference computed in CPRI appendix.

    This function can accept either truth data or predict data

    input:
        if preds (N, ens, 24, levels, H, W)
        if truths (N, 24, levels, H, W)

        and they are numpy

    output: 
        diff (24, )
    '''

    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if len(x.shape) == 6:
        N, ens, L,  _, _, _= x.shape
        x = x.reshape(N, ens, L, -1)
        diff = np.mean(np.abs(x[:, :, 1:, ...] - x[:, :, :-1, ...]), axis=(0, 1, 3))
    elif len(x.shape) == 5:
        N, L, _, _, _ = x.shape
        x = x.reshape(N, L, -1)
        # for each N perform the offset substraction
        diff = np.mean(np.abs(x[:, 1:, ...] - x[:, :-1, ...]), axis=(0, 2))
    else:
        print(x.shape)

    return diff


def temporal_difference_with_initial(x):
    '''
    Since if we estimate z_t using z_0, I'm thinking it might make more sense
    if we also evaulate it in this way.

    This function can accept either truth data or predict data

    input:
        if preds (N, ens, 24, levels, H, W)
        if truths (N, 24, levels, H, W)

        and they are numpy

    output: 
        diff (24, )
    '''
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    
    if len(x.shape) == 6:
        N, ens, L,  _, _, _= x.shape
        x = x.reshape(N, ens, L, -1)
        diff = np.mean(np.abs(x[:, :, 0:1, ...] - x), axis=(0, 1, 3))
    elif len(x.shape) == 5:
        N, L, _, _, _ = x.shape
        x = x.reshape(N, L, -1)
        # for each N perform the offset substraction
        diff = np.mean(np.abs(x[:, 0:1, ...] - x), axis=(0, 2))

    return diff


def plot_temporal_difference(diff):
    '''
    the diff here cannot be used directly after temporal difference
    diff should be a dict filled with all the difference you want to plot
    the key would be the label and the value would be diff (24, )
    so in this function, it would plot everything in the dict    

    diff: (n_diff, 24)
    '''

    _, ax = plt.subplots(figsize=(7, 4))
    for label, values in diff.items():
        values = np.asarray(values)
        x = np.arange(len(values)) 
        ax.plot(x, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("lag t")
    ax.set_ylabel("difference")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax



def get_latents(latent_shape, alpha, device='cpu', markov=False, z0=None):
    '''
    expect latents shape should be like (N, ens, leads, levels, H , W)
    only support one alpha for all the N windows
    '''
    assert len(latent_shape) == 6, f"you need input a 6 elements tuple, now it's like {latent_shape}" 
    N, ens, leads, C, H, W = latent_shape # latent_shape: (n_samples * n_ens, channels, dx, dy)
    sample_shape = (C, H, W)  # this should be the shape of one sample

    z0 = torch.randn(N, ens, *sample_shape, device=device) if z0 is None else z0
    assert z0.shape == (N, ens, *sample_shape), f"If you pass z0 as a parameter pls keep the dimension, not {z0.shape}"
    zs = torch.zeros(latent_shape, device=device)
    zs[:, :, 0, ...] = z0

    # for each trajectory
    # sample ensemble times independently
    # sample N  trajectories times independently
        
    if isinstance(alpha, float):
        # alpha would be baseline alpha
        alpha = np.full(leads-1, alpha)

    if isinstance(alpha, np.ndarray):
        # alpha would be fitted by kernel or just empirical mean
        if len(alpha.shape) == 1:
            alpha = np.repeat(alpha[None, :], N, axis=0)
        alpha = torch.tensor(alpha, device=device)

    if alpha.shape == torch.Size([leads - 1]):
        alpha = alpha[None, :].to(device)

    assert alpha.shape[0] == N, f"The number of alpha and number of trajectories should keep same. now they are {N}, {alpha.shape[0]}"
    assert alpha.shape == (N, leads-1), f"It should be like (N, 23) anyway, but now it's {alpha.shape}"
    assert alpha.is_cuda, "Are you going to perform forecasting with CPU?"

    for t in range(1, leads):
        noise = torch.randn(N, ens, *sample_shape, device=device)
        idx = t - 1 if markov else 0
        alpha_t = alpha[:, t - 1][:, None, None, None, None]
        zs[:, :, t, ...] = alpha_t * zs[:, :, idx, ...] + (1 - alpha_t**2).sqrt() * noise
    return zs


def visualize_results(x, n_col):
    '''
    The input should be a dictionary filled with data
    It would contain both predict and truth data
    They all should keep the same shape like (leads, 64, 64), like just take the first or the second level
    '''
    labels = list(x.keys())
    data = list(x.values())
    # let's do some check math.ceil(7 / 2)t first
    assert isinstance(labels[0], str), "The key should be string"
    assert data[0].shape == (24, 64, 64), f"The dimension of data is incorrect, it's {data[0].shape}."

    leads = data[0].shape[0]
    n_plots = len(x) 
    n_row = math.ceil(n_plots / n_col)
    assert n_row != 0, "Check if you have enough data to fill with a column."
    fig, axes = plt.subplots(n_row, n_col, figsize=(7, 3.5 * n_row))
    axes = np.atleast_1d(axes).flatten()

    # for ax in axes[n_plots:]:
    #     ax.axis('off')          # hide unused trailing axes

    for i, ax in enumerate(axes):
        if i >= n_plots:
            ax.axis('off')
            break
        ax.set_title(labels[i])
        ax.set_xticks([]); ax.set_yticks([])
        abs_max = np.max(np.abs(data[i][0]))

        image = axes[i].imshow(data[i][0], cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout()

    # I want this stupid function in function disappear
    def update(i, data, labels, n_lead, fig, axes, n_plots):
        abs_hour = i % n_lead + 1
        for j in range(n_plots):
            axes[j].images[0].set_data(data[j][i])
        fig.suptitle(f'+{abs_hour}h', fontsize=10, y=1.02)
        return axes[:n_plots].tolist()

    anim = FuncAnimation(fig, update, 
                        fargs=(data, labels, leads, fig, axes, n_plots),
                        frames=leads, interval=25)
    plt.close(fig)
    return anim  


def empirical_correlation(Z):
    """
    Compute the empirical temporal correlation here

    Args:
        Z: invert data
    
    Return:
        rho: empirical temporal correlation (N, leads, leads)
        rho_bar: average rho over trajectories (leads, leads)
        var(z): average variance of each zt over trajectories (leads, )
    """
    # flatten others except num_traj and lead time
    # (N, 24, 2*64*64)
    Xf = Z.reshape(Z.shape[0], Z.shape[1], -1)
    Xf = Xf - Xf.mean(axis=0, keepdims=True)              # remove per-lead mean field
    # It computes the correlation between leads
    # for each n, there is a L*L matrix
    cov  = np.einsum("nlx,nmx->nlm", Xf, Xf) / Xf.shape[-1] # (n, L, L) 
    
    # why mean here at first?
    # wipe out the trajectories
    v  = np.diagonal(cov.mean(axis=0)).copy()               # raw per-lead variances
    # outer produces a matrix from two vectors
    # outer(v,v) would be the variance of leads over all trajectories
    rho = cov / np.sqrt(np.outer(v, v))                      # normalised per sample

    #  (N, 24, 24), Gn.mean (24, 24), v (24, )
    return rho, rho.mean(axis=0), v

def collapse_by_lag(rho):
    '''
    Return the temporal correlation collapsed into rho(Delta).
    It is used to examine the stationarity.
    
    Args:
        rho: empirical temporal correlation (N, 24, 24)

    Return:
        deltas: delta t
        rho_bar: collapsed correlation pooled over trajectories
        rho: same but not averaged
    '''

    n, L_, _ = rho.shape
    deltas   = np.arange(1, L_)
    rhos = np.empty((L_-1, n))
    rho_bar  = np.empty(L_ - 1)
    for i, D in enumerate(deltas):
        per_sample = np.diagonal(rho, offset=D, axis1=1, axis2=2).mean(axis=1)  # (n, ) 
        rhos[i] = per_sample
        rho_bar[i] = per_sample.mean()

    return deltas, rho_bar, rhos


def bootstrap_rho(Z, n_boot=200, seed=0):
    """Bootstrap the correlation matrix by resampling windows with replacement.

    Args:
        Z      : (N, L, ...) latents, one row per window
        n_boot : replicates
        seed   : RNG seed, so the band is reproducible

    Returns:
        (n_boot, L, L) — one ρ̂ matrix per replicate. Percentiles across axis 0 give
        the CI of any quantity derived from ρ̂.

    Each replicate draws n = N with replacement: the width of a bootstrap CI has to
    reflect the sample size actually used.
    """
    rng, N = np.random.default_rng(seed), Z.shape[0]
    rho_bar_samples = []
    for _ in range(n_boot):
        sample = Z[rng.integers(0, N, size=N)]
        rho_bar_sample = empirical_correlation(sample)[1]
        rho_bar_samples.append(rho_bar_sample)

    return np.stack(rho_bar_samples, axis=0)



def alpha(Z):
    """
    Function for computing Cov(z0, zt)

    Args:
        Z: invert data

    Return:
        alpha but normalised by var
    """
    # flatten others except num_traj and lead time
    # (N, 24, 2*64*64)
    Xf = Z.reshape(Z.shape[0], Z.shape[1], -1)
    Xf = Xf - Xf.mean(axis=0, keepdims=True)              # remove per-lead mean field
    X0 = np.expand_dims(Xf[:, 0, ...], axis=1)  # Z0
    d = Xf.shape[-1]  # dimension of Z
    # It computes the correlation between t=0 and t= others
    alpha  = np.einsum("nlx,nmx->nlm", Xf, X0) / d   # (n, L, 1) 
    normalised_alpha = alpha / np.var(X0)

    return normalised_alpha



def compute_alpha(Z):
    """
    Lag-1 correlation corr(z_{t-1}, z_t).

    Args:
        Z: array (N, T, 2, 64, 64)
        per_sample: True -> one alpha per trajectory, shape (N, T-1)
                    False -> one alpha pooled over trajectories, shape (T-1,)
        center: remove the mean field across trajectories at each lead time

    Returns:
        alpha in [-1, 1]
    """
    Z = Z.reshape(Z.shape[0], Z.shape[1], -1)      # (N, T, D)
    Z = Z - Z.mean(axis=0, keepdims=True)

    Zp, Zc = Z[:, :-1], Z[:, 1:]                   # (N, T-1, D)
    cov = np.einsum("ntx,ntx->nt", Zp, Zc)
    vp  = np.einsum("ntx,ntx->nt", Zp, Zp)
    vc  = np.einsum("ntx,ntx->nt", Zc, Zc)

    return cov / np.sqrt(vp * vc)              # (N, T-1)


def get_window(meta, data_dir, std=2660):
    '''Expect meta is [N, 2]
    where N is the number of windows
    and for each n there would be a traj_idx, and anchor idx

    the input should be torch data and the output is also torch data.

    if you want to optimize this implement
    there would be multiple windows sample from one trajectory
    you can try to read each trajectory only once 
    '''

    if meta.device.type == 'cuda':
        meta = meta.cpu()


    N = meta.shape[0]
    traj_ids = meta[:, 0]
    anchor_ids = meta[:, 1]
    windows = torch.empty((N, 24, 2, 64, 64))
    hit = {}
    
    for i in range(N):

        traj_id = traj_ids[i]
        if not isinstance(traj_id, int):
            traj_id = int(traj_id.item())

        if traj_id in hit: 
            traj = hit.get(traj_id)
        else:
            try:
                file_name = data_dir / f'sqg_N64_1hrly_{traj_id}.npy'
                traj = np.load(file_name)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                return "BOOM!"
            hit[traj_id] = traj

        anchor_id = anchor_ids[i]
        if not isinstance(anchor_id, int):
            anchor_id = int(anchor_id.item())

        physical_window = traj[(anchor_id+1):anchor_id+25, ... ]
        windows[i] = torch.as_tensor(physical_window.astype(np.float32) / std)
    return windows