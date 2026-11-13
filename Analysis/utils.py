from pathlib import Path
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.animation import FuncAnimation 
import math
from scipy import stats


ROOT = Path(__file__).resolve().parent
save_path = ROOT / 'Result' / 'DistributionCheck'
save_path_rk4 = ROOT / 'Result' / 'DistributionCheck' / 'RK4'

def result_dir(ds):
    return {'euler': save_path, 'rk4': save_path_rk4}[ds]

def load_invert(dir):
    dir = Path(dir) if isinstance(dir, str) else dir 
    files = natsorted(dir.glob('invert_traj*initial*.npy'))
    assert files, f'no invert_traj*initial*.npy in {dir}'
    Z    = np.stack([np.load(f).astype(np.float32) for f in files])
    meta = np.array([[int(m.group(1)), int(m.group(2))] for f in files
                     for m in [re.search(r'invert_traj(\d+)initial(\d+)', f.name)]])
    return Z, meta[:, 0], meta[:, 1]

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

def plot_temporal_difference(diff):
    '''
    the diff here cannot be used directly after temporal difference
    diff should be a dict filled with all the difference you want to plot
    the key would be the label and the value would be diff (24, )
    so in this function, it would plot everything in the dict    

    diff: (n_diff, 24)
    '''

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, values in diff.items():
        values = np.asarray(values)
        x = np.arange(len(values)) 
        ax.plot(x, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("lag t")
    ax.set_ylabel("difference")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig, ax

# def get_latents(latent_shape, alpha, device='cpu', z0=None):
#     '''
#     expect latents shape should be like (N, ens, leads, levels, H , W)
#     only support one alpha for all the N windows
#     '''
#     assert len(latent_shape) == 6, f"you need input a 6 elements tuple, now it's like {latent_shape}" 
#     N, ens, leads, C, H, W = latent_shape # latent_shape: (n_samples * n_ens, channels, dx, dy)
#     sample_shape = (C, H, W)  # this should be the shape of one sample

#     z0 = torch.randn(N, ens, *sample_shape, device=device) if z0 is None else z0
#     assert z0.shape == (N, ens, *sample_shape), f"If you pass z0 as a parameter pls keep the dimension, not {z0.shape}"
#     zs = torch.zeros(latent_shape, device=device)
#     zs[:, :, 0, ...] = z0

#     # for each trajectory
#     # sample ensemble times independently
#     # sample N  trajectories times independently
        
#     if isinstance(alpha, float):
#         # alpha would be baseline alpha
#         alpha = np.full(leads-1, alpha)

#     if isinstance(alpha, np.ndarray):
#         # alpha would be fitted by kernel or just empirical mean
#         if len(alpha.shape) == 1:
#             alpha = np.repeat(alpha[None, :], N, axis=0)
#         alpha = torch.tensor(alpha, device=device)

#     if alpha.shape == torch.Size([leads - 1]):
#         alpha = alpha[None, :].to(device)

#     assert alpha.shape[0] == N, f"The number of alpha and number of trajectories should keep same. now they are {N}, {alpha.shape[0]}"
#     assert alpha.shape == (N, leads-1), f"It should be like (N, 23) anyway, but now it's {alpha.shape}"
#     assert alpha.is_cuda, "Are you going to perform forecasting with CPU?"

#     for t in range(1, leads):
#         noise = torch.randn(N, ens, *sample_shape, device=device)
#         idx = t - 1 
#         alpha_t = alpha[:, t - 1][:, None, None, None, None]
#         zs[:, :, t, ...] = alpha_t * zs[:, :, idx, ...] + (1 - alpha_t**2).sqrt() * noise
#     return zs

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
    X0s = torch.empty((N, 2, 64, 64))
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
        X0s[i] = torch.as_tensor(traj[anchor_id] / std)
        windows[i] = torch.as_tensor(physical_window.astype(np.float32) / std)
    return windows, X0s

# per-pixel normality and parameters check
def bh_adjust(p):
    L, m = p.shape
    order = np.argsort(p, axis=1)
    ranked = np.take_along_axis(p, order, axis=1)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[:, ::-1], axis=1)[:, ::-1]
    out = np.empty_like(p)
    np.put_along_axis(out, order, np.clip(adj, 0, 1), axis=1)
    return out

def plot_rejection_ratio(ratio_ss, ratio_z, ratio_chi2, ds='euler'):
    L = len(ratio_ss)
    leads = np.arange(1, L + 1)
    top = max(ratio_ss.max(), ratio_z.max(), ratio_chi2.max()) or 1e-3
    ss_color, z_color, chi2_color = '#2a78d6', '#eb6834', '#1baf7a'

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(leads, ratio_ss, color=ss_color, marker='o', ms=4, lw=1.5, label='Shapiro-Wilk')
    ax.plot(leads, ratio_z, color=z_color, marker='s', ms=4, lw=1.5, label='z-test')
    ax.plot(leads, ratio_chi2, color=chi2_color, marker='^', ms=4, lw=1.5, label='chi-square')
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.03 * top, 1.15 * top)
    ax.set_xlabel('lead time')
    ax.set_ylabel('rejection ratio after Benjamini-Hochberg procedure')
    ax.set_title('Rejection ratio per lead time', loc='left')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False)
    fig.tight_layout()

    # Lead 1 labels extend left and lead 2 labels right, so the two columns can't touch
    base, line_height = 8, 14
    curves = [ax.transData.transform(np.column_stack([leads, r])) for r in (ratio_ss, ratio_z, ratio_chi2)]
    samples = np.vstack([np.linspace(c[k], c[k + 1], 100) for c in curves for k in range(L - 1)])

    def hits_curve(label):
        box = label.get_window_extent().padded(3)
        return np.any((samples[:, 0] > box.x0) & (samples[:, 0] < box.x1)
                      & (samples[:, 1] > box.y0) & (samples[:, 1] < box.y1))

    for i, (ha, dx) in enumerate([('right', -4), ('left', 4)]):
        (low, low_color), (high, high_color) = sorted(
            [(ratio_ss[i], ss_color), (ratio_z[i], z_color)], key=lambda p: p[0]
        )
        gap = (ax.transData.transform((0, high))[1] - ax.transData.transform((0, low))[1]) * 72 / fig.dpi
        offsets = (base, max(base, base + line_height - gap))
        labels = [
            ax.annotate(f'{value:.4f}', (leads[i], value), xytext=(dx, dy), textcoords='offset points',
                        ha=ha, color=color, fontsize=9, fontweight='bold', zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
            for value, color, dy in zip((low, high), (low_color, high_color), offsets)
        ]
        while any(hits_curve(label) for label in labels):
            for label in labels:
                label.xyann = (label.xyann[0], label.xyann[1] + 4)

    fig.savefig(f"{result_dir(ds)}/rejection_ratio.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def pixel_level_test(Z, ds='euler'):
    '''
    Input: Z latent space data
    Output: p-value, p-value after BH adjust. For Anderson-Darling, Z-test, Chi-square test

    '''
    N, L, C, H, W = Z.shape
    D = C*H*W
    Zf = Z.reshape(N, L, D)
    # z-test
    z = np.mean(Zf, axis=0) * np.sqrt(N)     # (L, D)
    pz = 2 * stats.norm.sf(np.abs(z))
    # chi-square test
    s2 = np.var(Zf, axis=0, ddof=1)     # (L, D)
    chi2_stat = (N - 1) * s2
    cdf = stats.chi2.cdf(chi2_stat, df=N - 1)
    # idk
    pchi2 = np.minimum(cdf, 1 - cdf) * 2
    pz_BH = bh_adjust(pz)
    pchi2_BH = bh_adjust(pchi2)

    pss = np.empty((L, D))
    for lead in range(L):
        for i in range(D):
            samples = Zf[:, lead, i]                    # (N,)
            # Shapiro-Wilk normality test
            _, pss[lead, i] = stats.shapiro(samples)

    # BH correction across all pixels at this lead
    pss_BH = bh_adjust(pss)

    ratio_ss = (pss_BH < 0.05).mean(axis=1)      # (L,)
    ratio_z = (pz_BH < 0.05).mean(axis=1)       # (L,)
    ratio_chi2 = (pchi2_BH < 0.05).mean(axis=1) # (L,)

    plot_rejection_ratio(ratio_ss, ratio_z, ratio_chi2, ds=ds)

    return pss, pz, pchi2

def p_histogram(p, title, flex_scale=False, save=False, fine_bins=False, ds='euler'):
    assert title in ["z-test", "chi-square", "Sharpiral-Wilk"], "only z-test, chi-square, Sharpiral-Wilk are supported"
    L, D = p.shape
    if flex_scale:
        fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(4, 6, figsize=(18, 10), sharex=True, sharey=True)
    n_bins = 40 if fine_bins else 20
    for lead, ax in enumerate(axes.ravel()[:L]):
        ax.hist(p[lead], bins=n_bins, range=(0, 1))
        ax.set_title(f"lead time {lead + 1}")
    for ax in axes.ravel()[L:]:
        ax.set_visible(False)
    fig.suptitle(f"{title} test p-values per lead time")
    fig.tight_layout()
    if save:
        fig.savefig(f"{result_dir(ds)}/{title}_p-values.png")
    else:
        plt.show()


def mean_band(x):
    """x: (N, L). Returns mean and 95% CI across trajectories, per lead."""
    m = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    return m, m - 1.96 * se, m + 1.96 * se

# global experiment
def global_parameters(Z):
    N, L, C, H, W = Z.shape
    D = C*H*W
    Zf = Z.reshape(N, L, D)

    mean = np.mean(Zf, axis=2)                       # (N, L)
    var = np.var(Zf, axis=2, ddof=1)                 # (N, L)

    return mean, var

def plot_global_parameters(zf, ref, save=False, ds='euler'):
    zf_mean, zf_var = zf
    ref_mean, ref_var = ref

    L = zf_mean.shape[1]
    leads = np.arange(1, L + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    zf_mean_m, zf_mean_m_lo, zf_mean_m_hi = mean_band(zf_mean)
    zf_var_m, zf_var_m_lo, zf_var_m_hi = mean_band(zf_var)
    ref_mean_m, ref_mean_m_lo, ref_mean_m_hi = mean_band(ref_mean)
    ref_var_m, ref_var_m_lo, ref_var_m_hi = mean_band(ref_var)

    # Global Mean
    ax = axes[0]
    ax.plot(leads, zf_mean_m, marker="o", color="C0", label="Zf")
    ax.fill_between(leads, zf_mean_m_lo, zf_mean_m_hi, color="C0", alpha=0.3)
    ax.plot(leads, ref_mean_m, marker="s", color="C1", label="ref (N(0,1))")
    ax.fill_between(leads, ref_mean_m_lo, ref_mean_m_hi, color="C1", alpha=0.3)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlim(1, L)
    ax.set_xlabel("lead time")
    ax.set_ylabel("spatial mean")
    ax.set_title("Mean over lead time (95% CI)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Global Variance
    ax = axes[1]
    ax.plot(leads, zf_var_m, marker="o", color="C0", label="Zf")
    ax.fill_between(leads, zf_var_m_lo, zf_var_m_hi, color="C0", alpha=0.3)
    ax.plot(leads, ref_var_m, marker="s", color="C1", label="ref (N(0,1))")
    ax.fill_between(leads, ref_var_m_lo, ref_var_m_hi, color="C1", alpha=0.3)
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_xlim(1, L)
    ax.set_xlabel("lead time")
    ax.set_ylabel("variance")
    ax.set_title("Variance over lead time (95% CI)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(f"{result_dir(ds)}/Global_Parameters.png")
    else:
        plt.show()

def power_spectrum(Z, n_std=1.96, save=False, ds='euler'):
    """Z: (N, L, C, H, W)."""
    N, L, C, H, W = Z.shape

    # 2D FFT over (H, W) only, per channel
    F = np.fft.fft2(Z, axes=(3, 4))
    power = np.abs(F) ** 2 / (H * W)   # (N, L, C, H, W)
    power = power.mean(axis=2)          # average over channel -> (N, L, H, W)

    # radial coordinate over the (H, W) plane
    kh = np.fft.fftfreq(H, d=1.0 / H)
    kw = np.fft.fftfreq(W, d=1.0 / W)
    KH, KW = np.meshgrid(kh, kw, indexing='ij')
    radius = np.sqrt(KH ** 2 + KW ** 2)
    k_bins = np.arange(0, radius.max() + 1)
    bin_idx = np.digitize(radius.ravel(), k_bins) - 1

    def radial_average(img):
        # img: (..., H, W) -> (..., len(k_bins))
        flat = img.reshape(*img.shape[:-2], H * W)
        out = np.zeros((*img.shape[:-2], len(k_bins)))
        counts = np.bincount(bin_idx, minlength=len(k_bins))
        for b in range(len(k_bins)):
            mask = bin_idx == b
            if counts[b] > 0:
                out[..., b] = flat[..., mask].mean(axis=-1)
        return out

    p_k = radial_average(power)   # (N, L, n_k)
    white_noise_level = Z.reshape(N, L, -1).var(axis=2) # (N, L)
    
    # CLT mean/band over N trajectories, kept separate per lead l
    p_mean = np.mean(p_k / white_noise_level[..., None], axis=0)            # (L, n_k)
    p_se = p_k.std(axis=0) / np.sqrt(N)       # (L, n_k)
    n_rows, n_cols = 4, 6  # 4*6 = 24 leads
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for l in range(L):
        ax = axes[l]
        ax.plot(k_bins, p_mean[l], color='tab:blue', label='power')
        ax.fill_between(k_bins, p_mean[l] - n_std * p_se[l],
                         p_mean[l] + n_std * p_se[l],
                         color='tab:blue', alpha=0.2)
        ax.axhline(1, color='gray', linestyle='--', linewidth=1, label='reference')
        ax.set_title(f'lead time {l+1}', fontsize=9)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if save:
        fig.savefig(f"{result_dir(ds)}/Pixels_Cov.png")
    else:
        plt.show()

@torch.no_grad()
def get_latents(latent_shape, model, x0, device='cpu', batch=8):
    '''
    Sample latent trajectories from the trained time-varying OU model (always with an ensemble).

    latent_shape = (N, E, L, C, H, W): N windows, E members, L leads (index 0 = lead 1).
    x0: (N, C, H, W) physical state at the anchor, scaled like in training (get_window, std=2660).

    Same chain the training loss scores, per pixel:
        xi_1 ~ N(0, 1)
        xi_k = alpha_k * xi_{k-1} + sqrt(1 - alpha_k^2) * noise,   alpha_k = exp(-c2_k / 2)
        z_k  = sigma(lead k) * xi_k
    so Var(z_k) = sigma_k^2 and corr(z_{k-1}, z_k) = alpha_k.
    Returns zs of shape (N, E, L, C, H, W).
    '''
    assert len(latent_shape) == 6, f"you need input a 6 elements tuple, now it's like {latent_shape}"
    N, E, L, C, H, W = latent_shape
    assert x0.shape[0] == N, "Generated latent space data should match the physical condition state."
    was_training = model.training
    model.eval()                                            # switch off dropout in the CNN

    t, ch, x, y = torch.meshgrid(torch.arange(L - 1), torch.arange(C), torch.arange(H), torch.arange(W), indexing='ij')
    lead = (t.reshape(-1) + 2).float().to(device)           # leads 2..L, (L-1)*C*H*W points
    ch = ch.reshape(-1).to(device)
    x = x.reshape(-1).float().to(device)
    y = y.reshape(-1).float().to(device)

    zs = torch.empty(latent_shape, device=device)
    for s in range(0, N, batch):                            # chunk over windows to bound GPU memory
        x0_b = x0[s:s + batch].to(device).float()
        b = x0_b.shape[0]
        sigma_next, c2 = model(lead, x, y, ch, x0_b)        # sigma at lead k, c2 on [k-1, k]   (b, P)
        sigma_prev, _ = model(lead - 1, x, y, ch, x0_b)     # sigma at lead k-1
        shape = (b, 1, L - 1, C, H, W)                      # 1 = ensemble dim, broadcasts over E
        alpha = torch.exp(-0.5 * c2).reshape(shape)
        std_innov = (-torch.expm1(-c2)).sqrt().reshape(shape)   # sqrt(1 - alpha^2)
        # sigma for every lead: lead 1 from sigma_prev's first slice, leads 2..L from sigma_next
        sigma = torch.cat([sigma_prev.reshape(shape)[:, :, :1], sigma_next.reshape(shape)], 2)  # (b,1,L,C,H,W)

        xi = torch.randn(b, E, C, H, W, device=device)      # lead 1 marginal
        zs[s:s + b, :, 0] = sigma[:, :, 0] * xi
        for k in range(1, L):                               # Z index k is lead k+1, uses alpha[k-1]
            xi = alpha[:, :, k - 1] * xi + std_innov[:, :, k - 1] * torch.randn_like(xi)
            zs[s:s + b, :, k] = sigma[:, :, k] * xi

    model.train(was_training)
    return zs




