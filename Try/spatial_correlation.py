import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from data import SQGData
from scipy.ndimage import gaussian_filter

def _detrend_and_standardize(frames, local=False, local_sigma=0):
    """
    对帧序列去趋势并标准化。
    frames: (K, X, Y)
    local=False: 仅全局去均值、全局标准差
    local=True: 先用高斯滤波提取趋势(均值场)，再除以局部标准差（更严格）
    """
    K, X, Y = frames.shape
    F = frames.astype(np.float64)
    if not local:
        F = F - F.mean()
        std = F.std()
        if std > 0:
            F = F / std
        return F
    else:
        # 局部均值/方差（用高斯平滑做缓慢变化项）
        if local_sigma <= 0:
            local_sigma = max(3, min(X, Y)//16)  # 一个比较稳的缺省
        mu = np.stack([gaussian_filter(F[k], sigma=local_sigma) for k in range(K)], axis=0)
        R = F - mu
        # 局部标准差：先局部二阶矩，再减去mu^2
        m2 = np.stack([gaussian_filter(R[k]**2, sigma=local_sigma) for k in range(K)], axis=0)
        s = np.sqrt(np.clip(m2, 0, None))
        s[s == 0] = 1.0
        Z = R / s
        # 兜底全局标准化一次
        Z = (Z - Z.mean()) / (Z.std() + 1e-12)
        return Z

def _acf2d(img):
    """
    单帧2D自相关，wrap边界，归一化C(0,0)=1
    """
    c = correlate2d(img - img.mean(), img - img.mean(), mode='full', boundary='wrap')
    c = c / (np.max(c) + 1e-12)
    return c

def _radial_average(acf2d, r_max=None):
    H, W = acf2d.shape
    cx, cy = H//2, W//2
    y, x = np.indices(acf2d.shape)
    r = np.sqrt((y - cx)**2 + (x - cy)**2)
    if r_max is None:
        r_max = int(r.max())
    rb = r.astype(np.int32)
    radial = np.zeros(r_max+1, dtype=np.float64)
    counts = np.zeros(r_max+1, dtype=np.int64)
    np.add.at(radial, rb, acf2d)
    np.add.at(counts, rb, 1)
    counts[counts == 0] = 1
    radial = radial / counts
    return radial[:r_max+1]

def _directional_profiles(acf2d, n_angles=8, r_max=None):
    """
    取若干方向的ACF曲线：rho(r, theta_m)
    用极坐标分桶（半径整值+角度整桶），返回矩阵形状 (n_angles, r_bins)
    """
    H, W = acf2d.shape
    cy, cx = H//2, W//2
    yy, xx = np.indices(acf2d.shape)
    ry = yy - cy
    rx = xx - cx
    r = np.sqrt(ry**2 + rx**2)
    if r_max is None:
        r_max = int(r.max())
    rbin = r.astype(np.int32)
    theta = np.arctan2(ry, rx)  # [-pi, pi]
    # 仅用 0..pi（因为ACF是偶函数，对称即可），把角度映射到[0, pi)
    theta = np.mod(theta, np.pi)
    # 分桶
    ang_edges = np.linspace(0, np.pi, n_angles+1, endpoint=True)
    profiles = np.zeros((n_angles, r_max+1), dtype=np.float64)
    counts = np.zeros_like(profiles, dtype=np.int64)
    for m in range(n_angles):
        mask_ang = (theta >= ang_edges[m]) & (theta < ang_edges[m+1])
        rb = rbin[mask_ang]
        vals = acf2d[mask_ang]
        # 累加到半径桶
        np.add.at(profiles[m], rb, vals)
        np.add.at(counts[m], rb, 1)
    counts[counts == 0] = 1
    profiles = profiles / counts
    return profiles[:, :r_max+1]  # (n_angles, R)

def _subblocks_indices(X, Y, blocks=(2,2)):
    bx, by = blocks
    xs = np.linspace(0, X, bx+1, dtype=int)
    ys = np.linspace(0, Y, by+1, dtype=int)
    regions = []
    for i in range(bx):
        for j in range(by):
            regions.append((slice(xs[i], xs[i+1]), slice(ys[j], ys[j+1])))
    return regions

def _wss_block_diagnostics(frames, blocks=(2,2), acf_r0=8):
    """
    分块均值/方差 + 分块ACF一致性指标 Δ_{kℓ}
    返回：
      mu_k, std_k, ratio_std, delta_median
    """
    # 用第一帧（或平均帧）做空间统计更稳定
    F = frames.mean(axis=0)  # (X, Y)
    X, Y = F.shape
    regions = _subblocks_indices(X, Y, blocks=blocks)
    mus, stds = [], []
    acfs = []
    for slx, sly in regions:
        B = F[slx, sly]
        mus.append(B.mean())
        stds.append(B.std() + 1e-12)
        ac = _acf2d(B)
        # 取小滞后窗口内的acf向量作为“形状特征”
        H, W = ac.shape
        cy, cx = H//2, W//2
        yy, xx = np.indices(ac.shape)
        r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        mask = (r <= acf_r0)
        acfs.append(ac[mask].ravel())
    mus = np.array(mus)
    stds = np.array(stds)
    ratio_std = stds.max() / stds.min()
    # ACF形状一致性：两两欧氏距离的平方平均
    acf_mat = np.vstack(acfs)
    K = acf_mat.shape[0]
    deltas = []
    for i in range(K):
        for j in range(i+1, K):
            # 归一化后比较形状
            ai = acf_mat[i]
            aj = acf_mat[j]
            ai = (ai - ai.mean()) / (ai.std() + 1e-12)
            aj = (aj - aj.mean()) / (aj.std() + 1e-12)
            d = np.mean((ai - aj)**2)
            deltas.append(d)
    delta_median = np.median(deltas) if deltas else 0.0
    return mus, stds, ratio_std, delta_median

def _spectrum_and_kappa(frames, smooth_sigma=1.5):
    """
    频域周期图 + 结构张量特征值比 kappa
    用平均帧的FFT估计谱，作轻度高斯平滑后计算二阶矩矩阵M
    """
    F = frames.mean(axis=0)
    F = F - F.mean()
    S = np.abs(np.fft.fftshift(np.fft.fft2(F)))**2  # 周期图
    if smooth_sigma is not None and smooth_sigma > 0:
        Ssm = gaussian_filter(S, sigma=smooth_sigma)
    else:
        Ssm = S
    X, Y = Ssm.shape
    # 频率坐标（单位化到 [-1,1]）
    fy = np.linspace(-1, 1, X)
    fx = np.linspace(-1, 1, Y)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    # 结构张量
    Mxx = np.sum((FX**2) * Ssm)
    Myy = np.sum((FY**2) * Ssm)
    Mxy = np.sum((FX*FY) * Ssm)
    M = np.array([[Mxx, Mxy], [Mxy, Myy]], dtype=np.float64)
    evals = np.linalg.eigvalsh(M)
    evals = np.sort(evals)
    lam_min, lam_max = evals[0], evals[-1]
    kappa = (lam_max / (lam_min + 1e-12))
    return S, Ssm, kappa

def spatial_correlation_diagnostics(field,
                                    average_frames=10,
                                    blocks=(2,2),
                                    n_angles=8,
                                    r_max=None,
                                    use_local_standardize=False,
                                    show_plots=True,
                                    thresholds=dict(
                                        mu_abs_max=0.1,      # |μ_k| ≤ 0.1
                                        std_ratio_max=1.5,   # max std / min std ≤ 1.5
                                        delta_median_max=0.05,# 分块ACF形状差异
                                        D_median_max=0.02,   # 方向性ACF离散度
                                        kappa_max=1.3        # 结构张量特征值比
                                    )):
    """
    基于 2D ACF + 径向平均 的“诊断—决策”实现（第一版核心）。
    输入
    ----
    field: (T, X, Y)
    average_frames: 取前K帧做平均与ACF，避免偶然噪声
    blocks: 分块个数( bx, by )
    n_angles: 方向性ACF的方向数
    r_max: 径向最大半径（None为自动）
    use_local_standardize: 是否使用局部趋势/尺度去除（更严格）
    show_plots: 是否绘图
    thresholds: 各诊断阈值（工程阈值，可在你的数据上微调）

    返回
    ----
    results: dict，含诊断指标、2D ACF、径向曲线、方向性曲线、谱图与决策结论
    """
    T, X, Y = field.shape
    K = min(average_frames, T)
    frames0 = field[:K].astype(np.float64)

    # 0) 去趋势与标准化
    frames = _detrend_and_standardize(frames0, local=use_local_standardize)

    # 1) WSS：分块均值/方差 + 2) 分块ACF形状一致性
    mu_k, std_k, std_ratio, delta_median = _wss_block_diagnostics(frames, blocks=blocks, acf_r0=8)

    # 3) 计算平均2D ACF与径向平均
    acf2d = np.zeros((2*X-1, 2*Y-1), dtype=np.float64)
    for i in range(K):
        acf2d += _acf2d(frames[i])
    acf2d /= K
    radial = _radial_average(acf2d, r_max=r_max)
    # 相关长度（1/e）
    corr_len = int(np.argmax(radial < np.exp(-1))) if np.any(radial < np.exp(-1)) else None

    # 3) 各向同性：方向性ACF离散度
    dir_profiles = _directional_profiles(acf2d, n_angles=n_angles, r_max=(len(radial)-1))
    # 只在“有效半径”（样本对足够多）评估；这里简化为前 1/3~1/2 的半径
    R = dir_profiles.shape[1]
    r_lo, r_hi = max(1, R//12), max(2, R//2)
    D_r = np.var(dir_profiles[:, r_lo:r_hi], axis=0)  # 每个半径桶的方向方差
    D_median = float(np.median(D_r))

    # 4) 频域：周期图 + 结构张量特征值比 kappa
    S, Ssm, kappa = _spectrum_and_kappa(frames, smooth_sigma=1.5)

    # 5) 决策：是否可把“径向平均”作为主要摘要
    th = thresholds
    wss_ok = (np.max(np.abs(mu_k)) <= th['mu_abs_max']) and (std_ratio <= th['std_ratio_max']) and (delta_median <= th['delta_median_max'])
    iso_ok = (D_median <= th['D_median_max']) and (kappa <= th['kappa_max'])

    if wss_ok and iso_ok:
        decision = "ACCEPT_RADIAL"  # 采用径向平均作为主要摘要
        note = ("诊断显示近似WSS与近似各向同性成立（分块均值/方差稳定、分块ACF形状一致、方向性离散度低、谱近似圆对称）。"
                "因此后续可使用径向平均ACF作为主要结果与拟合依据。")
    elif wss_ok and not iso_ok:
        decision = "KEEP_DIRECTIONAL"
        note = ("二阶平稳近似成立，但存在各向异性（方向性ACF离散度或谱椭圆率较高）。"
                "不应做径向平均作为推断依据，应保留方向族 ρ(r,θ) 或采用各向异性核建模。")
    elif (not wss_ok) and iso_ok:
        decision = "LOCAL_ANALYSIS"
        note = ("存在非平稳性（分块统计或分块ACF差异明显），尽管各向同性近似尚可。"
                "建议去趋势/局部标准化后做**局部**ACF/谱分析，或报告不同区域的径向曲线。")
    else:
        decision = "DIRECTIONAL_AND_LOCAL"
        note = ("同时存在非平稳与各向异性。径向平均仅作可视化参考，不用于推断；"
                "正文呈现方向性+局部结果，建模采用各向异性与局部平稳方法。")

    results = dict(
        # 你原函数的输出
        acf2d=acf2d,
        radial_correlation=radial,
        correlation_length=corr_len,
        # 新增诊断指标
        block_means=mu_k, 
        block_stds=std_k, 
        std_ratio=float(std_ratio),
        block_acf_shape_delta_median=float(delta_median),
        directional_profiles=dir_profiles,       # 形状 (n_angles, R)
        directional_dispersion_by_r=D_r,         # 每个半径的方向方差
        directional_dispersion_median=float(D_median),
        periodogram=S, periodogram_smooth=Ssm,
        kappa=float(kappa),
        # 决策
        decision=decision,
        decision_note=note,
        thresholds=thresholds,
        meta=dict(
            average_frames=K, blocks=blocks, n_angles=n_angles, r_range=(r_lo, r_hi),
            use_local_standardize=use_local_standardize
        )
    )

    if show_plots:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        # 2D ACF
        im0 = axs[0,0].imshow(acf2d, cmap='RdBu_r', origin='lower')
        axs[0,0].set_title("2D ACF (normalized)")
        plt.colorbar(im0, ax=axs[0,0], fraction=0.046, pad=0.04)

        # Radial
        axs[0,1].plot(radial, lw=1.5)
        axs[0,1].axhline(np.exp(-1), ls='--')
        if corr_len is not None:
            axs[0,1].axvline(corr_len, ls='--')
        axs[0,1].set_title("Radial ACF")
        axs[0,1].set_xlabel("r (px)")
        axs[0,1].set_ylabel("C(r)")

        # Directional profiles (r in [r_lo, r_hi))
        for m in range(dir_profiles.shape[0]):
            axs[0,2].plot(np.arange(r_lo, r_hi), dir_profiles[m, r_lo:r_hi], alpha=0.8)
        axs[0,2].set_title(f"Directional ACFs (n={n_angles})")
        axs[0,2].set_xlabel("r (px)")
        axs[0,2].set_ylabel("C(r)")

        # Periodogram
        im1 = axs[1,0].imshow(np.log1p(S), cmap='magma', origin='lower')
        axs[1,0].set_title("Periodogram log(1+S)")
        plt.colorbar(im1, ax=axs[1,0], fraction=0.046, pad=0.04)
        im2 = axs[1,1].imshow(np.log1p(Ssm), cmap='magma', origin='lower')
        axs[1,1].set_title(f"Smoothed Spectrum (κ={results['kappa']:.2f})")
        plt.colorbar(im2, ax=axs[1,1], fraction=0.046, pad=0.04)

        # WSS block stats
        axs[1,2].bar(np.arange(len(mu_k))-0.2, mu_k, width=0.4, label='μ_k')
        axs[1,2].bar(np.arange(len(mu_k))+0.2, std_k, width=0.4, label='σ_k')
        axs[1,2].set_title(f"WSS blocks: std_ratio={std_ratio:.2f}, Δ_med={delta_median:.3f}")
        axs[1,2].legend()

        plt.tight_layout()
        plt.show()

        print("【决策】", decision)
        print("说明：", note)

    return results

# sqg = SQGData()
# noise0 = sqg.get_field()


# results = spatial_correlation_diagnostics(noise0, 
#                                           average_frames=10, 
#                                           blocks=(2,2), 
#                                           n_angles=8, 
#                                           r_max=None, 
#                                           show_plots=True)


# results_local = spatial_correlation_diagnostics(noise0,
#     average_frames=10,
#     use_local_standardize=True,
#     show_plots=True
# )



def _spectrum_kappa(img, smooth_sigma=1.5):
    F = img - img.mean()
    S = np.abs(np.fft.fftshift(np.fft.fft2(F)))**2
    if smooth_sigma and smooth_sigma > 0:
        Ssm = gaussian_filter(S, sigma=smooth_sigma)
    else:
        Ssm = S
    X, Y = Ssm.shape
    fy = np.linspace(-1, 1, X); fx = np.linspace(-1, 1, Y)
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    Mxx = np.sum((FX**2) * Ssm); Myy = np.sum((FY**2) * Ssm); Mxy = np.sum((FX*FY) * Ssm)
    evals = np.linalg.eigvalsh(np.array([[Mxx, Mxy],[Mxy, Myy]]))
    lam_min, lam_max = evals[0], evals[-1]
    kappa = lam_max / (lam_min + 1e-12)
    return S, Ssm, float(kappa)

def spatial_correlation_local_report(field,
                                     average_frames=10,
                                     blocks=(2,2),
                                     use_local_standardize=True,
                                     n_angles=8,
                                     r_max=None,
                                     show_plots=True):
    """
    对每个子块计算：2D ACF、径向ACF、相关长度 xi、方向离散度 D_k、谱椭圆率 kappa_k。
    产出：xi 热力图、分块径向曲线面板、分块谱图面板（可选）。
    """
    T, X, Y = field.shape
    K = min(average_frames, T)
    frames0 = field[:K].astype(np.float64)
    frames = _detrend_and_standardize(frames0, local=use_local_standardize)

    # 平均帧（更稳）：频域/展示
    mean_img = frames.mean(axis=0)

    # 分块
    regions = _subblocks_indices(X, Y, blocks)
    bx, by = blocks
    nb = len(regions)

    # 每块计算
    xi_map = np.full((bx, by), np.nan, dtype=float)
    D_list, kappa_map = np.full((bx, by), np.nan), np.full((bx, by), np.nan)
    radials = []    # 保存每块的径向曲线
    acf2ds = []     # 如需展示
    spectra = []    # 如需展示

    for idx, (sx, sy) in enumerate(regions):
        bi, bj = divmod(idx, by)
        # 用每帧的该子块，做 ACF 平均
        acf2d = np.zeros((2*(sx.stop - sx.start)-1, 2*(sy.stop - sy.start)-1), dtype=np.float64)
        for t in range(K):
            acf2d += _acf2d(frames[t, sx, sy])
        acf2d /= K
        acf2ds.append(acf2d)

        # 径向 + xi
        radial = _radial_average(acf2d, r_max=r_max)
        radials.append(radial)
        xi = int(np.argmax(radial < np.exp(-1))) if np.any(radial < np.exp(-1)) else np.nan
        xi_map[bi, bj] = xi

        # 方向离散度（各向同性的局部证据）
        prof = _directional_profiles(acf2d, n_angles=n_angles, r_max=(len(radial)-1))
        R = prof.shape[1]; r_lo, r_hi = max(1, R//12), max(2, R//2)
        D_r = np.var(prof[:, r_lo:r_hi], axis=0)
        D_med = float(np.median(D_r))
        D_list[bi, bj] = D_med

        # 谱椭圆率（方向性频域证据）
        S, Ssm, kappa = _spectrum_kappa(mean_img[sx, sy], smooth_sigma=1.2)
        spectra.append(Ssm)
        kappa_map[bi, bj] = kappa

    results = dict(
        blocks=blocks,
        xi_map=xi_map,
        D_map=D_list,
        kappa_map=kappa_map,
        radials=radials,
        acf2ds=acf2ds,
        spectra=spectra,
        meta=dict(average_frames=K, n_angles=n_angles, use_local_standardize=use_local_standardize)
    )

    if show_plots:
        # 1) 相关长度热力图
        fig, ax = plt.subplots(1, 3, figsize=(14,4))
        im0 = ax[0].imshow(xi_map, cmap='viridis', origin='lower')
        ax[0].set_title("Local correlation length ξ (pixels)")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(D_list, cmap='magma', origin='lower')
        ax[1].set_title("Directional dispersion D (median over r)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(kappa_map, cmap='cividis', origin='lower', vmin=1.0)
        ax[2].set_title("Spectral anisotropy κ (λ_max/λ_min)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

        # 2) 分块径向曲线面板
        rows, cols = blocks
        fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)
        for idx, radial in enumerate(radials):
            r = np.arange(len(radial))
            i, j = divmod(idx, cols)
            axs[i,j].plot(r, radial, lw=1.5)
            axs[i,j].axhline(np.exp(-1), ls='--', alpha=0.6)
            xi = xi_map[i, j]
            if np.isfinite(xi):
                axs[i,j].axvline(int(xi), ls='--', alpha=0.6)
            axs[i,j].set_title(f"Block ({i+1},{j+1})  ξ≈{xi_map[i,j]:.0f}, D≈{D_list[i,j]:.3f}, κ≈{kappa_map[i,j]:.2f}")
            axs[i,j].set_xlabel("r (px)"); axs[i,j].set_ylabel("C(r)")
        plt.tight_layout(); plt.show()

    # 简短文字结论（给你复用在报告里）
    print("—— 局部分析小结 ——")
    print(f"分块: {blocks}, 帧数用于平均: {K}, 局部标准化: {use_local_standardize}")
    print(f"ξ（相关长度）范围: [{np.nanmin(xi_map):.0f}, {np.nanmax(xi_map):.0f}] px")
    print(f"D（方向离散度）中位数: {np.nanmedian(D_list):.3f}（越小越各向同性）")
    print(f"κ（谱椭圆率）范围: [{np.nanmin(kappa_map):.2f}, {np.nanmax(kappa_map):.2f}]（≈1为圆对称）")

    return results


# # 强烈建议：局部标准化以削弱漂移
# local_res = spatial_correlation_local_report(
#     noise0,
#     average_frames=10,
#     blocks=(2,2),              # 可改 (3,3) / (4,4) 看空间分辨率需要
#     use_local_standardize=True,
#     n_angles=8,
#     r_max=None,
#     show_plots=True
# )

# 以上成功证明像素之间线性不相关

# -*- coding: utf-8 -*-
# Optimized HSIC (RBF kernel) — PyTorch version with caching + robust bandwidth + vectorized radial mean

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================
# Global dtype (configurable)
# ==========================
DTYPE = torch.float64  # you can switch to torch.float32 in hsic_map_rbf_torch(..., dtype=torch.float32)

# ==========================
# Caches
# ==========================
_H_CACHE: dict[tuple, Tensor] = {}          # key: (n, dtype, device) -> H
_GRID_CACHE: dict[tuple, tuple] = {}        # key: (H, W, device) -> (yy, xx, r_long, rmax)
_SIGMA_CACHE: dict[tuple, Tensor] = {}      # optional: (n, device, dtype, 'x'/'y') -> sigma (if you want to cache computed sigmas)


# -----------------------------------------
# 1) Sampling pairs (unchanged, already OK)
# -----------------------------------------
@torch.no_grad()
def sample_pairs_for_shift_torch(field: Tensor, dx: int, dy: int,
                                 max_samples: int = 3000, spatial_stride: int = 1,
                                 generator=None) -> tuple[Tensor, Tensor]:
    """
    Sample pixel pairs separated by spatial shift (dx, dy) from a 3D field (T, X, Y).
    Returns flattened 1D tensors (xvals, yvals). If total pairs > max_samples, random subsample.
    """
    T, X, Y = field.shape
    xs = torch.arange(0, X, spatial_stride, device=field.device)
    ys = torch.arange(0, Y, spatial_stride, device=field.device)

    a = field[:, xs, :][:, :, ys]  # (T, |xs|, |ys|)
    b = torch.roll(field, shifts=(0, dx, dy), dims=(0, 1, 2))[:, xs, :][:, :, ys]

    a = a.reshape(-1)
    b = b.reshape(-1)

    n = a.numel()
    if n > max_samples:
        gen = generator or torch.Generator(device=field.device)
        idx = torch.randperm(n, generator=gen, device=field.device)[:max_samples]
        a = a[idx]
        b = b[idx]
    return a, b


# ------------------------------------------------------
# 2) Vectorized radial mean with cached grid / binning
# ------------------------------------------------------
@torch.no_grad()
def radial_mean_from_map_torch(arr2d: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """
    Compute radial mean curve from a 2D map centered at (cx, cy).
    Uses cached (yy, xx, r) grid and vectorized binning, ignoring NaNs.
    Returns (radial_mean, (cx, cy)).
    """
    Hh, Ww = arr2d.shape
    cx, cy = Hh // 2, Ww // 2
    key = (Hh, Ww, arr2d.device)

    # cache (yy, xx, r_long)
    if key not in _GRID_CACHE:
        yy, xx = torch.meshgrid(
            torch.arange(Hh, device=arr2d.device),
            torch.arange(Ww, device=arr2d.device),
            indexing="ij"
        )
        r = torch.sqrt((yy - cx).to(DTYPE) ** 2 + (xx - cy).to(DTYPE) ** 2)
        r_long = r.long()
        rmax = int(r_long.max())
        _GRID_CACHE[key] = (yy, xx, r_long, rmax)
    else:
        yy, xx, r_long, rmax = _GRID_CACHE[key]

    # flatten and ignore NaNs
    vals = arr2d.reshape(-1)
    rflat = r_long.reshape(-1)
    finite_mask = torch.isfinite(vals)
    if not finite_mask.any():
        radial = torch.full((int(rflat.max().item()) + 1,), float("nan"), dtype=DTYPE, device=arr2d.device)
        return radial, (cx, cy)

    vals = vals[finite_mask].to(DTYPE)
    bins = rflat[finite_mask]

    # sum and count per radius via bincount (vectorized)
    # NOTE: torch.bincount supports with weights; counts are integer bincount
    rmax_eff = int(bins.max().item())
    sums = torch.bincount(bins, weights=vals, minlength=rmax_eff + 1).to(DTYPE)
    cnts = torch.bincount(bins, minlength=rmax_eff + 1).to(DTYPE)

    # avoid division by zero
    radial = torch.full((rmax_eff + 1,), float("nan"), dtype=DTYPE, device=arr2d.device)
    nz = cnts > 0
    radial[nz] = sums[nz] / cnts[nz]
    return radial, (cx, cy)


# ------------------------------------------------------
# 3) HSIC with robust bandwidth + optional no-H centering
# ------------------------------------------------------
def _get_centering_H(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Get (and cache) the centering matrix H = I - 11^T/n of size n."""
    key = (n, dtype, device)
    H = _H_CACHE.get(key, None)
    if H is None:
        I = torch.eye(n, dtype=dtype, device=device)
        O = torch.ones((n, n), dtype=dtype, device=device) / n
        H = I - O
        _H_CACHE[key] = H
    return H


@torch.no_grad()
def _center_kernel_mean(K: Tensor) -> Tensor:
    """
    Center a Gram matrix without explicit H, using mean subtraction:
      Kc = K - row_mean - col_mean + grand_mean
    This is algebraically equivalent to H K H but saves one large matrix alloc.
    """
    row_mean = K.mean(dim=1, keepdim=True)
    col_mean = K.mean(dim=0, keepdim=True)
    grand_mean = K.mean()
    Kc = K - row_mean - col_mean + grand_mean
    return Kc


@torch.no_grad()
def _rbf_gram(vec: Tensor, sigma: Tensor | float | None) -> Tensor:
    """
    Construct RBF Gram matrix K with robust bandwidth handling.
    vec: shape (n,) or (n,1)
    sigma: if None -> median heuristic; if float/Tensor -> use as bandwidth.
    """
    v = vec.view(-1, 1)
    # pairwise squared Euclidean
    d2 = torch.cdist(v, v, p=2) ** 2

    if sigma is None:
        # median heuristic with robust fallback
        # pick strictly positive distances to avoid zeros on diagonal
        mask = d2 > 0
        if mask.any():
            m = torch.median(d2[mask])
            if torch.isfinite(m) and (m > 0):
                s = torch.sqrt(0.5 * m)
            else:
                # fallback: use 0.5 * quantile(0.5) if finite, else 1.0
                q = torch.quantile(d2[mask], 0.5)
                s = torch.sqrt(0.5 * q) if torch.isfinite(q) and (q > 0) else torch.tensor(1.0, device=v.device, dtype=v.dtype)
        else:
            # all distances zero → constant data; degenerate case
            s = torch.tensor(1.0, device=v.device, dtype=v.dtype)
    else:
        s = torch.as_tensor(sigma, device=v.device, dtype=v.dtype)

    K = torch.exp(-d2 / (2 * s ** 2))
    return K


@torch.no_grad()
def hsic_stat_torch(x: Tensor,
                    y: Tensor,
                    sigma_x: float | Tensor | None = None,
                    sigma_y: float | Tensor | None = None,
                    use_explicit_H: bool = False,
                    dtype: torch.dtype | None = None) -> Tensor:
    """
    Biased HSIC estimator with options:
      - robust bandwidth (median heuristic with fallback) unless sigma_* provided
      - centering without explicitly forming H (default), or with cached H (optional)
      - configurable dtype

    HSIC_biased = trace(Kc Lc) / (n-1)^2
    where Kc = H K H or K - rowmean - colmean + grand_mean.
    """
    dt = dtype or DTYPE
    x = x.to(dt)
    y = y.to(dt)
    n = x.numel()

    # construct Gram matrices with robust sigma
    Kx = _rbf_gram(x, sigma_x)
    Ky = _rbf_gram(y, sigma_y)

    if use_explicit_H:
        H = _get_centering_H(n, dt, x.device)
        Kc = H @ Kx @ H
        Lc = H @ Ky @ H
    else:
        Kc = _center_kernel_mean(Kx)
        Lc = _center_kernel_mean(Ky)

    # trace(Kc Lc) / (n-1)^2
    HSIC = torch.trace(Kc @ Lc) / ((n - 1) ** 2)
    return HSIC


# ------------------------------------------------------
# 4) HSIC map with options (dtype, per-frame std, fixed sigma)
# ------------------------------------------------------
@torch.no_grad()
def hsic_map_rbf_torch(field: Tensor,
                       max_shift: int = 15,
                       average_frames: int | None = None,
                       radial: bool = True,
                       max_samples: int = 3000,
                       spatial_stride: int = 1,
                       device: str = "cuda",
                       seed: int = 0,
                       show_progress: bool = True,
                       dtype: torch.dtype | None = None,
                       per_frame_standardize: bool = False,
                       sigma_x: float | Tensor | None = None,
                       sigma_y: float | Tensor | None = None,
                       use_explicit_H: bool = False):
    """
    Compute HSIC map over spatial shifts.

    Parameters (new / key)
    ----------------------
    dtype : torch.dtype or None
        Compute dtype (default uses global DTYPE).
    per_frame_standardize : bool
        If True, subtract per-frame mean and divide per-frame std (robust to temporal drift).
    sigma_x, sigma_y : float/Tensor/None
        If provided, use fixed bandwidths for RBF on X and Y; otherwise use robust median heuristic.
    use_explicit_H : bool
        If True, center with Kc = H K H using cached H; by default use mean-centering (no explicit H).

    Returns
    -------
    hsic_map : np.ndarray
    radial_mean : np.ndarray | None
    """
    dt = dtype or DTYPE
    if not torch.is_tensor(field):
        field = torch.as_tensor(field, dtype=dt)
    else:
        field = field.to(dtype=dt)

    # move to device
    field = field.to(device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # optional temporal averaging
    if average_frames is not None:
        field = field[:average_frames]

    # standardization
    if per_frame_standardize:
        # per-frame mean/std: shape (T,1,1)
        mu = field.mean(dim=(1, 2), keepdim=True)
        std = field.std(dim=(1, 2), keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        field = (field - mu) / std
    else:
        # global mean-only removal (your original behavior)
        field = field - field.mean()

    shifts = range(-max_shift, max_shift + 1)
    size = 2 * max_shift + 1
    hsic_map = torch.full((size, size), float("nan"), dtype=dt, device=device)

    iterator = tqdm(shifts, desc="HSIC shifts", ncols=90, disable=not show_progress)
    for i, dx in enumerate(iterator):
        for j, dy in enumerate(shifts):
            if dx == 0 and dy == 0:
                continue
            xvals, yvals = sample_pairs_for_shift_torch(
                field, dx, dy, max_samples=max_samples,
                spatial_stride=spatial_stride, generator=gen
            )
            hsic_map[i, j] = hsic_stat_torch(
                xvals, yvals,
                sigma_x=sigma_x, sigma_y=sigma_y,
                use_explicit_H=use_explicit_H,
                dtype=dt
            )

    radial_mean = None
    if radial:
        radial_mean, _ = radial_mean_from_map_torch(hsic_map)

    # move back to CPU/np for plotting
    return hsic_map.detach().cpu().numpy(), (None if radial_mean is None else radial_mean.detach().cpu().numpy())


# ------------------------------------------------------
# 5) Nonlinear correlation length + plotting (unchanged)
# ------------------------------------------------------
def nonlinear_corr_length_from_hsic(radial_curve: np.ndarray):
    if radial_curve is None or not np.isfinite(radial_curve).any():
        return None
    base = np.nanmean(radial_curve[1:4]) if radial_curve.size > 4 else np.nanmean(radial_curve[1:])
    # guard base
    if not np.isfinite(base) or base <= 0:
        return None
    thr = 0.05 * base
    idx = np.where(radial_curve < thr)[0]
    return int(idx[0]) if idx.size > 0 else None


def plot_hsic_radials(radials,
                      labels=None,
                      normalize="first_band",
                      xlim=None,
                      ylim=None,
                      title="HSIC radial curves (block-wise)",
                      figsize=(6, 4),
                      smooth=None):
    """
    Plot radial HSIC curves explicitly (e.g., one curve per block).
    """
    if labels is None:
        labels = [f"block {k}" for k in range(len(radials))]

    def _ma(arr, w):
        if w is None or w < 3:
            return arr
        w = int(w)
        if w % 2 == 0:
            w += 1
        pad = w // 2
        padv = np.pad(arr, (pad, pad), mode="edge")
        ker = np.ones(w, dtype=float) / w
        return np.convolve(padv, ker, mode="valid")

    def _normalize(arr):
        if arr is None or not np.isfinite(arr).any():
            return arr
        if normalize == "none":
            return arr
        if normalize == "max":
            m = np.nanmax(arr)
            return arr / m if np.isfinite(m) and m != 0 else arr
        if normalize == "first_band":
            if arr.size > 4:
                base = np.nanmean(arr[1:4])
            else:
                base = np.nanmean(arr[1:]) if arr.size > 1 else np.nanmean(arr)
            return arr / base if np.isfinite(base) and base != 0 else arr
        return arr

    fig, ax = plt.subplots(figsize=figsize)
    for r, lab in zip(radials, labels):
        rr = r
        if rr is None:
            continue
        rr = _normalize(rr)
        rr = _ma(rr, smooth)
        ax.plot(rr, label=lab)

    ax.set_xlabel("radius r (pixels)")
    ax.set_ylabel("HSIC (normalized)" if normalize != "none" else "HSIC")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if labels is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


# ------------------------------------------------------
# 6) Block-wise HSIC (compatible; passes options through)
# ------------------------------------------------------
def hsic_blockwise(field,
                   blocks=(2, 2),
                   max_shift=15,
                   average_frames=None,
                   radial=True,
                   max_samples=3000,
                   spatial_stride=1,
                   device=None,
                   seed=0,
                   show_progress=False,
                   dtype: torch.dtype | None = None,
                   per_frame_standardize: bool = False,
                   sigma_x: float | Tensor | None = None,
                   sigma_y: float | Tensor | None = None,
                   use_explicit_H: bool = False):
    """
    Compute HSIC maps and radial curves per spatial block (locally stationary assumption).
    New options are forwarded to hsic_map_rbf_torch for consistency.
    """
    # resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ensure tensor
    dt = dtype or DTYPE
    if not torch.is_tensor(field):
        field_t = torch.as_tensor(field, dtype=dt)
    else:
        field_t = field.to(dtype=dt)

    if field_t.ndim != 3:
        raise ValueError("`field` must be a 3D array / tensor with shape (T, X, Y).")
    T, X, Y = field_t.shape

    # reuse your subblocks indexer (assumed already defined in your codebase)
    regions = _subblocks_indices(X, Y, blocks=blocks)

    results = {
        "blocks": blocks,
        "per_block": []
    }

    b = 0
    for i in range(blocks[0]):
        for j in range(blocks[1]):
            xs, ys = regions[b]
            b += 1

            sub = field_t[:, xs, ys]
            hsic_map, hsic_r = hsic_map_rbf_torch(
                sub,
                max_shift=max_shift,
                average_frames=average_frames,
                radial=radial,
                max_samples=max_samples,
                spatial_stride=spatial_stride,
                device=device,
                show_progress=show_progress,
                seed=seed,
                dtype=dt,
                per_frame_standardize=per_frame_standardize,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                use_explicit_H=use_explicit_H
            )

            xi_nl = nonlinear_corr_length_from_hsic(hsic_r)

            results["per_block"].append({
                "block_index": (i, j),
                "xslice": xs,
                "yslice": ys,
                "hsic_map": hsic_map,
                "radial": hsic_r,
                "xi_nl": xi_nl
            })

    return results


# noise0 = torch.as_tensor(noise0, dtype=DTYPE)  # (T, 64, 64)
# # ---------- 1) 全局 HSIC ----------
# hsic_map, hsic_r = hsic_map_rbf_torch(
#     noise0,
#     max_shift=15,
#     average_frames=10,
#     radial=True,
#     max_samples=3000,
#     spatial_stride=2,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     per_frame_standardize=True,     # 每帧标准化 (optional)
#     show_progress=True
# )

# xi_global = nonlinear_corr_length_from_hsic(hsic_r)
# print(f"Global nonlinear correlation length ξ_nl ≈ {xi_global}")

# # --- 可视化 ---
# plt.figure(figsize=(5, 5))
# plt.imshow(hsic_map, cmap="viridis", origin="lower")
# plt.colorbar(label="HSIC value")
# plt.title("Global HSIC map (RBF kernel)")
# plt.tight_layout()
# plt.show()

# # 使用你写好的函数绘制径向曲线
# plot_hsic_radials([hsic_r], labels=["Global field"], normalize="first_band", smooth=5)
# plt.show()


# # ---------- 2) 分块 HSIC ----------
# results = hsic_blockwise(
#     noise0,
#     blocks=(2, 2),                  # 2x2 blocks
#     max_shift=15,
#     average_frames=10,
#     radial=True,
#     max_samples=3000,
#     spatial_stride=2,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     per_frame_standardize=True,
#     show_progress=False
# )

# # 提取径向曲线与 block 标签
# radials = [b["radial"] for b in results["per_block"]]
# labels = [f"block {b['block_index']} (ξ={b['xi_nl']})" for b in results["per_block"]]

# # --- 使用同一个绘图函数绘制多条径向曲线 ---
# plot_hsic_radials(radials, labels=labels, normalize="first_band", smooth=5)
# plt.title("Block-wise HSIC radial curves")
# plt.show()



