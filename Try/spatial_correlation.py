import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
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
        # Only keep directional ACFs, smoothed spectrum, and WSS stats
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))

        # Directional profiles (r in [r_lo, r_hi))
        for m in range(dir_profiles.shape[0]):
            axs[0].plot(np.arange(r_lo, r_hi), dir_profiles[m, r_lo:r_hi], alpha=0.8)
        axs[0].set_title(f" ACFs (n={n_angles})")
        axs[0].set_xlabel("r (px)")
        axs[0].set_ylabel("C(r)")

        # Smoothed spectrum
        im2 = axs[1].imshow(np.log1p(Ssm), cmap='magma', origin='lower')
        axs[1].set_title(f"Smoothed Spectrum (κ={results['kappa']:.2f})")
        plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

        # WSS block stats
        axs[2].bar(np.arange(len(mu_k))-0.2, mu_k, width=0.4, label='μ_k')
        axs[2].bar(np.arange(len(mu_k))+0.2, std_k, width=0.4, label='σ_k')
        axs[2].set_title(f"WSS blocks: std_ratio={std_ratio:.2f}, Δ_med={delta_median:.3f}")
        axs[2].legend()

        plt.tight_layout()
        plt.show()

        print("【决策】", decision)
        print("说明：", note)

    return results


def calibrate_thresholds(
    n_rep=200,
    average_frames=10,
    H=64, W=64,
    blocks=(2, 2),
    n_angles=8,
    q=0.95,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # preallocate
    max_mu = np.empty(n_rep)
    std_ratio = np.empty(n_rep)
    delta_med = np.empty(n_rep)
    D_med = np.empty(n_rep)
    kappa = np.empty(n_rep)

    for i in range(n_rep):
        # only generate what you use
        white_noise = rng.standard_normal((average_frames, H, W))

        baseline = spatial_correlation_diagnostics(
            white_noise,
            average_frames=average_frames,  # function will average these frames
            blocks=blocks,
            n_angles=n_angles,
            r_max=None,
            show_plots=False
        )

        max_mu[i] = np.max(np.abs(baseline["block_means"]))
        std_ratio[i] = baseline["std_ratio"]
        delta_med[i] = baseline["block_acf_shape_delta_median"]
        D_med[i] = baseline["directional_dispersion_median"]
        kappa[i] = baseline["kappa"]

    thresholds = {
        "mu_abs_max": np.quantile(max_mu, q),
        "std_ratio_max": np.quantile(std_ratio, q),
        "delta_median_max": np.quantile(delta_med, q),
        "D_median_max": np.quantile(D_med, q),
        "kappa_max": np.quantile(kappa, q),
    }
    return thresholds

# 生成阈值
# thresholds = calibrate_thresholds(n_rep=100, q=0.99, seed=42)
# print(thresholds)


# 调用示例
# results = spatial_correlation_diagnostics(noise0, average_frames=10, blocks=(2,2), n_angles=8, r_max=None, show_plots=True, thresholds=thresholds)
# 和阈值比较的几个指标
# print(results['block_means'])
# print(results['std_ratio'])
# print(results['block_acf_shape_delta_median'])
# print(results['directional_dispersion_median'])
# print(results['kappa'])

# 进一步诊断
# results_local = spatial_correlation_diagnostics(noise0,
#     average_frames=10,
#     use_local_standardize=True,
#     show_plots=True,
#     thresholds=thresholds
# )

# print(results_local['block_means'])
# print(results_local['std_ratio'])
# print(results_local['block_acf_shape_delta_median'])
# print(results_local['directional_dispersion_median'])
# print(results_local['kappa'])