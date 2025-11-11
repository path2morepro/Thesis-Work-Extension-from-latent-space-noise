# Improved temporal correlation diagnostics with normalized distance and block-bootstrap CIs
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional
from data import SQGData

def rolling_mean_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflect padding."""
    if window <= 1:
        return arr.astype(np.float64, copy=True)
    w = int(window)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    arr = np.asarray(arr, dtype=np.float64)
    padv = np.pad(arr, (pad, pad), mode="reflect")
    ker = np.ones(w, dtype=float) / w
    out = np.convolve(padv, ker, mode="valid")
    return out

def temporal_detrend_per_pixel(field: np.ndarray, window: int = 7) -> np.ndarray:
    """Detrend each pixel over time by subtracting a moving average.
    field: (T, X, Y)
    """
    T, X, Y = field.shape
    out = np.empty_like(field, dtype=np.float64)
    for i in range(X):
        for j in range(Y):
            series = field[:, i, j].astype(np.float64)
            trend = rolling_mean_1d(series, window=window)
            out[:, i, j] = series - trend
    return out

def acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased ACF normalized to 1 at lag 0. Returns shape (max_lag+1,)"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    if var <= 0 or len(x) == 0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2
    ac = ac[mid: mid + max_lag + 1] / (len(x) * var)
    return ac

def block_slices(X: int, Y: int, blocks: Tuple[int, int]) -> List[Tuple[slice, slice]]:
    bx, by = blocks
    xs = np.linspace(0, X, bx + 1, dtype=int)
    ys = np.linspace(0, Y, by + 1, dtype=int)
    regions = []
    for i in range(bx):
        for j in range(by):
            regions.append((slice(xs[i], xs[i+1]), slice(ys[j], ys[j+1])))
    return regions

def _circular_block_bootstrap_1d(y: np.ndarray, L: int, B: int) -> np.ndarray:
    """Return bootstrap resamples as a 2D array (B, T) using circular block bootstrap.
    y shape: (T,)
    """
    y = np.asarray(y, dtype=np.float64)
    T = len(y)
    if T == 0:
        return np.empty((B, 0), dtype=np.float64)
    if L <= 1:
        # ordinary bootstrap on indices (less ideal for time series but fallback)
        idx = np.random.randint(0, T, size=(B, T))
        return y[idx]
    # wrap for circular indexing
    y_wrap = np.concatenate([y, y[:L-1]])
    n_blocks = int(np.ceil(T / L))
    out = np.empty((B, T), dtype=np.float64)
    for b in range(B):
        pieces = []
        for _ in range(n_blocks):
            start = np.random.randint(0, T)  # start anywhere
            pieces.append(y_wrap[start:start+L])
        boot = np.concatenate(pieces)[:T]
        out[b] = boot
    return out

def _pairwise_acf_similarity(acf_blocks: np.ndarray, drop_lag0: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise shape distances/correlations between block ACF curves.
    Distance is 1 - Pearson correlation on centered ACF vectors (scale-invariant).
    Returns:
      dists: 1 - corr per pair
      cors: Pearson correlation per pair
    """
    B, L = acf_blocks.shape
    start = 1 if drop_lag0 else 0
    curves = acf_blocks[:, start:]
    # center each curve to remove scale/offset
    curves = curves - curves.mean(axis=1, keepdims=True)
    # compute pairwise correlations
    idx_i, idx_j = np.triu_indices(B, k=1)
    dists = []
    cors = []
    for i, j in zip(idx_i, idx_j):
        a = curves[i]; b = curves[j]
        denom = np.sqrt(np.dot(a,a) * np.dot(b,b)) + 1e-12
        c = float(np.dot(a,b) / denom)
        cors.append(c)
        dists.append(1.0 - c)
    return np.array(dists, dtype=np.float64), np.array(cors, dtype=np.float64)


def _circular_block_indices(T: int, L: int) -> np.ndarray:
    """Build a length-T index array by concatenating circular blocks of length L."""
    if L <= 1:
        return np.random.randint(0, T, size=T)
    y_idx = np.arange(T)
    wrap = np.concatenate([y_idx, y_idx[:L-1]])
    n_blocks = int(np.ceil(T / L))
    blocks = []
    for _ in range(n_blocks):
        start = np.random.randint(0, T)
        blocks.append(wrap[start:start+L])
    idx = np.concatenate(blocks)[:T]
    return idx

def _default_block_len(T: int, acf_global: np.ndarray) -> int:
    """Choose a conservative default block length.
    Lower bounded at 8 to avoid overshort blocks that destroy low-frequency shape.
    """
    # Integrated autocorrelation time (positive tail)
    vals = acf_global[1:]
    pos = vals[vals>0]
    iat = 1.0 + 2.0 * np.sum(pos) if pos.size>0 else 1.0
    L_iat = int(round(iat))
    L_cube = int(round(T**(1/3)))  # Politis-White heuristic scale
    L = max(8, min(64, max(L_iat, L_cube)))  # clamp to [8, 64]
    return L
def temporal_stationarity_diagnostics(
    field: np.ndarray,
    max_lag: int = 50,
    blocks: Tuple[int, int] = (2, 2),
    detrend: bool = True,
    trend_window: int = 7,
    local_standardize: bool = False,
    thresholds: Optional[Dict[str, float]] = None,
    # Bootstrap options
    do_bootstrap: bool = True,
    B: int = 1000,
    block_len: Optional[int] = None,
    ci_level: float = 0.95,
) -> Dict[str, object]:
    """Diagnose temporal WSS and homogeneity of temporal correlation across subspaces.
    Adds normalized ACF distance (1 - corr) and circular block-bootstrap CIs.
    Returns dict with metrics and (if enabled) CIs.
    """
    if thresholds is None:
        thresholds = dict(
            mu_cv_max=0.1,
            sigma_ratio_max=1.5,
            # new distance is 1 - corr median (scale/offset invariant)
            acf_dist_median_max=0.15,   # ~ corr_median >= 0.85
            acf_corr_median_min=0.85
        )

    assert field.ndim == 3, "`field` must be (T, X, Y)"
    T, X, Y = field.shape
    F = field.astype(np.float64, copy=False)

    if detrend:
        F = temporal_detrend_per_pixel(F, window=trend_window)

    if local_standardize:
        mu_t = F.mean(axis=(1,2), keepdims=True)
        sd_t = F.std(axis=(1,2), keepdims=True)
        sd_t[sd_t == 0] = 1.0
        F = (F - mu_t) / sd_t

    # A1: mean/variance stability
    mu_t = F.mean(axis=(1,2))
    sigma_t = F.std(axis=(1,2)) + 1e-12
    mu_cv = float(np.std(mu_t) / (np.std(F) + 1e-12))
    sigma_ratio = float(np.max(sigma_t) / max(np.min(sigma_t), 1e-12))

    # global series ACF for visualization
    global_series = F.mean(axis=(1,2))
    acf_global = acf_1d(global_series, max_lag=max_lag)

    # A2: block-wise ACFs
    regs = block_slices(X, Y, blocks)
    acf_blocks = []
    series_blocks = []  # keep for bootstrap
    for sx, sy in regs:
        block_series = F[:, sx, sy].mean(axis=(1,2))
        series_blocks.append(block_series)
        acf_blocks.append(acf_1d(block_series, max_lag=max_lag))
    acf_blocks = np.stack(acf_blocks, axis=0)  # (B, L+1)
    Bblocks = acf_blocks.shape[0]

    # normalized, scale-invariant distance: 1 - corr on centered curves
    dists, cors = _pairwise_acf_similarity(acf_blocks, drop_lag0=True)
    dist_median = float(np.median(dists)) if dists.size else 0.0
    corr_median = float(np.median(cors)) if cors.size else 1.0

    th = thresholds
    A1_ok = (mu_cv <= th["mu_cv_max"]) and (sigma_ratio <= th["sigma_ratio_max"])
    A2_ok = (dist_median <= th["acf_dist_median_max"]) and (corr_median >= th["acf_corr_median_min"])

    if A1_ok and A2_ok:
        decision = "GLOBAL_TEMPORAL_OK"
        note = "时间二阶平稳且子空间间的时间相关形状一致；可用全局共享的时间模型。"
    elif A1_ok and not A2_ok:
        decision = "LOCAL_TEMPORAL_ONLY"
        note = "时间近似平稳，但不同子空间的时间相关形状存在差异；建议局部时间模型或空间自适应时间核。"
    elif (not A1_ok) and A2_ok:
        decision = "DETREND_OR_VARIANCE_STABILIZE"
        note = "时间非平稳（均值/方差随 t 变动），但各子空间时间相关形状相似；需更强去趋势/稳定化。"
    else:
        decision = "NONSTATIONARY_AND_HETEROGENEOUS"
        note = "时间非平稳且子空间时间相关不一致；建议局部+分段时间建模。"

    # ---------- Bootstrap CIs ----------
    ci = {}
    if do_bootstrap:
        # choose block length conservatively
        if block_len is None:
            Lb = _default_block_len(T, acf_global)
        else:
            Lb = int(block_len)

        Bboot = int(B)
        dist_samples = np.empty(Bboot, dtype=np.float64)
        corr_samples = np.empty(Bboot, dtype=np.float64)

        # synchronized circular block bootstrap: same time indices for all subspaces
        for b in range(Bboot):
            idx = _circular_block_indices(T, Lb)
            boot_acfs = []
            for s in series_blocks:
                yb = s[idx]
                boot_acfs.append(acf_1d(yb, max_lag=max_lag))
            boot_acfs = np.stack(boot_acfs, axis=0)
            d_b, c_b = _pairwise_acf_similarity(boot_acfs, drop_lag0=True)
            dist_samples[b] = np.median(d_b) if d_b.size else 0.0
            corr_samples[b] = np.median(c_b) if c_b.size else 1.0

        alpha = 1.0 - float(ci_level)
        lo = 100*alpha/2.0
        hi = 100*(1.0 - alpha/2.0)
        ci["acf_dist_median"] = (float(np.percentile(dist_samples, lo)),
                                 float(np.percentile(dist_samples, hi)))
        ci["acf_corr_median"] = (float(np.percentile(corr_samples, lo)),
                                 float(np.percentile(corr_samples, hi)))
        ci["block_len_used"] = Lb
        ci["B"] = Bboot
        ci["level"] = ci_level
# ---------- Plots (compatible with Matplotlib>=3.8) ----------
# ---------- Plots (compatible with Matplotlib>=3.8) ----------
    # 1) mu_t
    plt.figure()
    plt.plot(mu_t)
    plt.title("μ_t over time (after detrending)" if detrend else "μ_t over time")
    plt.xlabel("t"); plt.ylabel("μ_t"); plt.grid(True, alpha=0.3); plt.tight_layout()

    # 2) sigma_t
    plt.figure()
    plt.plot(sigma_t)
    plt.title("σ_t over time (after detrending)" if detrend else "σ_t over time")
    plt.xlabel("t"); plt.ylabel("σ_t"); plt.grid(True, alpha=0.3); plt.tight_layout()

    # 3) Global ACF
    plt.figure()
    plt.plot(np.arange(len(acf_global)), acf_global, marker="o")
    plt.title("Global temporal ACF (of spatial mean series)")
    plt.xlabel("lag h"); plt.ylabel("ρ(h)"); plt.grid(True, alpha=0.3); plt.tight_layout()

    # 4) Block-wise ACFs
    plt.figure()
    for k in range(Bblocks):
        plt.plot(acf_blocks[k], label=f"block {k+1}")
    plt.title("Block-wise temporal ACFs (spatial means in subspaces)")
    plt.xlabel("lag h"); plt.ylabel("ρ(h)"); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

    metrics = dict(
        mu_cv=float(mu_cv),
        sigma_ratio=float(sigma_ratio),
        acf_dist_median=float(dist_median),
        acf_corr_median=float(corr_median),
        A1_time_WSS_ok=bool(A1_ok),
        A2_temporal_homogeneity_ok=bool(A2_ok),
        decision=decision,
        note=note,
        thresholds=thresholds,
    )
    out = dict(
        mu_t=mu_t,
        sigma_t=sigma_t,
        acf_global=acf_global,
        acf_blocks=acf_blocks,
        metrics=metrics,
    )
    if do_bootstrap:
        out["bootstrap"] = ci
    return out



sqg = SQGData('SQG.npy', 'inverted_SQG.npy')
noise0 = sqg.get_field()

res = temporal_stationarity_diagnostics(
    field=noise0,
    max_lag=50,
    blocks=(2,2),
    detrend=True,
    trend_window=7,
    local_standardize=False,
    do_bootstrap=True,
    B=1000,
    block_len=None,   # 现在默认会自动采用更保守的 L，通常 >= 8
    ci_level=0.95
)
print(res["metrics"])
print(res["bootstrap"])
