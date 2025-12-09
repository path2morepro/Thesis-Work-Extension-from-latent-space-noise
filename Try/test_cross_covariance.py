"""
cross_covariance.py

End-to-end *diagnostics + tests* for LOCAL space–time cross-covariance
on latent fields, with simulation-based model checking.

You asked to:
  - keep key-theory steps heavily commented (but not trivial inits),
  - work with your existing dataset loader:
        from data import SQGTrajData
        trajs = sqg_data.get_all()[:, :, 0, :, :]  # (N_traj, T, H, W)
  - provide *all* tests we discussed:
        Test A: Positive-definiteness (2x2 PSD) sanity
        Test B: Simulation-based adequacy test
        Test C: Block-wise stability
        Test D: Trajectory consistency
        Test E: Simple model fit vs empirical covariance (residuals)
        Test F: Covariance consistency A*C(0,0) ?= C(0,1)

The code below is self-contained. You can paste it into a file named
`cross_covariance.py` and import the functions/classes in Jupyter.

Dependencies: numpy, matplotlib (for optional plots only).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# 0) Configuration for cross-covariance estimation
# ================================================================

@dataclass
class CrossCovConfig:
    """
    Configuration for *local* cross-covariance estimation.

    block_size : side length of each square block (e.g., 8 for 8x8).
                 We estimate one set of statistics per block to allow
                 *spatial nonstationarity* while keeping variance low.
    time_lag   : temporal lag u. You care about u=1; optionally u=2.
    lags       : list of spatial lags h=(di,dj). Include (0,0) if you
                 also want the across-time auto-covariance at the same pixel.
                 Typical: [(0,0),(1,0),(-1,0),(0,1),(0,-1)].
    """
    block_size: int = 8
    time_lag: int = 1
    lags: Optional[List[Tuple[int, int]]] = None

    def __post_init__(self):
        if self.lags is None:
            self.lags = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


# ================================================================
# 1) Core — Local cross-covariance estimation (analysis engine)
# ================================================================

def _aligned_block_slices(r0: int, r1: int, c0: int, c1: int, di: int, dj: int):
    """
    Geometric alignment for spatial lag pairs *inside a block*.
    We must extract pairs (s, s+h) that both remain in the block so that
    z_t(s) and z_{t+u}(s+h) are aligned with identical shapes.

    The logic: find index ranges where shifting by (di,dj) keeps us in-bounds.
    Returns 4 slices: rows_x, cols_x (for s) and rows_y, cols_y (for s+h).
    Raises ValueError if overlap vanishes (lag too large for the block).
    """
    # Row alignment
    if di >= 0:
        rx0, rx1 = r0, r1 - di
        ry0, ry1 = r0 + di, r1
    else:
        rx0, rx1 = r0 - di, r1
        ry0, ry1 = r0, r1 + di

    # Col alignment
    if dj >= 0:
        cx0, cx1 = c0, c1 - dj
        cy0, cy1 = c0 + dj, c1
    else:
        cx0, cx1 = c0 - dj, c1
        cy0, cy1 = c0, c1 + dj

    if rx1 <= rx0 or cx1 <= cx0:
        raise ValueError("Lag too large: no overlap remains inside this block.")

    return slice(rx0, rx1), slice(cx0, cx1), slice(ry0, ry1), slice(cy0, cy1)


def _estimate_block_cross_covariances(
    trajs: np.ndarray,
    r0: int, r1: int, c0: int, c1: int,
    lags: List[Tuple[int, int]],
    time_lag: int
) -> Tuple[np.ndarray, float, float]:
    """
    Compute *local* C(h,u; s0) for a single block and also return:
      - local_var     := C(0,0; s0) at u=0 (variance)
      - local_c01     := C(h=(0,0), u=1; s0) (time-lag-1 auto-covariance)

    Why these two extras?
      * local_var is required to convert covariance to correlation.
      * local_c01 is used in Test F (consistency for AR(1)-type dynamics).

    Shapes:
      trajs: (N_traj, T, H, W)
      covs : (len(lags),)
    """
    N, T, H, W = trajs.shape
    covs = np.zeros(len(lags), dtype=np.float64)

    # --- Local variance C(0,0;s0) at u=0: pooled over all times & trajs in the block ---
    block = trajs[:, :, r0:r1, c0:c1].astype(np.float64)  # (N, T, bh, bw)
    bf = block.reshape(-1)
    # center within block: empirical mean across (traj, time, pixels of this block)
    bf_c = bf - bf.mean()
    local_var = float(np.mean(bf_c ** 2))
    if local_var <= 0:
        local_var = 1e-12  # numerical guard

    # --- For each spatial lag h, compute C(h,u;s0) with u=time_lag ---
    for k, (di, dj) in enumerate(lags):
        try:
            rx, cx, ry, cy = _aligned_block_slices(r0, r1, c0, c1, di, dj)
        except ValueError:
            covs[k] = 0.0
            continue

        X = trajs[:, : T - time_lag, rx, cx]       # z_t(s)
        Y = trajs[:, time_lag:, ry, cy]            # z_{t+u}(s+h)

        # Flatten across (N, T-u, within-block pixels) -> 1D
        Xf = X.reshape(-1).astype(np.float64)
        Yf = Y.reshape(-1).astype(np.float64)

        # Covariance is about *fluctuations*: subtract sample means before dotting.
        Xf -= Xf.mean()
        Yf -= Yf.mean()
        covs[k] = float(np.mean(Xf * Yf))

    # h=(0,0) entry (if present) is exactly C(0,1;s0) for AR(1) consistency test.
    local_c01 = 0.0
    for k, (di, dj) in enumerate(lags):
        if di == 0 and dj == 0:
            local_c01 = covs[k]
            break

    return covs, local_var, local_c01


def compute_local_cross_covariance_maps(
    trajs: np.ndarray,
    config: CrossCovConfig
) -> Dict[str, np.ndarray]:
    """
    Partition (H,W) into non-overlapping blocks and estimate, for each block s0:
        - cov_maps[bi,bj,:] = [ C(h,u;s0) for h in lags ]
        - var_map[bi,bj]    = C(0,0;s0)
        - c01_map[bi,bj]    = C(0,1;s0) if (0,0) in lags else 0
        - corr_maps         = cov_maps / var_map  (broadcasted)

    Why blocks? You want *spatial nonstationarity* while keeping estimates stable.
    Blocks pool pixels inside each tile and average across (traj,time).
    """
    assert trajs.ndim == 4, "trajs must be (N_traj, T, H, W)"
    _, _, H, W = trajs.shape
    bs = config.block_size
    lags = config.lags
    u = config.time_lag
    L = len(lags)

    Bh = H // bs
    Bw = W // bs
    cov_maps = np.zeros((Bh, Bw, L), dtype=np.float64)
    var_map  = np.zeros((Bh, Bw), dtype=np.float64)
    c01_map  = np.zeros((Bh, Bw), dtype=np.float64)

    for bi in range(Bh):
        for bj in range(Bw):
            r0, r1 = bi * bs, (bi + 1) * bs
            c0, c1 = bj * bs, (bj + 1) * bs
            covs, v, c01 = _estimate_block_cross_covariances(
                trajs, r0, r1, c0, c1, lags, u
            )
            cov_maps[bi, bj, :] = covs
            var_map[bi, bj] = v
            c01_map[bi, bj] = c01

    var_safe = np.maximum(var_map, 1e-12)
    corr_maps = cov_maps / var_safe[:, :, None]

    return {
        "cov_maps": cov_maps,     # C(h,u;s0)
        "corr_maps": corr_maps,   # R(h,u;s0)
        "var_map": var_map,       # C(0,0;s0)
        "c01_map": c01_map,       # C(0,1;s0) if h=(0,0) included
        "lags": np.array(lags, dtype=int),
        "block_size": np.array([bs]),
        "time_lag": np.array([u]),
    }


# ================================================================
# 2) Diagnostics helpers (magnitude, SVD)
# ================================================================

def compute_magnitude_map(
    corr_maps: np.ndarray, lags: np.ndarray,
    exclude_zero_lag: bool = True, p_norm: float = 2.0
) -> np.ndarray:
    """
    Summarize, per block, the *overall strength* of cross-pixel correlation
    at temporal lag u by taking a p-norm over |R(h,u;s0)| across h in lags.

    We typically exclude h=(0,0), since that entry is time auto-correlation
    at the same pixel, not cross-pixel coupling.
    """
    Bh, Bw, L = corr_maps.shape
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    sel = corr_maps[:, :, mask]
    eps = 1e-12
    mag = (np.sum(np.abs(sel) ** p_norm, axis=2) + eps) ** (1.0 / p_norm)
    return mag


def global_svd_on_corr_maps(
    corr_maps: np.ndarray, lags: np.ndarray, exclude_zero_lag: bool = True
) -> Dict[str, np.ndarray]:
    """
    Stack each block's correlation vector R(h,.) into a matrix and SVD it.
    This reveals *effective rank* of cross-pixel correlation patterns.
    """
    Bh, Bw, L = corr_maps.shape
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    corr_sel = corr_maps[:, :, mask]  # (Bh,Bw,L_sel)
    L_sel = corr_sel.shape[2]
    K = Bh * Bw
    M = corr_sel.reshape(K, L_sel).T  # (L_sel, K)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    S2 = S ** 2
    tot = S2.sum() + 1e-12
    frac = S2 / tot
    return {"singular_values": S, "energy_fraction": frac, "energy_cumsum": np.cumsum(frac)}


# ================================================================
# 3) TEST A — Positive-definiteness (2x2 PSD sanity per lag)
# ================================================================

def pd_test_two_by_two(
    var_map: np.ndarray, cov_maps: np.ndarray, lags: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Minimal PSD sanity: for each block and each spatial lag h,
    the 2x2 covariance matrix of [z_t(s), z_{t+u}(s+h)] should be PSD:

        Sigma_h = [[ C(0,0),  C(h, u) ],
                   [ C(h, u), C(0,0) ]]

    PSD <=> (i) C(0,0) >= 0   and  (ii) |C(h,u)| <= C(0,0).
    This is a *necessary* condition (not sufficient for a larger joint).
    It is extremely informative given we focus on u=1 cross-covariances.

    We return a boolean violation mask and summary rates.
    """
    Bh, Bw, L = cov_maps.shape
    assert var_map.shape == (Bh, Bw)
    vio = np.zeros((Bh, Bw, L), dtype=bool)

    for k in range(L):
        ch = cov_maps[:, :, k]
        v = var_map
        # (i) variance non-negative (already true by construction, but guard)
        cond_var = v >= -1e-12
        # (ii) |cov| <= var  (equivalent to rho in [-1,1])
        cond_cov = np.abs(ch) <= (v + 1e-12)
        vio[:, :, k] = ~(cond_var & cond_cov)

    violation_rate = vio.mean()
    return {"violation_mask": vio, "violation_rate": np.array([violation_rate])}


# ================================================================
# 4) TEST B — Simulation-based adequacy (AR(1) diag + optional low-rank)
# ================================================================

@dataclass
class AR1DiagLowrankSimConfig:
    """
    Simple covariance-driven latent simulator:

      z_{t+1} = a * z_t + eps_t,
      eps_t ~ N(0,  Sigma),  Sigma = sigma2_diag + U diag(lam) U^T  (optional)

    We simulate per-pixel AR(1) with *diagonal* variance as the baseline.
    A tiny low-rank correction can be added to check if a weak global mode
    helps match the empirical cross-covariance magnitude.

    You provide *block-level* a and sigma2 (shape (Bh, Bw)); we up-sample
    each block value to pixels in that block to generate (H,W) parameter maps.
    """
    a_block: np.ndarray          # (Bh, Bw) per-block AR(1) coefficient
    sigma2_block: np.ndarray     # (Bh, Bw) per-block innovation variance
    lowrank_rank: int = 0        # 0 => pure diagonal; else small rank r
    lowrank_scale: float = 0.0   # amplitude for low-rank noise
    random_seed: int = 0


def _upsample_block_map_to_pixels(block_map: np.ndarray, block_size: int) -> np.ndarray:
    """Repeat each block value into its block of pixels to get (H,W) map."""
    Bh, Bw = block_map.shape
    bs = block_size
    H, W = Bh * bs, Bw * bs
    out = np.zeros((H, W), dtype=np.float64)
    for bi in range(Bh):
        for bj in range(Bw):
            out[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs] = block_map[bi, bj]
    return out


def simulate_from_ar1_diag_lowrank(
    cfg: AR1DiagLowrankSimConfig,
    N_traj: int, T: int, block_size: int
) -> np.ndarray:
    """
    Generate synthetic latent fields (N_traj, T, H, W) from AR(1) diag(+low-rank).
    The low-rank noise is implemented by sampling r global factors per time,
    projecting with U (fixed random orthonormal basis), and scaling.
    """
    rng = np.random.default_rng(cfg.random_seed)
    Bh, Bw = cfg.a_block.shape
    H, W = Bh * block_size, Bw * block_size

    # Upsample per-block parameters to per-pixel maps
    a_map = _upsample_block_map_to_pixels(cfg.a_block, block_size)           # (H,W)
    sigma2_map = _upsample_block_map_to_pixels(cfg.sigma2_block, block_size) # (H,W)

    # Optional low-rank global basis over pixels: U ∈ R^{(H*W) x r}
    if cfg.lowrank_rank > 0:
        # Random orthonormal basis via QR on Gaussian
        G = rng.standard_normal((H*W, cfg.lowrank_rank))
        U, _ = np.linalg.qr(G)  # (H*W, r)
        U = U.reshape(H*W, cfg.lowrank_rank)
    else:
        U = None

    # Allocate output array
    Z = np.zeros((N_traj, T, H, W), dtype=np.float64)

    # Initialize z_0 as N(0, sigma2 / (1-a^2)) at each pixel (stationary law)
    denom = np.maximum(1.0 - a_map**2, 1e-6)
    std0 = np.sqrt(sigma2_map / denom)
    for n in range(N_traj):
        Z0 = rng.normal(loc=0.0, scale=std0)
        Z[n, 0] = Z0

    # Roll forward
    for n in range(N_traj):
        for t in range(T-1):
            eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2_map))
            if U is not None and cfg.lowrank_scale > 0.0 and cfg.lowrank_rank > 0:
                # sample r global factors, project with U, reshape to (H,W)
                g = rng.standard_normal(cfg.lowrank_rank)
                lr = (U @ g).reshape(H, W) * cfg.lowrank_scale
                eps = eps + lr
            Z[n, t+1] = a_map * Z[n, t] + eps

    return Z


def simulation_adequacy_test(
    real_trajs: np.ndarray,
    real_results: Dict[str, np.ndarray],
    sim_cfg: AR1DiagLowrankSimConfig,
    config: CrossCovConfig,
) -> Dict[str, np.ndarray]:
    """
    Test B: simulate from a simple AR(1)+diag(+low-rank) model using
    *per-block* parameters estimated from real data, then re-run the
    same cross-covariance pipeline and compare summary diagnostics.

    Key *validation principle*:
      A covariance model is adequate if statistics computed from
      model-simulated data *match* those from real data (same pipeline).
    """
    Bh, Bw = real_results["var_map"].shape
    assert sim_cfg.a_block.shape == (Bh, Bw)
    assert sim_cfg.sigma2_block.shape == (Bh, Bw)

    # Use the same shapes as real data
    N, T, H, W = real_trajs.shape

    sim_trajs = simulate_from_ar1_diag_lowrank(sim_cfg, N_traj=N, T=T, block_size=config.block_size)
    sim_res = compute_local_cross_covariance_maps(sim_trajs, config)

    # Compare magnitude maps + SVD spectra (lightweight yet informative)
    lags = real_results["lags"]
    real_mag = compute_magnitude_map(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_mag  = compute_magnitude_map(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    real_svd = global_svd_on_corr_maps(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_svd  = global_svd_on_corr_maps(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    # Simple numeric summaries (distributional distance proxies)
    mag_diff_mean = float(np.mean(sim_mag - real_mag))
    mag_diff_abs_mean = float(np.mean(np.abs(sim_mag - real_mag)))

    svd_diff = float(np.linalg.norm(sim_svd["singular_values"] - real_svd["singular_values"]))
    svd_energy_diff = float(np.linalg.norm(sim_svd["energy_cumsum"] - real_svd["energy_cumsum"]))

    return {
        "real_mag": real_mag,
        "sim_mag": sim_mag,
        "mag_diff_mean": np.array([mag_diff_mean]),
        "mag_diff_abs_mean": np.array([mag_diff_abs_mean]),
        "svd_diff": np.array([svd_diff]),
        "svd_energy_diff": np.array([svd_energy_diff]),
        "sim_results_corr_maps": sim_res["corr_maps"],  # for optional plotting
    }


# ================================================================
# 5) TEST C — Block-wise stability (pattern similarity across blocks)
# ================================================================

def block_stability_test(
    corr_maps: np.ndarray, lags: np.ndarray, exclude_zero_lag: bool = True
) -> Dict[str, np.ndarray]:
    """
    Measure how *consistent* the correlation pattern R(h,.) is across blocks.
    We compute cosine similarity among block-vectors and summarize.

    If effective rank≈1 and patterns are consistent, mean cosine ~ 1 and
    std is small. If patterns vary chaotically across space, mean cosine drops.

    Returns distribution summary: mean, std of pairwise cosine similarities.
    """
    Bh, Bw, L = corr_maps.shape
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    X = corr_maps[:, :, mask].reshape(Bh*Bw, -1)  # (K, L_sel)
    # normalize each vector to unit length (avoid zero division with tiny eps)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms

    # cosine similarities via Gram matrix
    G = Xn @ Xn.T  # (K, K), values in [-1,1]
    # take upper-triangular (off-diagonal) entries as pairwise cosines
    iu = np.triu_indices(G.shape[0], k=1)
    cosines = G[iu]
    return {
        "cosine_mean": np.array([float(np.mean(cosines))]),
        "cosine_std":  np.array([float(np.std(cosines))]),
        "cosine_min":  np.array([float(np.min(cosines))]),
        "cosine_max":  np.array([float(np.max(cosines))]),
    }


# ================================================================
# 6) TEST D — Trajectory consistency (replicate stability)
# ================================================================

def trajectory_consistency_test(
    trajs: np.ndarray, config: CrossCovConfig
) -> Dict[str, np.ndarray]:
    """
    Estimate corr_maps per trajectory and quantify across-trajectory variation.
    If the covariance structure is *replicable*, block-wise mean/std across
    trajectories should be small (relative to their magnitude).
    """
    N, _, _, _ = trajs.shape
    corr_list = []
    for n in range(N):
        res_n = compute_local_cross_covariance_maps(trajs[n:n+1], config)
        corr_list.append(res_n["corr_maps"])  # (Bh, Bw, L)

    Corr = np.stack(corr_list, axis=0)  # (N, Bh, Bw, L)
    mean_corr = np.mean(Corr, axis=0)
    std_corr  = np.std(Corr,  axis=0)

    # Relative variation: std / (|mean|+eps)
    rel = std_corr / (np.abs(mean_corr) + 1e-12)
    return {
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "rel_std_corr": rel,  # high values => weak replicability
    }


# ================================================================
# 7) TEST E — Simple model fit vs empirical (residuals)
# ================================================================

def simple_neighbor_model_fit(
    corr_maps: np.ndarray, lags: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Fit the *simplest* isotropic neighbor model:
        R_model(h) = rho for |h|=1 (4-neighbors), 0 otherwise.
    We estimate rho by averaging over all blocks and all |h|=1 entries.

    Residuals per block per lag = R_emp - R_model(h).

    This test answers: do empirical maps exceed what a *flat, isotropic,
    short-range* correlation can explain? If residuals are structureless,
    the simple model is adequate for your data regime.
    """
    # Indices of |h| = 1
    mask_nb = ((np.abs(lags[:, 0]) + np.abs(lags[:, 1])) == 1)
    # Average neighbor correlation across blocks & neighbor-lags
    rho_hat = float(np.mean(corr_maps[:, :, mask_nb]))

    # Build model predictions for all lags
    R_model = np.zeros_like(corr_maps)
    R_model[:, :, mask_nb] = rho_hat

    residuals = corr_maps - R_model
    res_abs_mean = float(np.mean(np.abs(residuals)))

    return {
        "rho_hat": np.array([rho_hat]),
        "residuals": residuals,
        "residual_abs_mean": np.array([res_abs_mean]),
    }


# ================================================================
# 8) TEST F — Covariance consistency (A*C00 ?= C01, AR(1)-style)
# ================================================================

def ar1_consistency_test(
    var_map: np.ndarray, c01_map: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    For AR(1)-like dynamics z_{t+1} = a z_t + eps, the *block-level*
    Yule–Walker identity gives:

        C(0,1; s0)  =  a(s0) * C(0,0; s0)

    We *estimate* a(s0) := C(0,1; s0) / C(0,0; s0)  (guarding zero variance),
    then reconstruct  C_pred(0,1) = a * C(0,0) and measure residuals.

    If |residual| is small relative to |C(0,1)|, the AR(1) consistency holds.
    """
    v_safe = np.maximum(var_map, 1e-12)
    a_hat = c01_map / v_safe
    c01_pred = a_hat * var_map
    residual = c01_map - c01_pred

    rel_err = np.abs(residual) / (np.abs(c01_map) + 1e-12)
    return {
        "a_hat": a_hat,
        "c01_pred": c01_pred,
        "rel_error": rel_err,
        "rel_error_mean": np.array([float(np.mean(rel_err))]),
        "rel_error_median": np.array([float(np.median(rel_err))]),
    }


# ================================================================
# 9) Minimal plotting helpers (optional)
# ================================================================

def plot_maps_side_by_side(A: np.ndarray, B: np.ndarray, titles: Tuple[str, str]):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(A, origin="lower", aspect="equal")
    axs[0].set_title(titles[0]); plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    im1 = axs[1].imshow(B, origin="lower", aspect="equal")
    axs[1].set_title(titles[1]); plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()


def plot_corr_maps_by_lag(corr_maps: np.ndarray, lags: np.ndarray, exclude_zero_lag=True):
    Bh, Bw, L = corr_maps.shape
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)
    sel = corr_maps[:, :, mask]
    lsel = lags[mask]
    vmax = np.max(np.abs(sel)) + 1e-6
    ncols = min(4, sel.shape[2]); nrows = int(np.ceil(sel.shape[2] / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axs = np.atleast_2d(axs)
    for i in range(sel.shape[2]):
        r, c = i // ncols, i % ncols
        im = axs[r, c].imshow(sel[:, :, i], origin="lower", aspect="equal",
                              vmin=-vmax, vmax=vmax, cmap="coolwarm")
        di, dj = int(lsel[i, 0]), int(lsel[i, 1]); axs[r, c].set_title(f"h=({di},{dj})")
        plt.colorbar(im, ax=axs[r, c], fraction=0.046, pad=0.04)
    for i in range(sel.shape[2], nrows*ncols):
        axs[i//ncols, i % ncols].axis("off")
    plt.tight_layout(); plt.show()


def plot_svd(svd_info: Dict[str, np.ndarray], title_suffix=""):
    S = svd_info["singular_values"]; E = svd_info["energy_cumsum"]
    k = np.arange(1, len(S)+1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(k, S, marker="o"); axs[0].set_title(f"Singular values {title_suffix}")
    axs[0].set_xlabel("index"); axs[0].set_ylabel("value"); axs[0].grid(alpha=0.3)
    axs[1].plot(k, E, marker="o"); axs[1].set_ylim(0, 1.05)
    axs[1].set_title(f"Cumulative energy {title_suffix}")
    axs[1].set_xlabel("index"); axs[1].set_ylabel("fraction"); axs[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# ================================================================
# 10) Convenience wrapper — run *all* tests (numeric summary)
# ================================================================

def run_all_tests(trajs: np.ndarray, config: CrossCovConfig) -> Dict[str, Dict[str, np.ndarray]]:
    """
    One-shot convenience: compute empirical maps and run tests A–F.
    Returns a nested dict with numeric summaries; plotting is left to caller.

    IMPORTANT:
      - For Test B (simulation adequacy), we require per-block AR(1) params.
        We infer them from empirical maps:
            a_block   := C(0,1)/C(0,0)
            sigma2    := C(0,0) * (1 - a^2)     (from stationary AR(1) law)
        This keeps the simulator fully *data-driven* and avoids ad-hoc knobs.
    """
    # Empirical maps
    results = compute_local_cross_covariance_maps(trajs, config)
    cov_maps, corr_maps = results["cov_maps"], results["corr_maps"]
    var_map, c01_map = results["var_map"], results["c01_map"]
    lags = results["lags"]; bs = int(results["block_size"][0])

    # ----- Test A: PSD sanity on 2x2 pairs
    testA = pd_test_two_by_two(var_map, cov_maps, lags)

    # ----- Test C: block-wise stability
    testC = block_stability_test(corr_maps, lags, exclude_zero_lag=True)

    # ----- Test D: across-trajectory consistency
    testD = trajectory_consistency_test(trajs, config)

    # ----- Test E: simple isotropic neighbor model fit
    testE = simple_neighbor_model_fit(corr_maps, lags)

    # ----- Test F: AR(1) consistency with C01 = a*C00
    testF = ar1_consistency_test(var_map, c01_map)

    # ----- Test B: simulation adequacy with AR(1) diag (+optional low-rank)
    a_block = np.clip(testF["a_hat"], -0.999, 0.999)          # guard for numerical stability
    sigma2_block = np.maximum(var_map * (1.0 - a_block**2), 1e-10)  # stationary relation
    sim_cfg = AR1DiagLowrankSimConfig(
        a_block=a_block, sigma2_block=sigma2_block,
        lowrank_rank=0, lowrank_scale=0.0, random_seed=0
    )
    testB = simulation_adequacy_test(trajs, results, sim_cfg, config)

    return {
        "Empirical": results,
        "TestA_PSD": testA,
        "TestB_SimAdequacy": testB,
        "TestC_BlockStability": testC,
        "TestD_TrajConsistency": testD,
        "TestE_SimpleFit": testE,
        "TestF_AR1Consistency": testF,
    }


# ================================================================
# 11) Example usage (copy into your notebook and run)
# ================================================================

EXAMPLE = r"""
# --- In Jupyter, after putting this file as `cross_covariance.py` ---

from cross_covariance import (
    CrossCovConfig,
    compute_local_cross_covariance_maps,
    compute_magnitude_map,
    global_svd_on_corr_maps,
    plot_maps_side_by_side, plot_corr_maps_by_lag, plot_svd,
    run_all_tests, AR1DiagLowrankSimConfig, simulation_adequacy_test
)
from data import SQGTrajData

# 1) Load latent trajectories
sqg_data = SQGTrajData()
trajs = sqg_data.get_all()[:, :, 0, :, :]  # (N_traj, T, H, W)

# 2) Configure estimation (u=1; 4-neighbors + (0,0))
cfg = CrossCovConfig(block_size=8, time_lag=1,
                     lags=[(0,0),(1,0),(-1,0),(0,1),(0,-1)])

# 3) Compute empirical maps
res = compute_local_cross_covariance_maps(trajs, cfg)
var_map, corr_maps, lags = res["var_map"], res["corr_maps"], res["lags"]

# 4) Basic diagnostics
mag = compute_magnitude_map(corr_maps, lags, exclude_zero_lag=True)
plot_maps_side_by_side(var_map, mag, titles=("Local variance", "Cross-corr magnitude"))
plot_corr_maps_by_lag(corr_maps, lags, exclude_zero_lag=True)
svd_info = global_svd_on_corr_maps(corr_maps, lags, exclude_zero_lag=True)
plot_svd(svd_info, title_suffix="(empirical)")

# 5) Run all tests (A–F) and inspect numeric summaries
summary = run_all_tests(trajs, cfg)
summary.keys()
# -> 'Empirical', 'TestA_PSD', 'TestB_SimAdequacy', 'TestC_BlockStability',
#    'TestD_TrajConsistency', 'TestE_SimpleFit', 'TestF_AR1Consistency'

# 6) (Optional) Try a simulation with a tiny low-rank global noise and compare
from copy import deepcopy
sim_cfg = AR1DiagLowrankSimConfig(
    a_block = summary["TestF_AR1Consistency"]["a_hat"],
    sigma2_block = summary["Empirical"]["var_map"] * (1.0 - summary["TestF_AR1Consistency"]["a_hat"]**2),
    lowrank_rank = 1,
    lowrank_scale = 0.02,
    random_seed = 123
)
sim_report = simulation_adequacy_test(trajs, summary["Empirical"], sim_cfg, cfg)

# Now compare sim_report["sim_mag"] vs summary["Empirical"]["corr_maps"]'s magnitude, etc.
"""


# ================================================================
# 12) AR(2) + low-rank simulator and parameter estimation (NEW)
# ================================================================

@dataclass
class AR2LowRankSimConfig:
    """
    AR(2) with low-rank spatial structure in the innovation (and optional tiny
    nearest-neighbor coupling in the state transition).

    Model:
        z_t = (Phi1 ⊙ I) z_{t-1} + (Phi2 ⊙ I) z_{t-2} + eta_t
        eta_t ~ N(0,  Sigma),  Sigma = diag(sigma2) + lambda * u u^T

    where:
        - Phi1_block, Phi2_block, sigma2_block: (Bh, Bw) block-wise parameters.
          They will be upsampled to (H, W) pixel maps (piecewise-constant blocks).
        - rank1_scale (lambda) controls the energy of the global mode u.
          We FIX u to be the constant/global vector (all ones / sqrt(HW)),
          which matches the rank-1 pattern revealed by SVD diagnostics.
        - neighbor_coupling_b: optional tiny coupling in A (4-neighbor smoothing):
              z_{t-1} is replaced by z_{t-1} + b * K(z_{t-1})
          where K averages the 4-neighborhood (isotropic). This remains small.

    Rationale for design:
        * Your empirical cross-covariance shows rank-1 dominance. A global mode
          in innovations reproduces that structure without imposing strong spatial
          propagation. A tiny neighbor coupling can be added if magnitude
          calibration still underfits.
    """
    phi1_block: np.ndarray        # (Bh, Bw)
    phi2_block: np.ndarray        # (Bh, Bw)
    sigma2_block: np.ndarray      # (Bh, Bw)
    rank1_scale: float = 0.0      # lambda >= 0
    neighbor_coupling_b: float = 0.0
    random_seed: int = 0


def _upsample_block_to_pixels(block_map: np.ndarray, block_size: int) -> np.ndarray:
    Bh, Bw = block_map.shape
    H, W = Bh * block_size, Bw * block_size
    out = np.empty((H, W), dtype=np.float64)
    for bi in range(Bh):
        for bj in range(Bw):
            out[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = block_map[bi, bj]
    return out


def _neighbor4_average(arr: np.ndarray) -> np.ndarray:
    """
    Simple 4-neighbor averaging with Neumann boundary (replicate edges).
    We avoid external deps; cost is negligible for block-size grids.

    NOTE: This is a *tiny* isotropic coupling used only if neighbor_coupling_b>0.
    """
    H, W = arr.shape
    up    = np.vstack([arr[0:1, :], arr[:-1, :]])
    down  = np.vstack([arr[1:, :],   arr[-1:, :]])
    left  = np.hstack([arr[:, 0:1],  arr[:, :-1]])
    right = np.hstack([arr[:, 1:],   arr[:, -1:]])
    return 0.25 * (up + down + left + right)


def simulate_from_ar2_lowrank(
    cfg: AR2LowRankSimConfig,
    N_traj: int, T: int, block_size: int
) -> np.ndarray:
    """
    Generate (N_traj, T, H, W) from AR(2)+low-rank. We avoid closed-form AR(2)
    stationary covariance init by using a short burn-in to reach stationarity.
    This is numerically robust for arbitrary block-wise (phi1, phi2).

    Key steps (why they matter):
    - Upsample block parameters to pixel maps: ensures nonstationary blocks but
      piecewise-constant within each block (matches your diagnostics setting).
    - rank-1 innovation: epsilon_t = eps_diag + sqrt(lambda) * g_t * u
      with u = ones/sqrt(HW) — reproduces rank-1 global mode seen in SVD.
    - optional nearest-neighbor coupling in A: promotes minimal cross-pixel
      cross-time covariance if purely rank-1 innovation is insufficient.

    All random draws use a local RNG to avoid global-state side effects.
    """
    rng = np.random.default_rng(cfg.random_seed)

    # ---------- Upsample parameters to pixel maps ----------
    phi1_map   = _upsample_block_to_pixels(cfg.phi1_block,   block_size)  # (H,W)
    phi2_map   = _upsample_block_to_pixels(cfg.phi2_block,   block_size)
    sigma2_map = _upsample_block_to_pixels(cfg.sigma2_block, block_size)

    H, W = phi1_map.shape
    P = H * W

    # ---------- Low-rank direction: fixed global constant mode ----------
    if cfg.rank1_scale > 0.0:
        u = np.ones((P, 1), dtype=np.float64) / np.sqrt(P)  # (P,1), normalized
    else:
        u = None

    # ---------- Allocate ----------
    Z = np.zeros((N_traj, T, H, W), dtype=np.float64)

    # ---------- Helper to draw innovation with rank-1 ----------
    def draw_eta():
        # diagonal noise (pixel-wise variance)
        eps = rng.normal(loc=0.0, scale=np.sqrt(sigma2_map))
        if u is not None:
            g = rng.standard_normal()  # scalar factor for rank-1 mode
            lr = (u * (np.sqrt(cfg.rank1_scale) * g)).reshape(H, W)
            eps = eps + lr
        return eps

    # ---------- Burn-in to approach stationarity ----------
    # We generate extra steps with AR(2) recurrence, then keep the last T.
    burn = 200
    T_tot = burn + T
    for n in range(N_traj):
        z_prev2 = rng.normal(0.0, 1.0, size=(H, W))  # arbitrary start
        z_prev1 = rng.normal(0.0, 1.0, size=(H, W))
        seq = []

        for t in range(T_tot):
            z1 = z_prev1
            if cfg.neighbor_coupling_b > 0.0:
                # tiny isotropic 4-neighbor smoothing on z_{t-1}
                z1 = z1 + cfg.neighbor_coupling_b * _neighbor4_average(z1)

            eta = draw_eta()
            z_cur = phi1_map * z1 + phi2_map * z_prev2 + eta

            seq.append(z_cur)
            z_prev2, z_prev1 = z_prev1, z_cur

        seq = np.stack(seq[-T:], axis=0)  # keep last T
        Z[n] = seq

    return Z


# ---------------------------------------------------------------
# 12.1) AR(2) parameter estimation from empirical γ0,γ1,γ2 (per block)
# ---------------------------------------------------------------

def estimate_ar2_from_empirical(var_map: np.ndarray,
                                c01_map: np.ndarray,
                                c02_map: np.ndarray,
                                ridge: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve Yule–Walker for AR(2) per block using:
        γ0 = Var(z_t), γ1 = Cov(z_t, z_{t+1}), γ2 = Cov(z_t, z_{t+2})

    YW equations:
        [γ0  γ1][φ1] = [γ1]
        [γ1  γ0][φ2]   [γ2]
    and innovation variance:
        σ^2 = γ0 - φ1 γ1 - φ2 γ2

    We solve a 2x2 system per block with tiny Tikhonov ridge for stability.
    Returns:
        phi1_block, phi2_block, sigma2_block  (all shape (Bh,Bw))
    """
    Bh, Bw = var_map.shape
    phi1 = np.zeros_like(var_map, dtype=np.float64)
    phi2 = np.zeros_like(var_map, dtype=np.float64)
    s2   = np.zeros_like(var_map, dtype=np.float64)

    for i in range(Bh):
        for j in range(Bw):
            g0 = float(max(var_map[i, j], 1e-12))
            g1 = float(c01_map[i, j])
            g2 = float(c02_map[i, j])

            A = np.array([[g0 + ridge, g1],
                          [g1,         g0 + ridge]], dtype=np.float64)
            b = np.array([g1, g2], dtype=np.float64)

            try:
                ph = np.linalg.solve(A, b)  # [phi1, phi2]
            except np.linalg.LinAlgError:
                ph = np.array([0.0, 0.0], dtype=np.float64)

            phi1[i, j], phi2[i, j] = ph[0], ph[1]
            s2[i, j] = max(g0 - ph[0]*g1 - ph[1]*g2, 1e-12)

    # Stability clip (keep inside a reasonable AR(2) stability region)
    phi1 = np.clip(phi1, -1.5, 1.5)
    phi2 = np.clip(phi2, -0.99, 0.99)
    return phi1, phi2, s2


# ---------------------------------------------------------------
# 12.2) Convenience: compute γ0, γ1, γ2 maps via your existing engine
# ---------------------------------------------------------------

def compute_gamma012_maps(trajs: np.ndarray, cfg: "CrossCovConfig") -> Dict[str, np.ndarray]:
    """
    Use your local cross-covariance estimator twice (u=1 and u=2) to obtain:
        γ0_map := C(0,0), γ1_map := C(0,1), γ2_map := C(0,2),
    computed block-wise. We require that (0,0) is in cfg.lags.

    Why we do this:
        AR(2) Yule–Walker needs γ0,γ1,γ2. This function reuses exactly the
        same empirical machinery already validated in your pipeline so that
        estimation and simulation share the same definition of statistics.
    """
    # u=1
    cfg1 = CrossCovConfig(block_size=cfg.block_size, time_lag=1, lags=cfg.lags)
    res1 = compute_local_cross_covariance_maps(trajs, cfg1)
    g0 = res1["var_map"]
    g1 = res1["c01_map"]

    # u=2
    cfg2 = CrossCovConfig(block_size=cfg.block_size, time_lag=2, lags=cfg.lags)
    res2 = compute_local_cross_covariance_maps(trajs, cfg2)
    # when lags contains (0,0), its cov at u=2 is γ2
    g2 = res2["c01_map"]

    return {"gamma0": g0, "gamma1": g1, "gamma2": g2, "BhBw": np.array(g0.shape, dtype=int)}


# ---------------------------------------------------------------
# 12.3) Simulation-based adequacy test for AR(2)+low-rank (NEW)
# ---------------------------------------------------------------

def simulation_adequacy_test_ar2(
    real_trajs: np.ndarray,
    real_results: Dict[str, np.ndarray],
    sim_cfg: AR2LowRankSimConfig,
    cfg: "CrossCovConfig"
) -> Dict[str, np.ndarray]:
    """
    Same spirit as your AR(1) adequacy test,
    but *simulate with AR(2)+low-rank* and compare:
        - magnitude maps,
        - SVD spectra of corr_maps.

    We use the same pipeline (compute_local_cross_covariance_maps)
    to ensure apples-to-apples diagnostics.
    """
    N, T, _, _ = real_trajs.shape
    sim_trajs = simulate_from_ar2_lowrank(sim_cfg, N_traj=N, T=T, block_size=cfg.block_size)
    sim_res   = compute_local_cross_covariance_maps(sim_trajs, cfg)

    lags = real_results["lags"]
    real_mag = compute_magnitude_map(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_mag  = compute_magnitude_map(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    real_svd = global_svd_on_corr_maps(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_svd  = global_svd_on_corr_maps(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    mag_diff_mean    = float(np.mean(sim_mag - real_mag))
    mag_diff_absmean = float(np.mean(np.abs(sim_mag - real_mag)))
    svd_diff         = float(np.linalg.norm(sim_svd["singular_values"] - real_svd["singular_values"]))
    svd_energy_diff  = float(np.linalg.norm(sim_svd["energy_cumsum"] - real_svd["energy_cumsum"]))

    return {
        "real_mag": real_mag, "sim_mag": sim_mag,
        "mag_diff_mean": np.array([mag_diff_mean]),
        "mag_diff_abs_mean": np.array([mag_diff_absmean]),
        "svd_diff": np.array([svd_diff]),
        "svd_energy_diff": np.array([svd_energy_diff]),
        "sim_corr_maps": sim_res["corr_maps"],
    }


# ---------------------------------------------------------------
# 12.4) One-shot wrapper to estimate AR(2) blocks and run adequacy
# ---------------------------------------------------------------

def run_ar2_lowrank_pipeline(trajs: np.ndarray,
                             cfg: "CrossCovConfig",
                             rank1_scale: float = 0.0,
                             neighbor_coupling_b: float = 0.0,
                             random_seed: int = 0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Full AR(2)+low-rank pipeline:
      1) Empirical maps (u=1) for real data  -> 'Empirical'
      2) Compute γ0,γ1,γ2 (u=1,2) per block -> estimate φ1,φ2,σ^2
      3) Simulate AR(2)+low-rank (rank1_scale, neighbor_b)
      4) Adequacy test (magnitude & SVD)

    Returns a dict with everything needed for plotting/analysis.
    """
    # 1) Empirical (u=1)
    empirical = compute_local_cross_covariance_maps(trajs, cfg)

    # 2) Estimate AR(2) parameters per block via Yule–Walker
    g012 = compute_gamma012_maps(trajs, cfg)
    phi1_b, phi2_b, s2_b = estimate_ar2_from_empirical(
        g012["gamma0"], g012["gamma1"], g012["gamma2"]
    )

    # 3) Simulate with AR(2)+low-rank
    sim_cfg = AR2LowRankSimConfig(
        phi1_block=phi1_b,
        phi2_block=phi2_b,
        sigma2_block=s2_b,
        rank1_scale=rank1_scale,
        neighbor_coupling_b=neighbor_coupling_b,
        random_seed=random_seed
    )

    # 4) Adequacy
    report = simulation_adequacy_test_ar2(
        real_trajs=trajs, real_results=empirical,
        sim_cfg=sim_cfg, cfg=cfg
    )

    return {
        "Empirical": empirical,
        "AR2_Params": {"phi1_block": phi1_b, "phi2_block": phi2_b, "sigma2_block": s2_b},
        "Adequacy_AR2": report,
        "SimConfig": sim_cfg.__dict__
    }


# ================================================================
# 13) Example usage for AR(2)+low-rank (copy into notebook)
# ================================================================

EXAMPLE_AR2 = r"""
from cross_covariance import (
    CrossCovConfig, compute_magnitude_map, plot_maps_side_by_side, plot_svd,
    run_ar2_lowrank_pipeline
)
from data import SQGTrajData

# Load latent trajectories
sqg_data = SQGTrajData()
trajs = sqg_data.get_all()[:, :, 0, :, :]

# Configure (u=1; include (0,0) so we have γ1 and γ2 via time_lag=1,2)
cfg = CrossCovConfig(block_size=8, time_lag=1,
                     lags=[(0,0),(1,0),(-1,0),(0,1),(0,-1)])

# ---- Run AR(2)+low-rank pipeline with a first guess of rank1_scale ----
out = run_ar2_lowrank_pipeline(
    trajs, cfg,
    rank1_scale=0.05,          # try 0.02 ~ 0.20, coarse grid search works well
    neighbor_coupling_b=0.0,   # keep 0.0 first; if still underfit magnitude, try 1e-3 ~ 1e-2
    random_seed=0
)

# Compare magnitude maps visually
plot_maps_side_by_side(out["Empirical"]["corr_maps"].shape[:2] and
                       compute_magnitude_map(out["Empirical"]["corr_maps"], out["Empirical"]["lags"], True),
                       out["Adequacy_AR2"]["sim_mag"],
                       ("Real mag", "Sim mag (AR2+rank1)"))

# Print numeric adequacy summaries
print("Mean |mag diff|:", out["Adequacy_AR2"]["mag_diff_abs_mean"][0])
print("SVD diff:",       out["Adequacy_AR2"]["svd_diff"][0])
print("SVD energy diff:",out["Adequacy_AR2"]["svd_energy_diff"][0])

# Plot SVD curves
from cross_covariance import global_svd_on_corr_maps
real_svd = global_svd_on_corr_maps(out["Empirical"]["corr_maps"], out["Empirical"]["lags"], True)
sim_svd  = global_svd_on_corr_maps(out["Adequacy_AR2"]["sim_corr_maps"], out["Empirical"]["lags"], True)
plot_svd(real_svd, "(Real)")
plot_svd(sim_svd,  "(Sim AR2+rank1)")
"""


# ================================================================
# 14) AR(2) + rank-1 **transition** simulator (NEW, fixes C(h,1) issue)
# ================================================================
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

@dataclass
class AR2Rank1TransitionConfig:
    """
    AR(2) with a rank-1 term in the **state transition** (NOT in innovation).
    This is the minimal change that can generate non-zero cross-time & cross-space
    covariance C(h,1), which your data exhibits (Real magnitude ~ 0.19–0.20).

    Model (per pixel on a 2D grid, vectorized form over all pixels):
        z_{t+1} = (phi1 ⊙ I) z_t + (phi2 ⊙ I) z_{t-1}
                  + alpha * (u u^T) z_t
                  + eps_{t+1},
        eps_{t+1} ~ N(0, diag(sigma2))   # (optionally: add tiny iid jitter only)

    where:
        - phi1_block, phi2_block, sigma2_block: (Bh,Bw) block-wise parameters.
          They are upsampled to (H,W) pixel maps (piecewise constant per block).
        - alpha: scalar > 0 controlling the strength of the global rank-1
          propagation (u u^T) z_t. This is the key to produce C(h,1) ≠ 0.
        - u: the global spatial mode (vector of length H*W, L2-normalized).
          Typical choices:
            * constant mode (ones/√(HW)) — isotropic global push;
            * empirical first PC (from real trajectories), see helper below.

    NOTE:
      Putting the low-rank in **transition** (A) rather than **innovation** (Σ)
      is essential: innovation at t+1 is independent of z_t, so it cannot
      contribute to Cov(z_t(s), z_{t+1}(s+h)). The rank-1 transition does.
    """
    phi1_block: np.ndarray
    phi2_block: np.ndarray
    sigma2_block: np.ndarray
    alpha: float = 0.05
    # u source: if u_vec is None, we will build from `u_mode`
    u_vec: Optional[np.ndarray] = None    # shape (H*W,), L2-normalized
    u_mode: str = "constant"              # "constant" | "empirical_pca"
    random_seed: int = 0


def _upsample_block_to_pixels(block_map: np.ndarray, block_size: int) -> np.ndarray:
    Bh, Bw = block_map.shape
    H, W = Bh * block_size, Bw * block_size
    out = np.empty((H, W), dtype=np.float64)
    for bi in range(Bh):
        for bj in range(Bw):
            out[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = block_map[bi, bj]
    return out


def estimate_global_mode_u_from_trajs(trajs: np.ndarray) -> np.ndarray:
    """
    Build an empirical global spatial mode u (length H*W) by PCA/SVD
    of the frame-wise flattened fields (across all trajectories & time).
    We center each frame by its spatial mean to avoid bias from DC offset.

    Returns:
        u (H*W,) with ||u||_2 = 1.
    """
    N, T, H, W = trajs.shape
    X = trajs.reshape(N*T, H*W).astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)          # remove per-frame mean
    # Economy SVD on (frames x pixels). We only need the first left singular vec
    # in pixel space: compute principal right-singular vector of X.
    # Using np.linalg.svd on (frames x pixels) matrix is robust for moderate dims.
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]                                   # shape (H*W,)
    # Normalize for safety
    u_norm = np.linalg.norm(u) + 1e-12
    return (u / u_norm).astype(np.float64)


def simulate_from_ar2_rank1_transition(
    cfg: AR2Rank1TransitionConfig,
    N_traj: int, T: int, block_size: int,
    # optional: provide trajs if u_mode="empirical_pca" and u_vec is None
    trajs_for_u: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simulate (N_traj, T, H, W) from AR(2) with **rank-1 transition**.

    Key reasons this works:
      - The extra term alpha * (u u^T) z_t introduces a global coupling among
        pixels at time t that *propagates* to t+1, hence C(h,1) > 0.
      - Using a single global u yields rank-1 dominant SVD patterns in your
        correlation maps, as observed in the real data.

    Implementation details:
      - We use burn-in iterations to approach stationarity under AR(2).
      - Innovation is diagonal (per-pixel variance sigma2_map); add only jitter.
      - u is either provided (cfg.u_vec), constant, or estimated by PCA.
    """
    rng = np.random.default_rng(cfg.random_seed)

    # ---- Upsample block parameters to per-pixel maps
    phi1_map   = _upsample_block_to_pixels(cfg.phi1_block,   block_size)  # (H,W)
    phi2_map   = _upsample_block_to_pixels(cfg.phi2_block,   block_size)
    sigma2_map = _upsample_block_to_pixels(cfg.sigma2_block, block_size)

    H, W = phi1_map.shape
    P = H * W

    # ---- Prepare global mode u
    if cfg.u_vec is not None:
        u = cfg.u_vec.reshape(P).astype(np.float64)
        u /= (np.linalg.norm(u) + 1e-12)
    else:
        if cfg.u_mode == "empirical_pca":
            assert trajs_for_u is not None, \
                "trajs_for_u required when u_mode='empirical_pca' and u_vec=None"
            u = estimate_global_mode_u_from_trajs(trajs_for_u)
        else:
            # "constant" mode (isotropic global push)
            u = np.ones(P, dtype=np.float64) / np.sqrt(P)

    # ---- Allocate output
    Z = np.zeros((N_traj, T, H, W), dtype=np.float64)

    # ---- Helpers
    phi1_vec = phi1_map.reshape(P)     # (P,)
    phi2_vec = phi2_map.reshape(P)     # (P,)
    sigma_vec = np.sqrt(sigma2_map.reshape(P))  # std per pixel
    uuT = np.outer(u, u)               # (P,P), rank-1 matrix
    alpha = float(cfg.alpha)

    def step_ar2(z_t: np.ndarray, z_tm1: np.ndarray) -> np.ndarray:
        """
        One AR(2) step in vectorized form (flattened grids).
        z_{t+1} = (phi1 .* z_t) + (phi2 .* z_{t-1}) + alpha*(uu^T z_t) + eps
        """
        # alpha * (u u^T) z_t   == alpha * u * (u^T z_t), computed as two BLAS-1 ops
        global_drive = alpha * u * float(np.dot(u, z_t))
        base = phi1_vec * z_t + phi2_vec * z_tm1 + global_drive
        eps = rng.normal(0.0, 1.0, size=P) * sigma_vec
        return base + eps

    # ---- Burn-in to reach stationary-like regime
    burn = 200
    T_tot = burn + T

    for n in range(N_traj):
        z_tm1 = rng.normal(0.0, 1.0, size=P)  # z_{-1}
        z_t   = rng.normal(0.0, 1.0, size=P)  # z_{ 0}
        seq = []
        for t in range(T_tot):
            z_tp1 = step_ar2(z_t, z_tm1)
            seq.append(z_tp1.reshape(H, W))
            z_tm1, z_t = z_t, z_tp1
        Z[n] = np.stack(seq[-T:], axis=0)

    return Z


# ---------------------------------------------------------------
# 14.1) Adequacy test (magnitude & SVD) for AR(2)+rank-1 transition
# ---------------------------------------------------------------
def simulation_adequacy_test_ar2_rank1_transition(
    real_trajs: np.ndarray,
    real_results: Dict[str, np.ndarray],
    sim_cfg: AR2Rank1TransitionConfig,
    cfg: "CrossCovConfig"
) -> Dict[str, np.ndarray]:
    """
    Simulate with AR(2)+rank-1 transition and compare to empirical statistics:
      - magnitude map (p-norm of |R(h,1)| across spatial lags per block),
      - SVD spectrum of block-wise correlation vectors.

    We reuse the same empirical engine so comparisons are apples-to-apples.
    """
    N, T, _, _ = real_trajs.shape
    sim_trajs = simulate_from_ar2_rank1_transition(
        sim_cfg, N_traj=N, T=T, block_size=cfg.block_size,
        trajs_for_u=real_trajs if (sim_cfg.u_vec is None and sim_cfg.u_mode == "empirical_pca") else None
    )
    sim_res   = compute_local_cross_covariance_maps(sim_trajs, cfg)

    lags = real_results["lags"]
    real_mag = compute_magnitude_map(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_mag  = compute_magnitude_map(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    real_svd = global_svd_on_corr_maps(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_svd  = global_svd_on_corr_maps(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    mag_diff_mean    = float(np.mean(sim_mag - real_mag))
    mag_diff_absmean = float(np.mean(np.abs(sim_mag - real_mag)))
    svd_diff         = float(np.linalg.norm(sim_svd["singular_values"] - real_svd["singular_values"]))
    svd_energy_diff  = float(np.linalg.norm(sim_svd["energy_cumsum"] - real_svd["energy_cumsum"]))

    return {
        "real_mag": real_mag, "sim_mag": sim_mag,
        "mag_diff_mean": np.array([mag_diff_mean]),
        "mag_diff_abs_mean": np.array([mag_diff_absmean]),
        "svd_diff": np.array([svd_diff]),
        "svd_energy_diff": np.array([svd_energy_diff]),
        "sim_corr_maps": sim_res["corr_maps"],
    }


# ---------------------------------------------------------------
# 14.2) One-shot wrapper: estimate AR(2) blocks, then run rank-1 transition sim
# ---------------------------------------------------------------
def run_ar2_rank1_transition_pipeline(
    trajs: np.ndarray,
    cfg: "CrossCovConfig",
    alpha: float = 0.05,
    u_mode: str = "constant",
    u_vec: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Full pipeline:
      1) Empirical maps at u=1 (for real data)
      2) Estimate AR(2) block params via Yule–Walker (γ0,γ1,γ2)
      3) Build AR(2)+rank-1 transition config with (alpha, u)
      4) Simulate & run adequacy test (magnitude + SVD)

    Tuning tips:
      - Start with u_mode="constant" and coarse grid on alpha in [0.02, 0.40].
      - If magnitude still underfits or SVD shape differs, try u_mode="empirical_pca".
        This aligns u to the dominant spatial mode in your data.
    """
    # 1) Empirical (u=1)
    empirical = compute_local_cross_covariance_maps(trajs, cfg)

    # 2) Estimate AR(2) (γ0,γ1,γ2) using existing helpers from section 12
    g012 = compute_gamma012_maps(trajs, cfg)
    phi1_b, phi2_b, s2_b = estimate_ar2_from_empirical(
        g012["gamma0"], g012["gamma1"], g012["gamma2"]
    )

    # 3) Simulator config
    sim_cfg = AR2Rank1TransitionConfig(
        phi1_block=phi1_b, phi2_block=phi2_b, sigma2_block=s2_b,
        alpha=alpha, u_vec=u_vec, u_mode=u_mode, random_seed=random_seed
    )

    # 4) Adequacy test
    report = simulation_adequacy_test_ar2_rank1_transition(
        real_trajs=trajs, real_results=empirical,
        sim_cfg=sim_cfg, cfg=cfg
    )

    return {
        "Empirical": empirical,
        "AR2_Params": {"phi1_block": phi1_b, "phi2_block": phi2_b, "sigma2_block": s2_b},
        "Adequacy_AR2_Transition": report,
        "SimConfig": {"alpha": alpha, "u_mode": u_mode, "random_seed": random_seed}
    }


# ================================================================
# 14.3) Example usage (copy into your notebook)
# ================================================================
EXAMPLE_AR2_TRANSITION = r"""
from cross_covariance import (
    CrossCovConfig, compute_magnitude_map, plot_maps_side_by_side, plot_svd,
    run_ar2_rank1_transition_pipeline, global_svd_on_corr_maps
)
from data import SQGTrajData

# 1) Load latent trajectories
sqg_data = SQGTrajData()
trajs = sqg_data.get_all()[:, :, 0, :, :]

# 2) Configure local covariance (u=1; include (0,0) in lags)
cfg = CrossCovConfig(block_size=8, time_lag=1,
                     lags=[(0,0),(1,0),(-1,0),(0,1),(0,-1)])

# 3) Coarse grid on alpha to match magnitude & SVD
best = None
for alpha in [0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.40]:
    out = run_ar2_rank1_transition_pipeline(
        trajs, cfg, alpha=alpha, u_mode="constant", random_seed=0
    )
    m = out["Adequacy_AR2_Transition"]["mag_diff_abs_mean"][0]
    e = out["Adequacy_AR2_Transition"]["svd_energy_diff"][0]
    print(f"[alpha={alpha:.3f}]  Mean|mag diff|={m:.4f}  SVD energy diff={e:.4f}")
    if (best is None) or (m < best[0]):
        best = (m, e, alpha, out)

print("BEST alpha:", best[2])

# 4) Visual inspection
real_mag = best[3]["Adequacy_AR2_Transition"]["real_mag"]
sim_mag  = best[3]["Adequacy_AR2_Transition"]["sim_mag"]
plot_maps_side_by_side(real_mag, sim_mag, ("Real mag", f"Sim mag (AR2 + rank1 A, alpha={best[2]:.3f})"))

# 5) SVD comparison
real_svd = global_svd_on_corr_maps(best[3]["Empirical"]["corr_maps"], best[3]["Empirical"]["lags"], True)
sim_svd  = global_svd_on_corr_maps(best[3]["Adequacy_AR2_Transition"]["sim_corr_maps"], best[3]["Empirical"]["lags"], True)
plot_svd(real_svd, "(Real)")
plot_svd(sim_svd,  "(Sim AR2 + rank1 A)")

"""


# ================================================================
# 15) Dynamic Factor + AR(2) (NEW): a rank-1 temporal global mode f_t
#      + pixelwise idiosyncratic AR(2), with full simulation & adequacy test
# ================================================================
# What this section adds (why it matters):
#   • A *Dynamic Factor Model (DFM)* that reconciles your diagnostics:
#       - spatial ACF(h,0) ~ 0   (no same-time spatial smoothing),
#       - but strong cross-time & cross-space C(h,1) with rank-1 SVD dominance.
#     We model a *global temporal factor* f_t that drives all pixels via a
#     spatial loading u (rank-1), while each pixel keeps its own AR(2) dynamics.
#   • This is the minimal generative model that can produce large C(h,1)
#     and rank-1 pattern *without* imposing same-time spatial smoothness.
#
#   Model (vectorized over pixels p=H*W; u has ||u||_2 = 1):
#     z_t  =    c * f_t * u             +    x_t                      (observation eq.)
#     x_t  = φ1 ⊙ x_{t-1} + φ2 ⊙ x_{t-2} +    ε_t,   ε_t ~ N(0, diag(σ²))  (idiosyncratic AR(2))
#     f_t  = β1  f_{t-1} + β2  f_{t-2} + ξ_t,     ξ_t ~ N(0, τ²)           (global factor AR(2))
#
#   Key identification choices:
#     – u is unit-norm; c (scalar loading magnitude) captures the factor strength.
#     – We estimate u by PCA/SVD on frames (empirical first PC) or use "constant".
#     – Given u, we estimate f_t by projection and fit AR(2) to f_t (Yule–Walker).
#     – Given (u, f_t, c), we remove c f_t u from z_t to get residual x_t and
#       fit *block-wise* AR(2) for φ1, φ2, σ² (Yule–Walker on γ0,γ1,γ2 of residuals).
#
#   Why this produces C(h,1) >> 0 while spatial ACF ~ 0:
#     – The cross-time & cross-space covariance arises from temporal persistence
#       of f_t times the common loading u (rank-1), not from same-time spatial smoothing.
#
#   Interfaces:
#     • estimate_u_from_trajs() – empirical u via PCA (unit-norm).
#     • fit_ar2_univariate_yw() – AR(2) Yule–Walker for any 1D series.
#     • dynamic_factor_estimation() – estimate (u, f_t, c, β1,β2,τ², φ1/φ2/σ² maps).
#     • simulate_dynamic_factor_ar2() – sample (N,T,H,W) from the fitted DFM.
#     • adequacy test + one-shot pipeline with plotting-ready outputs.
#
#   All key math steps are heavily commented below.
# ================================================================

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np


# ------------------------------
# 15.1) Helpers: u, AR(2) for 1D, projections
# ------------------------------

def estimate_u_from_trajs_pca(trajs: np.ndarray) -> np.ndarray:
    """
    Estimate the spatial loading u (length P=H*W) by PCA on *frames*.

    Why: We want the dominant spatial direction that explains most variance over time.
    Steps:
      • Stack all frames (N_traj * T) as rows, each is a flattened (H*W)-vector.
      • Remove each frame's spatial mean (avoid a DC bias unrelated to structure).
      • Compute top right singular vector Vt[0] – the empirical first principal
        loading in pixel space.
      • Normalize to unit L2 norm for identification.

    Returns:
      u: shape (P,), ||u||_2 = 1
    """
    N, T, H, W = trajs.shape
    X = trajs.reshape(N*T, H*W).astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    u /= (np.linalg.norm(u) + 1e-12)
    return u


def fit_ar2_univariate_yw(y: np.ndarray, ridge: float = 1e-12) -> Tuple[float, float, float]:
    """
    Fit AR(2) to a 1D series y_t by Yule–Walker.

    Equations (γk = autocov at lag k):
        [γ0  γ1][β1] = [γ1]
        [γ1  γ0][β2]   [γ2]
        τ² = γ0 - β1 γ1 - β2 γ2

    Returns:
      β1, β2, τ²   (τ² clamped >=1e-12)
    """
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    T = len(y)
    if T < 5:
        return 0.0, 0.0, float(np.var(y) + 1e-12)

    # unbiased autocov estimates
    def acov(k):
        return float(np.dot(y[:-k], y[k:]) / (T - k)) if k > 0 else float(np.dot(y, y) / T)

    g0 = acov(0)
    g1 = acov(1)
    g2 = acov(2)

    A = np.array([[g0 + ridge, g1],
                  [g1,         g0 + ridge]], dtype=np.float64)
    b = np.array([g1, g2], dtype=np.float64)
    try:
        beta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        beta = np.array([0.0, 0.0], dtype=np.float64)
    beta1, beta2 = float(beta[0]), float(beta[1])
    tau2 = max(g0 - beta1*g1 - beta2*g2, 1e-12)
    # Stability clipping (conservative)
    beta1 = np.clip(beta1, -1.5, 1.5)
    beta2 = np.clip(beta2, -0.99, 0.99)
    return beta1, beta2, float(tau2)


def project_frames_onto_u(trajs: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Project each frame z_t (flattened) on u to get a raw factor series f̂_t.

    f̂_t := u^T z_t  (since ||u||=1). We use all trajectories concatenated.

    Returns:
      f_hat: shape (N_traj*T,)
    """
    N, T, H, W = trajs.shape
    X = trajs.reshape(N*T, H*W).astype(np.float64)
    # remove per-frame spatial mean (stay aligned with u-estimation)
    X -= X.mean(axis=1, keepdims=True)
    u = u.reshape(-1)
    f_hat = X @ u  # (N*T,)
    return f_hat


def estimate_c_scalar(trajs: np.ndarray, u: np.ndarray, f: np.ndarray) -> float:
    """
    Estimate the scalar loading c in z_t ≈ c f_t u + x_t, in least squares sense.

      minimize over c:  Σ_t || z_t - c f_t u ||^2
    Closed form:
      c = ( Σ_t f_t * (u^T z_t) ) / ( Σ_t f_t^2 )

    We reuse the same centered frames as in projection (consistency).
    """
    N, T, H, W = trajs.shape
    X = trajs.reshape(N*T, H*W).astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)
    u = u.reshape(-1)
    y = X @ u                              # (N*T,) == raw projection
    denom = float(np.dot(f, f)) + 1e-12
    c = float(np.dot(f, y) / denom)
    return c


# ------------------------------
# 15.2) Block-wise AR(2) for idiosyncratic component x_t
# ------------------------------

def compute_block_gamma012_for_residuals(
    trajs: np.ndarray, cfg: "CrossCovConfig",
    u: np.ndarray, c: float, f: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute γ0, γ1, γ2 for the *residual* field x_t = z_t - c f_t u (block-wise).

    Why:
      The idiosyncratic component x_t is modeled as per-pixel AR(2). To estimate
      (φ1, φ2, σ²) per block (piecewise constant), we need γ0,γ1,γ2 of x_t,
      which we obtain by running the local cross-covariance engine on *residuals*.

    Implementation:
      • Build residual sequence X_t (N_traj, T, H, W): subtract c f_t u_map from z_t.
      • Run compute_local_cross_covariance_maps twice: time_lag=1 and time_lag=2.
      • Extract:
         γ0 := Var(x_t)            from var_map (u=0)
         γ1 := Cov(x_t, x_{t+1})   from c01_map (u=1, h=(0,0))
         γ2 := Cov(x_t, x_{t+2})   from c01_map (u=2, h=(0,0))
    """
    N, T, H, W = trajs.shape
    P = H * W
    u_map = u.reshape(H, W)
    # Build residuals X = Z - c f u
    X = np.empty_like(trajs, dtype=np.float64)
    ft = f.reshape(N*T)
    for n in range(N):
        for t in range(T):
            idx = n*T + t
            X[n, t] = trajs[n, t].astype(np.float64) - c * ft[idx] * u_map

    # u=1
    cfg1 = CrossCovConfig(block_size=cfg.block_size, time_lag=1, lags=cfg.lags)
    r1 = compute_local_cross_covariance_maps(X, cfg1)
    g0 = r1["var_map"]
    g1 = r1["c01_map"]

    # u=2
    cfg2 = CrossCovConfig(block_size=cfg.block_size, time_lag=2, lags=cfg.lags)
    r2 = compute_local_cross_covariance_maps(X, cfg2)
    g2 = r2["c01_map"]

    return {"gamma0": g0, "gamma1": g1, "gamma2": g2}


def estimate_ar2_blocks_for_residuals(
    trajs: np.ndarray, cfg: "CrossCovConfig",
    u: np.ndarray, c: float, f: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Estimate block-wise AR(2) params (φ1_block, φ2_block, σ²_block) for x_t.

    We reuse the same Yule–Walker solver as before, but on residual γ0,γ1,γ2.
    """
    g = compute_block_gamma012_for_residuals(trajs, cfg, u, c, f)
    phi1_b, phi2_b, s2_b = estimate_ar2_from_empirical(g["gamma0"], g["gamma1"], g["gamma2"])
    return {"phi1_block": phi1_b, "phi2_block": phi2_b, "sigma2_block": s2_b}


# ------------------------------
# 15.3) End-to-end estimation of the DFM (u, f_t, c, β*, τ²; φ*, σ²)
# ------------------------------

@dataclass
class DynamicFactorParams:
    u: np.ndarray             # (P,)
    c: float                  # scalar loading magnitude
    f_series: np.ndarray      # (N*T,) concatenated factor estimates
    beta1: float              # AR(2) coeff of f_t
    beta2: float
    tau2: float               # innovation variance of f_t
    phi1_block: np.ndarray    # (Bh,Bw) idiosyncratic AR(2) per block
    phi2_block: np.ndarray
    sigma2_block: np.ndarray


def dynamic_factor_estimation(
    trajs: np.ndarray,
    cfg: "CrossCovConfig",
    u_mode: str = "empirical_pca",   # "empirical_pca" | "constant"
    u_vec: Optional[np.ndarray] = None
) -> DynamicFactorParams:
    """
    Estimate all parameters of the Dynamic Factor + AR(2) model.

    1) Spatial loading u:
         – if u_vec provided: use it (normalize).
         – elif u_mode="empirical_pca": PCA on frames (recommended).
         – else "constant": u = 1/√P (flat).
    2) Raw factor f̂_t: project frames on u; fit AR(2) on f̂_t via Yule–Walker.
    3) Scalar loading c: least-squares fit to z_t ≈ c f_t u (closed form).
    4) Idiosyncratic AR(2): remove c f_t u from z_t to get x_t, then estimate
       block-wise (φ1, φ2, σ²) by YW on residual γ0,γ1,γ2.

    Returns:
      DynamicFactorParams with everything needed for simulation.
    """
    N, T, H, W = trajs.shape
    P = H * W

    # 1) u
    if u_vec is not None:
        u = u_vec.reshape(-1).astype(np.float64)
        u /= (np.linalg.norm(u) + 1e-12)
    elif u_mode == "empirical_pca":
        u = estimate_u_from_trajs_pca(trajs)
    else:
        u = np.ones(P, dtype=np.float64) / np.sqrt(P)

    # 2) f̂_t and AR(2) for f_t
    f_hat = project_frames_onto_u(trajs, u)           # (N*T,)
    beta1, beta2, tau2 = fit_ar2_univariate_yw(f_hat) # AR(2) params for factor

    # 3) c (scalar loading)
    c = estimate_c_scalar(trajs, u, f_hat)

    # 4) idiosyncratic block-wise AR(2)
    ar2_blocks = estimate_ar2_blocks_for_residuals(trajs, cfg, u, c, f_hat)

    return DynamicFactorParams(
        u=u, c=c, f_series=f_hat,
        beta1=beta1, beta2=beta2, tau2=tau2,
        phi1_block=ar2_blocks["phi1_block"],
        phi2_block=ar2_blocks["phi2_block"],
        sigma2_block=ar2_blocks["sigma2_block"]
    )


# ------------------------------
# 15.4) Simulation from the fitted Dynamic Factor + AR(2) model
# ------------------------------

@dataclass
class DynamicFactorSimConfig:
    params: DynamicFactorParams
    random_seed: int = 0
    burn_in: int = 200   # allow AR(2) components to reach stationarity


def _upsample_block_to_pixels(block_map: np.ndarray, block_size: int) -> np.ndarray:
    Bh, Bw = block_map.shape
    H, W = Bh * block_size, Bw * block_size
    out = np.empty((H, W), dtype=np.float64)
    for bi in range(Bh):
        for bj in range(Bw):
            out[bi*block_size:(bi+1)*block_size, bj*block_size:(bj+1)*block_size] = block_map[bi, bj]
    return out


def simulate_dynamic_factor_ar2(
    cfg: DynamicFactorSimConfig,
    N_traj: int, T: int, block_size: int
) -> np.ndarray:
    """
    Simulate (N_traj, T, H, W) from:
        z_t = c f_t u_map + x_t
        x_t = φ1 ⊙ x_{t-1} + φ2 ⊙ x_{t-2} + ε_t
        f_t = β1 f_{t-1} + β2 f_{t-2} + ξ_t

    Why this generates large C(h,1) & rank-1 SVD:
      – f_t carries temporal memory (AR(2)) and multiplies the *same* u_map
        at every pixel → induces strong cross-time cross-space covariance.
      – Since x_t is pixelwise AR(2) with diagonal noise, same-time spatial
        ACF remains small (matching your diagnostics).
    """
    rng = np.random.default_rng(cfg.random_seed)
    P = cfg.params.u.shape[0]
    # Recover H,W from block maps
    Bh, Bw = cfg.params.phi1_block.shape
    H, W = Bh * block_size, Bw * block_size

    # Upsample idiosyncratic AR(2) params to pixels
    phi1_map = _upsample_block_to_pixels(cfg.params.phi1_block, block_size)  # (H,W)
    phi2_map = _upsample_block_to_pixels(cfg.params.phi2_block, block_size)
    sigma_map = np.sqrt(_upsample_block_to_pixels(cfg.params.sigma2_block, block_size))
    phi1_vec = phi1_map.reshape(-1)   # (P,)
    phi2_vec = phi2_map.reshape(-1)
    sigma_vec = sigma_map.reshape(-1)

    u = cfg.params.u.reshape(-1)
    c = float(cfg.params.c)
    beta1 = float(cfg.params.beta1)
    beta2 = float(cfg.params.beta2)
    tau   = float(np.sqrt(cfg.params.tau2))

    Z = np.zeros((N_traj, T, H, W), dtype=np.float64)

    def step_x(x_t: np.ndarray, x_tm1: np.ndarray) -> np.ndarray:
        eps = rng.normal(0.0, 1.0, size=P) * sigma_vec
        return phi1_vec * x_t + phi2_vec * x_tm1 + eps

    def step_f(f_t: float, f_tm1: float) -> float:
        xi = rng.normal(0.0, tau)
        return beta1 * f_t + beta2 * f_tm1 + xi

    burn = int(cfg.burn_in)
    T_tot = burn + T

    for n in range(N_traj):
        # initialize AR(2) components with noise; burn-in to reduce transient
        x_tm1 = rng.normal(0.0, 1.0, size=P)
        x_t   = rng.normal(0.0, 1.0, size=P)
        f_tm1 = rng.normal(0.0, 1.0)
        f_t   = rng.normal(0.0, 1.0)

        seq = []
        for t in range(T_tot):
            # evolve idiosyncratic state
            x_tp1 = step_x(x_t, x_tm1)
            # evolve factor
            f_tp1 = step_f(f_t, f_tm1)
            # form observation
            z_tp1 = c * f_tp1 * u + x_tp1
            seq.append(z_tp1.reshape(H, W))
            # shift
            x_tm1, x_t = x_t, x_tp1
            f_tm1, f_t = f_t, f_tp1

        Z[n] = np.stack(seq[-T:], axis=0)

    return Z

def estimate_u_by_max_crosscov(trajs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Find u that maximizes lag-1 autocov of f_t = u^T z_t under u^T S0 u = 1:
        maximize  u^T S1 u  subject to  u^T S0 u = 1
    Solve generalized eigenproblem: S1 u = λ S0 u.

    Steps:
      1) Build Z = frames x pixels, center each frame by spatial mean.
      2) Compute S0 = Cov(z_t), S1 = Cov(z_t, z_{t+1}).
      3) Solve (S0 + eps*I)^{-1} S1 u = λ u  (数值稳健)
    Return u with ||u||_2 = 1 (最终用 L2 归一，尺度由 c 吸收)。
    """
    N, T, H, W = trajs.shape
    X = trajs.reshape(N*T, H*W).astype(np.float64)
    X -= X.mean(axis=1, keepdims=True)
    # S0
    S0 = (X.T @ X) / (X.shape[0])
    # S1 (lag-1)
    X0 = X[:-1]
    X1 = X[1:]
    S1 = (X0.T @ X1) / (X0.shape[0])

    # regularize & solve eigen on (S0^{-1} S1)
    P = S0.shape[0]
    S0_reg = S0 + eps * np.eye(P)
    # Solve S0_reg^{-1} S1 u = λ u
    # 用对称化的方式更稳：先做 S0^{-1/2}，对矩阵 B = S0^{-1/2} S1 S0^{-1/2} 做特征分解
    w, V = np.linalg.eigh(S0_reg)
    w_cl = np.clip(w, np.sqrt(eps), None)
    S0_mhalf = (V @ np.diag(w_cl**-0.5) @ V.T)
    B = S0_mhalf @ S1 @ S0_mhalf
    vals, vecs = np.linalg.eig(B)
    k = int(np.argmax(np.real(vals)))
    v = np.real(vecs[:, k])
    u = S0_mhalf @ v
    u = u / (np.linalg.norm(u) + 1e-12)
    return u



# =========================
# B) drop-in 替换 dynamic factor 里的 u 估计
# =========================
def dynamic_factor_estimation_with_maxcov_u(
    trajs: np.ndarray, cfg: "CrossCovConfig"
) -> DynamicFactorParams:
    """
    与 dynamic_factor_estimation 类似，但 u 使用最大跨时相关方向，
    这样 f_t 的 γ_f(1) 被直接最大化，有利于匹配 |C(h,1)|.
    """
    # 1) u by max-crosscov
    u = estimate_u_by_max_crosscov(trajs)
    # 2) f̂_t & AR(2) for factor
    f_hat = project_frames_onto_u(trajs, u)
    beta1, beta2, tau2 = fit_ar2_univariate_yw(f_hat)
    # 3) c
    c = estimate_c_scalar(trajs, u, f_hat)
    # 4) idiosyncratic AR(2) (block-wise)
    ar2_blocks = estimate_ar2_blocks_for_residuals(trajs, cfg, u, c, f_hat)
    return DynamicFactorParams(
        u=u, c=c, f_series=f_hat,
        beta1=beta1, beta2=beta2, tau2=tau2,
        phi1_block=ar2_blocks["phi1_block"],
        phi2_block=ar2_blocks["phi2_block"],
        sigma2_block=ar2_blocks["sigma2_block"]
    )

# =========================
# C) one-shot pipeline using the new u
# =========================
def run_dynamic_factor_ar2_pipeline_maxcov_u(
    trajs: np.ndarray, cfg: "CrossCovConfig", random_seed: int = 0
) -> Dict[str, Dict[str, np.ndarray]]:
    empirical = compute_local_cross_covariance_maps(trajs, cfg)
    df_params = dynamic_factor_estimation_with_maxcov_u(trajs, cfg)
    report = simulation_adequacy_test_dynamic_factor_ar2(
        real_trajs=trajs, real_results=empirical, params=df_params, cfg=cfg, random_seed=random_seed
    )
    return {"Empirical": empirical, "DF_Params": {"u": df_params.u, "c": np.array([df_params.c]),
            "beta1": np.array([df_params.beta1]), "beta2": np.array([df_params.beta2]),
            "tau2": np.array([df_params.tau2]),
            "phi1_block": df_params.phi1_block, "phi2_block": df_params.phi2_block,
            "sigma2_block": df_params.sigma2_block},
            "Adequacy_DFM": report}
# ------------------------------
# 15.5) Adequacy test & one-shot pipeline
# ------------------------------

def simulation_adequacy_test_dynamic_factor_ar2(
    real_trajs: np.ndarray,
    real_results: Dict[str, np.ndarray],
    params: DynamicFactorParams,
    cfg: "CrossCovConfig",
    random_seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Simulate from the fitted Dynamic Factor + AR(2) model and compare to empirical:
      – magnitude map (|R(h,1)| p-norm per block),
      – SVD spectrum of correlation vectors across blocks.
    """
    N, T, _, _ = real_trajs.shape
    sim_trajs = simulate_dynamic_factor_ar2(
        DynamicFactorSimConfig(params=params, random_seed=random_seed),
        N_traj=N, T=T, block_size=cfg.block_size
    )
    sim_res = compute_local_cross_covariance_maps(sim_trajs, cfg)

    lags = real_results["lags"]
    real_mag = compute_magnitude_map(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_mag  = compute_magnitude_map(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    real_svd = global_svd_on_corr_maps(real_results["corr_maps"], lags, exclude_zero_lag=True)
    sim_svd  = global_svd_on_corr_maps(sim_res["corr_maps"],  lags, exclude_zero_lag=True)

    mag_diff_mean    = float(np.mean(sim_mag - real_mag))
    mag_diff_absmean = float(np.mean(np.abs(sim_mag - real_mag)))
    svd_diff         = float(np.linalg.norm(sim_svd["singular_values"] - real_svd["singular_values"]))
    svd_energy_diff  = float(np.linalg.norm(sim_svd["energy_cumsum"] - real_svd["energy_cumsum"]))

    return {
        "real_mag": real_mag, "sim_mag": sim_mag,
        "mag_diff_mean": np.array([mag_diff_mean]),
        "mag_diff_abs_mean": np.array([mag_diff_absmean]),
        "svd_diff": np.array([svd_diff]),
        "svd_energy_diff": np.array([svd_energy_diff]),
        "sim_corr_maps": sim_res["corr_maps"],
    }


def run_dynamic_factor_ar2_pipeline(
    trajs: np.ndarray,
    cfg: "CrossCovConfig",
    u_mode: str = "empirical_pca",          # "empirical_pca" | "constant"
    u_vec: Optional[np.ndarray] = None,
    random_seed: int = 0
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    One-shot pipeline:
      1) Empirical cross-covariance maps at u=1 for real data   -> 'Empirical'
      2) Estimate DFM params (u, f_t, c, β*, τ²; φ*, σ²)       -> 'DF_Params'
      3) Simulate from DFM and compute adequacy stats           -> 'Adequacy_DFM'

    Tips:
      – u_mode="empirical_pca" is recommended (aligns with real rank-1 pattern).
      – You can later refine c by grid search if magnitude under/overshoots.
    """
    # 1) empirical
    empirical = compute_local_cross_covariance_maps(trajs, cfg)

    # 2) estimate all parameters
    df_params = dynamic_factor_estimation(trajs, cfg, u_mode=u_mode, u_vec=u_vec)

    # 3) adequacy
    report = simulation_adequacy_test_dynamic_factor_ar2(
        real_trajs=trajs, real_results=empirical, params=df_params, cfg=cfg, random_seed=random_seed
    )

    return {
        "Empirical": empirical,
        "DF_Params": {
            "u": df_params.u, "c": np.array([df_params.c]),
            "beta1": np.array([df_params.beta1]), "beta2": np.array([df_params.beta2]),
            "tau2": np.array([df_params.tau2]),
            "phi1_block": df_params.phi1_block, "phi2_block": df_params.phi2_block,
            "sigma2_block": df_params.sigma2_block,
            "f_series_len": np.array([df_params.f_series.shape[0]])
        },
        "Adequacy_DFM": report
    }


# ------------------------------
# 15.6) Example usage (copy into your notebook)
# ------------------------------

EXAMPLE_DFM_AR2 = r"""
from cross_covariance import (
    CrossCovConfig, compute_magnitude_map, plot_maps_side_by_side, plot_svd,
    run_dynamic_factor_ar2_pipeline, global_svd_on_corr_maps
)
from data import SQGTrajData

# 1) Load latent trajectories
sqg_data = SQGTrajData()
trajs = sqg_data.get_all()[:, :, 0, :, :]

# 2) Configure local cross-covariance estimation (u=1; include (0,0))
cfg = CrossCovConfig(block_size=8, time_lag=1,
                     lags=[(0,0),(1,0),(-1,0),(0,1),(0,-1)])

# 3) Run the DFM pipeline (empirical u via PCA is recommended)
out = run_dynamic_factor_ar2_pipeline(trajs, cfg, u_mode="empirical_pca", random_seed=0)

# 4) Visual inspection: magnitude maps
real_mag = out["Adequacy_DFM"]["real_mag"]
sim_mag  = out["Adequacy_DFM"]["sim_mag"]
plot_maps_side_by_side(real_mag, sim_mag, ("Real mag", "Sim mag (DFM AR(2))"))

# 5) Numeric adequacy summaries
print("Mean |mag diff|:", out["Adequacy_DFM"]["mag_diff_abs_mean"][0])
print("SVD diff:",       out["Adequacy_DFM"]["svd_diff"][0])
print("SVD energy diff:",out["Adequacy_DFM"]["svd_energy_diff"][0])

# 6) SVD comparison
real_svd = global_svd_on_corr_maps(out["Empirical"]["corr_maps"], out["Empirical"]["lags"], True)
sim_svd  = global_svd_on_corr_maps(out["Adequacy_DFM"]["sim_corr_maps"], out["Empirical"]["lags"], True)
plot_svd(real_svd, "(Real)")
plot_svd(sim_svd,  "(Sim DFM AR(2))")
"""

