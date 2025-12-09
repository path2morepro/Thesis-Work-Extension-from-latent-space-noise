"""
cross_cov_pipeline.py

Pipeline for LOCAL space–time cross-covariance analysis on latent fields.

You said you want to:
    - Use SQGTrajData
    - Call: trajs = sqg_data.get_all()[:, :, 0, :, :]  # (N_traj, T, H, W)
    - Then run a pipeline that estimates:
        * local variance map  sigma^2(s_0)
        * local cross-covariance   C(h, u=1; s_0)
        * local cross-correlation  R(h, 1; s_0)
        * global SVD / effective-rank diagnostics
      and provide convenient plotting helpers.

All important mathematical steps are commented in detail.
Trivial things like shape unpacking / simple assignments are *not* over-commented.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Configuration dataclass
# ---------------------------------------------------------------------


@dataclass
class CrossCovConfig:
    """
    Configuration for local cross-covariance estimation.

    Attributes
    ----------
    block_size : int
        Side length (in pixels) of each square spatial block B(s_0).
        Blocks are taken as non-overlapping tiles of the full field.
        Example: H=W=64, block_size=8 -> 8x8 blocks in a 8x8 grid.

    time_lag : int
        Temporal lag u to use in C(h, u; s_0). We mostly care about u = 1.

    lags : List[Tuple[int, int]]
        List of spatial lags h = (delta_row, delta_col) to evaluate
        cross-covariance at. h = (0,0) gives local variance at lag (u, u).

        Typical choice:
            [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
        where:
            - (0,0) mainly used for variance at u=0
            - others check nearest-neighbor coupling across time.

    center_block : bool
        If True, each block is treated as a local window centered at some s_0.
        For now blocks are strictly non-overlapping, so this flag is only
        for semantic clarity (we always use non-overlapping tiles).
    """
    block_size: int = 8
    time_lag: int = 1
    lags: List[Tuple[int, int]] = None
    center_block: bool = True

    def __post_init__(self):
        if self.lags is None:
            # Default: self + four-neighbors
            self.lags = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


# ---------------------------------------------------------------------
# 2. Low-level helper: compute aligned slices for spatial lag
# ---------------------------------------------------------------------


def _aligned_block_slices(
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    di: int,
    dj: int,
) -> Tuple[slice, slice, slice, slice]:
    """
    For a given block [r0:r1, c0:c1] and spatial lag h = (di, dj),
    compute two *aligned* slices (rows_x, cols_x) and (rows_y, cols_y)
    such that:

        - (rows_x, cols_x) indexes pixels in the block at location s
        - (rows_y, cols_y) indexes pixels at location s + h
        - both sets stay inside the same block
        - array shapes match exactly, so we can compute covariance
    """
    # --- Row dimension ---
    if di >= 0:
        rows_x_start = r0
        rows_x_end = r1 - di
        rows_y_start = r0 + di
        rows_y_end = r1
    else:
        rows_x_start = r0 - di  # di < 0 -> subtract negative -> move start down
        rows_x_end = r1
        rows_y_start = r0
        rows_y_end = r1 + di    # di < 0 -> end shrinks

    # --- Column dimension ---
    if dj >= 0:
        cols_x_start = c0
        cols_x_end = c1 - dj
        cols_y_start = c0 + dj
        cols_y_end = c1
    else:
        cols_x_start = c0 - dj
        cols_x_end = c1
        cols_y_start = c0
        cols_y_end = c1 + dj

    # Check if there is at least one pixel left
    if rows_x_end <= rows_x_start or cols_x_end <= cols_x_start:
        raise ValueError("Lag too large for this block, no overlapping pixels.")

    rows_x = slice(rows_x_start, rows_x_end)
    rows_y = slice(rows_y_start, rows_y_end)
    cols_x = slice(cols_x_start, cols_x_end)
    cols_y = slice(cols_y_start, cols_y_end)

    return rows_x, cols_x, rows_y, cols_y


# ---------------------------------------------------------------------
# 3. Core computation: local cross-covariance per block
# ---------------------------------------------------------------------
def _estimate_block_cross_covariances(
    trajs: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    lags: List[Tuple[int, int]],
    time_lag: int,
) -> Tuple[np.ndarray, float]:
    """
    We consider a block B = [r0:r1, c0:c1]. For each spatial lag h=(di,dj),
    and temporal lag u (time_lag), we want:

        C(h,u; s_0) = Cov( z_t(s), z_{t+u}(s + h) ),
        where s runs over all pixels in B that keep s and s+h inside B,
        and t runs from 0..T-u-1 over time, and also across all trajectories.

    In practice we estimate this as:

        1. Construct aligned arrays X and Y:
               X = z_t(s)         for all valid (traj, t, s)
               Y = z_{t+u}(s+h)   for the corresponding (traj, t, s)
        2. Center them by subtracting sample means:
               X_c = X - mean(X)
               Y_c = Y - mean(Y)
        3. Covariance estimate:
               cov = mean( X_c * Y_c )

       This corresponds to the usual unbiased-ish covariance estimate
       (strict 1/(N-1) factor not important here because we only use
        covariance *shape* and relative magnitude, not exact variance).

    Parameters
    ----------
    trajs : np.ndarray
        Array of shape (N_traj, T, H, W) with latent fields.
    r0, r1, c0, c1 : int
        Block boundaries in row/column indices.
    lags : list of (di, dj)
        Spatial lags at which to compute C(h, u; s_0).
    time_lag : int
        Temporal lag u.

    Returns
    -------
    covs : np.ndarray
        1D array of length len(lags). covs[k] = C(h_k, u; s_0).

    local_var : float
        Local variance estimate C(0,0; s_0), computed separately
        using lag h=(0,0) and u=0.
        Namely, Cov(block)
        This can be used later to normalize covariances into correlations.
    """
    N_traj, T, H, W = trajs.shape
    covs = np.zeros(len(lags), dtype=np.float64)

    # ------------------------------------------------------------
    # 1) First estimate local variance C(0,0; s_0) on the block.
    #    This is used later to convert covariance into correlation.
    # ------------------------------------------------------------
    # We gather all values z_t(s) in this block across all trajectories
    # and all time steps.
    block_data = trajs[:, :, r0:r1, c0:c1]  # shape (N_traj, T, bh, bw)
    # Flatten over (traj, time, within-block pixels)
    block_flat = block_data.reshape(-1).astype(np.float64)
    # Centering blocks' data (subtract empirical mean)
    block_flat_c = block_flat - block_flat.mean()
    # Variance is mean of squared deviations
    local_var = np.mean(block_flat_c ** 2)

    # If variance is numerically zero (rare), we keep it small but positive
    if local_var <= 0.0:
        local_var = 1e-12

    # ------------------------------------------------------------
    # 2) For each spatial lag h = (di, dj), estimate C(h, u; s_0).
    # ------------------------------------------------------------
    for k, (di, dj) in enumerate(lags):
        # We want pairs (z_t(s), z_{t+u}(s+h)) inside the *same* block.
        try:
            rows_x, cols_x, rows_y, cols_y = _aligned_block_slices(
                r0, r1, c0, c1, di, dj
            )
        except ValueError:
            # Lag too large to have any overlap; we simply set cov to 0.
            covs[k] = 0.0
            continue

        # X corresponds to z_t(s) in that overlapping region of the block
        # Y corresponds to z_{t+u}(s+h) in the shifted overlapping region
        # Along the time axis we align (t, t+u) by slicing:
        #     trajs[:, :T-u, ...] vs trajs[:, u:, ...]
        X = trajs[:, : T - time_lag, rows_x, cols_x]
        Y = trajs[:, time_lag:, rows_y, cols_y]

        # Sanity: X and Y must have the same shape.
        # Shape is (N_traj, T - u, bh_eff, bw_eff)
        if X.shape != Y.shape:
            raise RuntimeError("Shape mismatch after lag alignment.")

        # Flatten all (traj, time, pixel) dimensions into one long vector.
        X_flat = X.reshape(-1).astype(np.float64)
        Y_flat = Y.reshape(-1).astype(np.float64)

        # Center both X and Y to estimate covariance.
        # This is important: covariance is about fluctuations around mean,
        # not raw values.
        X_flat -= X_flat.mean()
        Y_flat -= Y_flat.mean()

        # Covariance = E[(X - E[X])*(Y - E[Y])]
        # Here we approximate expectation by empirical average.
        cov = np.mean(X_flat * Y_flat)
        covs[k] = cov

    return covs, float(local_var)


# ---------------------------------------------------------------------
# 4. High-level function: compute maps over all blocks
# ---------------------------------------------------------------------


def compute_local_cross_covariance_maps(
    trajs: np.ndarray,
    config: CrossCovConfig,
) -> Dict[str, np.ndarray]:
    """
    Compute local cross-covariance and -correlation maps over all blocks.

    The image (H x W) is partitioned into non-overlapping square blocks
    of size `block_size x block_size`. For each block, we call
    `_estimate_block_cross_covariances()` to obtain:

        - covs_block[h_index] = C(h, u; s_0)
        - local_var = C(0, 0; s_0)

    These are then aggregated into 3D arrays indexed by
    (block_row, block_col, lag_index).

    Parameters
    ----------
    trajs : np.ndarray
        Array of latent fields with shape (N_traj, T, H, W).
    config : CrossCovConfig
        Configuration specifying block_size, time_lag, lags, etc.

    Returns
    -------
    results : dict
        Dictionary containing:

        - "cov_maps": np.ndarray, shape (B_h, B_w, L)
            C(h, u; s_0) for each block and lag.
        - "corr_maps": np.ndarray, shape (B_h, B_w, L)
            Corresponding correlations R(h, u; s_0) = C / C(0,0; s_0).
        - "var_map": np.ndarray, shape (B_h, B_w)
            Local variance map C(0,0; s_0).
        - "lags": list of (di,dj)
            The list of spatial lags used.
    """
    assert trajs.ndim == 4, "trajs must be (N_traj, T, H, W)"
    N_traj, T, H, W = trajs.shape

    bs = config.block_size
    time_lag = config.time_lag
    lags = config.lags
    L = len(lags)

    # Number of blocks along each dimension.
    # We use floor division; any leftover rows/cols at the edge are ignored
    # for simplicity (you can pad if you want perfect coverage).
    n_blocks_h = H // bs
    n_blocks_w = W // bs

    cov_maps = np.zeros((n_blocks_h, n_blocks_w, L), dtype=np.float64)
    var_map = np.zeros((n_blocks_h, n_blocks_w), dtype=np.float64)

    # ------------------------------------------------------------
    # Loop over blocks and estimate covariances.
    # This is the main loop over space (blocks).
    # Time and trajectory dimensions are handled vectorially inside.
    # ------------------------------------------------------------
    for bi in range(n_blocks_h):
        for bj in range(n_blocks_w):
            r0 = bi * bs
            r1 = r0 + bs
            c0 = bj * bs
            c1 = c0 + bs

            covs_block, local_var = _estimate_block_cross_covariances(
                trajs, r0, r1, c0, c1, lags, time_lag
            )
            cov_maps[bi, bj, :] = covs_block
            var_map[bi, bj] = local_var

    # ------------------------------------------------------------
    # Convert covariance to correlation for each block:
    #    R(h, u; s_0) = C(h, u; s_0) / C(0,0; s_0)
    #
    # We broadcast var_map over the lag dimension to do this in one shot.
    # ------------------------------------------------------------
    # Avoid division by zero by enforcing a tiny lower bound.
    var_safe = np.maximum(var_map, 1e-12)
    corr_maps = cov_maps / var_safe[:, :, None]

    results = {
        "cov_maps": cov_maps,   # local C(h, u; s_0)
        "corr_maps": corr_maps, # local R(h, u; s_0)
        "var_map": var_map,     # local variance
        "lags": np.array(lags, dtype=int),
    }
    return results


# ---------------------------------------------------------------------
# 5. Aggregated diagnostics: magnitude map + global SVD
# ---------------------------------------------------------------------


def compute_magnitude_map(
    corr_maps: np.ndarray,
    lags: np.ndarray,
    exclude_zero_lag: bool = True,
    p_norm: float = 2.0,
) -> np.ndarray:
    """
    Compute a scalar "magnitude" map for cross-correlation strength.

    Idea
    ----
    At each block s_0 we have correlations R(h, u; s_0) for multiple h.
    We want a single scalar that summarizes "how strong overall is the
    cross-correlation with neighbors at lag u". (R(h, u; s_0) = C(h, u; s₀) / C(0,0; s₀))

    A natural choice is a p-norm over the lag dimension, e.g.:

        Mag(s_0) = ( sum_{h ∈ H'} |R(h, u; s_0)|^p )^{1/p}

    where H' excludes h=(0,0) if we only care about cross-pixel coupling.
    (h=(0,0) is auto-correlation across time at the same pixel.)

    Parameters
    ----------
    corr_maps : np.ndarray
        Array of shape (B_h, B_w, L) with correlations.
    lags : np.ndarray
        Array of shape (L, 2) with (di, dj) for each lag index.
    exclude_zero_lag : bool
        If True, ignore the lag (0,0) when computing magnitude; i.e. we
        only look at cross-pixel correlations.
    p_norm : float
        Which p-norm exponent to use. p=1 gives sum of absolute values,
        p=2 gives Euclidean norm, etc.

    Returns
    -------
    mag_map : np.ndarray
        Array of shape (B_h, B_w) with cross-correlation magnitude.
    """
    assert corr_maps.ndim == 3
    B_h, B_w, L = corr_maps.shape
    assert lags.shape == (L, 2)

    # Determine which lag indices to include.
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    # Select relevant lags along the last axis.
    selected = corr_maps[:, :, mask]

    # p-norm over lag dimension.
    # add small epsilon to avoid sqrt(0) issues for p<2, though not essential.
    eps = 1e-12
    mag_p = np.sum(np.abs(selected) ** p_norm, axis=-1) + eps
    mag_map = mag_p ** (1.0 / p_norm)
    return mag_map


def global_svd_on_corr_maps(
    corr_maps: np.ndarray,
    lags: np.ndarray,
    exclude_zero_lag: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Perform a global SVD on stacked correlation vectors across blocks.

    Purpose
    -------
    We want to know whether the cross-correlation pattern across lags
    is essentially low-dimensional. This is analogous to checking
    "effective rank" of the covariance structure.

    Construction
    ------------
    1. For each block (bi, bj), we have a correlation vector:
           r_{bi,bj} ∈ R^L
    2. We stack these as columns into a matrix:
           M ∈ R^{L x K}, K = number of blocks.
    3. We may optionally remove the (0,0) lag from M, because that is
       auto-correlation in time rather than cross-pixel coupling.
    4. Perform SVD:
           M = U Σ V^T
    5. Singular values Σ tell us how many dominant "patterns" of
       cross-correlation exist across the image.

    Parameters
    ----------
    corr_maps : np.ndarray
        Array of shape (B_h, B_w, L) with correlations.
    lags : np.ndarray
        Array of shape (L, 2) with (di, dj) for each lag index.
    exclude_zero_lag : bool
        Whether to drop lag h=(0,0) from the analysis.

    Returns
    -------
    info : dict
        Contains:
        - "singular_values": np.ndarray, shape (m,)
        - "energy_fraction": np.ndarray, shape (m,)
              fraction of total squared Frobenius norm explained
              by each singular value.
        - "energy_cumsum": np.ndarray, shape (m,)
              cumulative sum of energy_fraction.
        - "lags_used": np.ndarray
              lags corresponding to rows in M (after exclusion).
    """
    B_h, B_w, L = corr_maps.shape
    assert lags.shape == (L, 2)

    # Select whether to include h=(0,0) or not.
    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    corr_sel = corr_maps[:, :, mask]  # shape (B_h, B_w, L_sel)
    lags_sel = lags[mask]

    # Stack block vectors into columns of M ∈ R^{L_sel x K}
    L_sel = corr_sel.shape[2]
    K = B_h * B_w
    M = corr_sel.reshape(K, L_sel).T  # (L_sel, K)

    # SVD
    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    # Energy decomposition: squared singular values
    S2 = S ** 2
    total = S2.sum() + 1e-12
    energy_fraction = S2 / total
    energy_cumsum = np.cumsum(energy_fraction)

    info = {
        "singular_values": S,
        "energy_fraction": energy_fraction,
        "energy_cumsum": energy_cumsum,
        "lags_used": lags_sel,
    }
    return info


# ---------------------------------------------------------------------
# 6. Plotting helpers
# ---------------------------------------------------------------------


def plot_variance_and_magnitude_maps(
    var_map: np.ndarray,
    mag_map: np.ndarray,
    title_suffix: str = "",
):
    """
    Plot local variance map and cross-correlation magnitude map side by side.

    The first panel shows how variance σ^2(s_0) changes across blocks.
    The second panel shows how strong cross-correlation is in each block.

    Parameters
    ----------
    var_map : np.ndarray
        Shape (B_h, B_w) local variance C(0,0; s_0).
    mag_map : np.ndarray
        Shape (B_h, B_w) magnitude of cross-correlation.
    title_suffix : str
        Extra string appended to subplot titles to distinguish experiments.

    Notes
    -----
    Both maps are plotted in the block-grid index, not in pixel coordinates.
    Each cell roughly corresponds to one spatial block in the original field.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(var_map, origin="lower", aspect="equal")
    axs[0].set_title(f"Local variance map {title_suffix}")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(mag_map, origin="lower", aspect="equal")
    axs[1].set_title(f"Cross-corr magnitude map {title_suffix}")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def plot_corr_maps_for_each_lag(
    corr_maps: np.ndarray,
    lags: np.ndarray,
    exclude_zero_lag: bool = True,
    vmax: float = None,
):
    """
    Plot correlation maps R(h, u; s_0) for each spatial lag h in a grid.

    Each subplot corresponds to a different lag h = (di, dj).
    At each block (bi, bj), the color shows R(h, u; s_0).

    Parameters
    ----------
    corr_maps : np.ndarray
        Shape (B_h, B_w, L).
    lags : np.ndarray
        Shape (L, 2), spatial lags.
    exclude_zero_lag : bool
        If True, skip h=(0,0) in the plot.
    vmax : float or None
        Optional symmetric color scale bound. If None, use max absolute
        correlation across all plotted maps.

    Notes
    -----
    This is useful to see if there is any directional bias in correlations,
    e.g. if R((1,0),1; s_0) systematically differs from R((0,1),1; s_0).
    """
    B_h, B_w, L = corr_maps.shape
    assert lags.shape == (L, 2)

    if exclude_zero_lag:
        mask = ~((lags[:, 0] == 0) & (lags[:, 1] == 0))
    else:
        mask = np.ones(L, dtype=bool)

    corr_sel = corr_maps[:, :, mask]
    lags_sel = lags[mask]
    L_sel = corr_sel.shape[2]

    # Determine global symmetric color scale if not provided
    if vmax is None:
        vmax = np.max(np.abs(corr_sel)) + 1e-6

    n_cols = min(4, L_sel)
    n_rows = int(np.ceil(L_sel / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = np.atleast_2d(axs)

    for idx in range(L_sel):
        r = idx // n_cols
        c = idx % n_cols
        ax = axs[r, c]
        im = ax.imshow(
            corr_sel[:, :, idx],
            origin="lower",
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        di, dj = int(lags_sel[idx, 0]), int(lags_sel[idx, 1])
        ax.set_title(f"h=({di},{dj})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for idx in range(L_sel, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axs[r, c].axis("off")

    plt.tight_layout()
    plt.show()


def plot_singular_values(info: Dict[str, np.ndarray], title_suffix: str = ""):
    """
    Plot singular values and cumulative energy from `global_svd_on_corr_maps`.

    Parameters
    ----------
    info : dict
        Output from `global_svd_on_corr_maps`.
    title_suffix : str
        Extra string appended to the titles.
    """
    S = info["singular_values"]
    energy_fraction = info["energy_fraction"]
    energy_cumsum = info["energy_cumsum"]

    k = np.arange(1, len(S) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(k, S, marker="o")
    axs[0].set_xlabel("Singular value index")
    axs[0].set_ylabel("Singular value")
    axs[0].set_title(f"Singular values {title_suffix}")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(k, energy_cumsum, marker="o")
    axs[1].set_xlabel("Singular value index")
    axs[1].set_ylabel("Cumulative energy fraction")
    axs[1].set_ylim(0.0, 1.05)
    axs[1].set_title(f"Cumulative energy {title_suffix}")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 7. Example top-level helper (for Jupyter)
# ---------------------------------------------------------------------


def run_cross_covariance_pipeline():
    """
    Example usage inside a .py file.

    You do NOT need to call this automatically.
    Instead, in your Jupyter notebook you can:

        from cross_cov_pipeline import (
            CrossCovConfig,
            compute_local_cross_covariance_maps,
            compute_magnitude_map,
            global_svd_on_corr_maps,
            plot_variance_and_magnitude_maps,
            plot_corr_maps_for_each_lag,
            plot_singular_values,
        )
        from data import SQGTrajData

        sqg_data = SQGTrajData()
        trajs = sqg_data.get_all()[:, :, 0, :, :]   # (N_traj, T, H, W)

        config = CrossCovConfig(block_size=8, time_lag=1)
        results = compute_local_cross_covariance_maps(trajs, config)

        var_map = results["var_map"]
        corr_maps = results["corr_maps"]
        lags = results["lags"]

        mag_map = compute_magnitude_map(corr_maps, lags, exclude_zero_lag=True)

        plot_variance_and_magnitude_maps(var_map, mag_map)
        plot_corr_maps_for_each_lag(corr_maps, lags, exclude_zero_lag=True)

        svd_info = global_svd_on_corr_maps(corr_maps, lags, exclude_zero_lag=True)
        plot_singular_values(svd_info)

    This example function simply encodes the above steps so you can
    run it directly if you want a quick smoke test.
    """
    try:
        from data import SQGTrajData  # type: ignore
    except ImportError:
        raise ImportError(
            "Cannot import SQGTrajData. Make sure `data.py` is on PYTHONPATH."
        )

    sqg_data = SQGTrajData()
    trajs = sqg_data.get_all()[:, :, 0, :, :]  # (N_traj, T, H, W)

    config = CrossCovConfig(block_size=8, time_lag=1)
    results = compute_local_cross_covariance_maps(trajs, config)

    var_map = results["var_map"]
    corr_maps = results["corr_maps"]
    lags = results["lags"]

    mag_map = compute_magnitude_map(corr_maps, lags, exclude_zero_lag=True)
    plot_variance_and_magnitude_maps(var_map, mag_map, title_suffix="(example)")

    plot_corr_maps_for_each_lag(
        corr_maps, lags, exclude_zero_lag=True,
    )

    svd_info = global_svd_on_corr_maps(corr_maps, lags, exclude_zero_lag=True)
    plot_singular_values(svd_info, title_suffix="(example)")


# if __name__ == "__main__":
#     # This will only run if you execute the .py file directly:
#     #   python cross_cov_pipeline.py
#     # In notebooks you typically do not run this part.
#     run_cross_covariance_pipeline()
