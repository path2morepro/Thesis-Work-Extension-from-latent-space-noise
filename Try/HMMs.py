import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from data import SQGTrajData
from sampler import Sampler


# ================================================================
# 1. Utilities for building latent transitions and low-rank A
# ================================================================

def build_latent_transition_pairs(sqg_data: SQGTrajData, level: int = 0):
    """
    Extract latent trajectories and build (z_t, z_{t+1}) pairs.

    Assumes:
        sqg_data.get_all() returns an array of shape
            (N_traj, T_traj, levels, H, W)
        and we use a single level (e.g. level=0) as latent.

    Returns:
        Z_t   : (N_pairs, D) design matrix
        Z_tp1 : (N_pairs, D) response matrix
        H, W  : spatial height and width
    """
    trajs = sqg_data.get_all()[:, :, level, :, :]  # (N_traj, T_traj, H, W)
    N_traj, T_traj, H, W = trajs.shape
    D = H * W

    # Flatten (H, W) → D
    traj_flat = trajs.reshape(N_traj, T_traj, D).astype(np.float32)

    # Build all (z_t, z_{t+1}) pairs
    Z_t   = traj_flat[:, :-1, :].reshape(-1, D)   # (N_traj*(T-1), D)
    Z_tp1 = traj_flat[:,  1:, :].reshape(-1, D)

    return Z_t, Z_tp1, H, W


def fit_lowrank_A_RRR(Z_t: np.ndarray,
                      Z_tp1: np.ndarray,
                      rank_r: int = 64,
                      ridge: float = 1e-2):
    """
    Fit a low-rank transition matrix A using Reduced-Rank Regression (RRR).

    We solve:
        minimize ||Y - X A||_F^2  subject to rank(A) <= r

    where:
        X = Z_t   : (N, D)
        Y = Z_tp1 : (N, D)
        A        : (D, D)

    Steps:
      1. Compute the OLS solution A_ols.
      2. Compute C = Y^T X (X^T X + λI)^{-1} X^T Y.
      3. Take the top-r eigenvectors of C.
      4. Project A_ols onto the subspace spanned by those eigenvectors.

    Args:
        Z_t   : (N, D) design matrix
        Z_tp1 : (N, D) response matrix
        rank_r: target rank of A
        ridge : λ for numerical stabilization (X^T X + λI)

    Returns:
        A_rrr : (D, D) low-rank transition matrix
        info  : dict with some diagnostics
    """
    X = Z_t
    Y = Z_tp1
    N, D = X.shape
    assert Y.shape == (N, D)

    # G = X^T X + λI
    G = X.T @ X
    if ridge > 0.0:
        G = G + ridge * np.eye(D, dtype=np.float32)

    # A_ols^T = G^{-1} X^T Y
    XtY = X.T @ Y  # (D, D)
    A_ols_T = np.linalg.solve(G, XtY)  # (D, D)
    A_ols = A_ols_T.T                  # (D, D)

    # C = Y^T X G^{-1} X^T Y = Y^T P_X Y
    XAolsT = X @ A_ols_T               # (N, D)
    C = Y.T @ XAolsT                   # (D, D), symmetric

    # Eigen-decomposition of C, take top-r eigenvectors
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]      # descending
    V_r = evecs[:, idx[:rank_r]]       # (D, r)

    # Reduced-rank solution A_rrr = A_ols V_r V_r^T
    A_rrr = A_ols @ (V_r @ V_r.T)

    info = {
        "ridge": ridge,
        "rank_r": rank_r,
        "eigvals_top": evals[idx[:rank_r]],
        "A_ols_normF": float(np.linalg.norm(A_ols, "fro")),
        "A_rrr_normF": float(np.linalg.norm(A_rrr, "fro"))
    }
    return A_rrr.astype(np.float32), info


def compute_ou_residuals(Z_t: np.ndarray,
                         Z_tp1: np.ndarray,
                         A: np.ndarray):
    """
    Compute OU residuals eps_t = z_{t+1} - z_t A.

    Args:
        Z_t   : (N, D)
        Z_tp1 : (N, D)
        A     : (D, D)

    Returns:
        eps   : (N, D) residuals
    """
    mean_tp1 = Z_t @ A  # (N, D)
    eps = Z_tp1 - mean_tp1
    return eps.astype(np.float32)


# ================================================================
# 2. Sigma (noise covariance) models
#    - DiagonalSigma
#    - LowRankSigma
#    - KroneckerSigma (row⊗col)
# ================================================================

class DiagonalSigma:
    """
    Diagonal noise covariance:
        Σ = diag(σ_1^2, ..., σ_D^2)

    Sampling:
        ε ~ N(0, Σ)  <=>  ε_i ~ N(0, σ_i^2) independently.
    """
    def __init__(self, sigma2: np.ndarray):
        sigma2 = np.asarray(sigma2, dtype=np.float32)
        assert sigma2.ndim == 1
        self.sigma2 = sigma2
        self.D = sigma2.shape[0]

    @classmethod
    def from_residuals(cls, eps: np.ndarray):
        """
        Estimate diagonal variances from residuals.

        eps: (N, D)
        σ_i^2 = mean(eps[:, i]^2) across time/trajectories.
        """
        eps = np.asarray(eps, dtype=np.float32)
        sigma2 = np.mean(eps * eps, axis=0)
        return cls(sigma2)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample ε ~ N(0, diag(σ^2)) as a 1D vector of shape (D,).
        """
        std = np.sqrt(self.sigma2)
        return rng.normal(loc=0.0, scale=std, size=self.D).astype(np.float32)


class LowRankSigma:
    """
    Low-rank noise covariance:
        Σ ≈ U Λ U^T + δ^2 I

    where:
        U  : (D, r) orthonormal columns (principal directions)
        Λ  : diag(λ_1, ..., λ_r)  eigenvalues (variance along each direction)
        δ^2: small isotropic jitter for numerical stability

    Sampling:
        η  ~ N(0, I_r)
        ξ  ~ N(0, I_D)
        ε  = U diag(sqrt(Λ)) η + δ ξ
    """
    def __init__(self,
                 U: np.ndarray,
                 lambdas: np.ndarray,
                 delta2: float = 1e-6):
        U = np.asarray(U, dtype=np.float32)
        lambdas = np.asarray(lambdas, dtype=np.float32)

        assert U.ndim == 2
        D, r = U.shape
        assert lambdas.shape == (r,)

        self.U = U               # (D, r)
        self.lambdas = lambdas   # (r,)
        self.D = D
        self.r = r
        self.delta2 = float(delta2)

    @classmethod
    def from_residuals(cls,
                       eps: np.ndarray,
                       rank_r: int = 32,
                       delta2: float = 1e-6):
        """
        Estimate low-rank covariance from residuals using SVD.

        eps: (N, D), residuals ε_t
        We compute sample covariance:
            S = (1/(N-1)) eps_centered^T eps_centered
          and approximate it by top-r eigenpairs.

        For readability, we directly use np.linalg.svd on eps_centered.
        For large D,N this can be expensive; in production you may want
        a randomized SVD instead.
        """
        eps = np.asarray(eps, dtype=np.float32)
        N, D = eps.shape

        # Center residuals
        mean_eps = np.mean(eps, axis=0, keepdims=True)
        X = eps - mean_eps  # (N, D)

        # SVD: X = U_svd S V^T
        # Sample covariance S_cov = V diag(S^2/(N-1)) V^T
        U_svd, S_svd, Vt_svd = np.linalg.svd(X, full_matrices=False)
        V = Vt_svd.T  # (D, min(N,D))

        r = min(rank_r, V.shape[1])
        U_cov = V[:, :r]                     # (D, r)
        lambdas = (S_svd[:r] ** 2) / (N - 1) # (r,)

        return cls(U_cov, lambdas, delta2=delta2)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample ε ~ N(0, UΛU^T + δ^2 I).

        Steps:
        - Sample η ~ N(0, I_r)
        - Sample ξ ~ N(0, I_D)
        - ε = U diag(sqrt(Λ)) η + δ ξ
        """
        eta = rng.normal(loc=0.0, scale=1.0, size=self.r).astype(np.float32)
        xi = rng.normal(loc=0.0, scale=1.0, size=self.D).astype(np.float32)

        sqrt_lambdas = np.sqrt(self.lambdas)  # (r,)
        # Project latent noise into D-dim space along principal directions
        part_lowrank = self.U @ (sqrt_lambdas * eta)  # (D,)
        part_iso = np.sqrt(self.delta2) * xi          # (D,)

        return (part_lowrank + part_iso).astype(np.float32)


class KroneckerSigma:
    """
    Kronecker-structured noise covariance for 2D fields:

        Σ ≈ Σ_col ⊗ Σ_row

    where:
        Σ_row: (H, H) covariance across rows
        Σ_col: (W, W) covariance across columns

    If we reshape each residual ε_t into a matrix E_t (H, W),
    then vec(E_t) ~ MN(0, Σ_row, Σ_col) implies
    cov(vec(E_t)) = Σ_col ⊗ Σ_row.

    We estimate Σ_row, Σ_col from residual matrices using:
        Σ_row ≈ (1/(N*W)) Σ_t E_t E_t^T
        Σ_col ≈ (1/(N*H)) Σ_t E_t^T E_t

    (You can see this as a simple moment-matching approach.)
    """
    def __init__(self,
                 Sigma_row: np.ndarray,
                 Sigma_col: np.ndarray,
                 jitter: float = 1e-6):
        Sigma_row = np.asarray(Sigma_row, dtype=np.float32)
        Sigma_col = np.asarray(Sigma_col, dtype=np.float32)
        assert Sigma_row.shape[0] == Sigma_row.shape[1]
        assert Sigma_col.shape[0] == Sigma_col.shape[1]

        H = Sigma_row.shape[0]
        W = Sigma_col.shape[0]

        # Ensure positive definiteness by adding a small jitter
        self.Sigma_row = Sigma_row + jitter * np.eye(H, dtype=np.float32)
        self.Sigma_col = Sigma_col + jitter * np.eye(W, dtype=np.float32)

        # Cholesky factors for sampling
        self.L_row = np.linalg.cholesky(self.Sigma_row)  # (H, H)
        self.L_col = np.linalg.cholesky(self.Sigma_col)  # (W, W)

        self.H = H
        self.W = W
        self.D = H * W

    @classmethod
    def from_residuals(cls,
                       eps: np.ndarray,
                       H: int,
                       W: int,
                       jitter: float = 1e-6):
        """
        Estimate row/column covariance from residuals.

        eps: (N, D) residuals, where D = H * W
        We reshape each residual to a matrix E_t of shape (H, W),
        center them (optional), then accumulate:

            Σ_row ≈ (1/(N*W)) Σ_t E_t E_t^T
            Σ_col ≈ (1/(N*H)) Σ_t E_t^T E_t
        """
        eps = np.asarray(eps, dtype=np.float32)
        N, D = eps.shape
        assert D == H * W

        # Center residuals globally
        mean_eps = np.mean(eps, axis=0, keepdims=True)
        X = eps - mean_eps  # (N, D)

        Sigma_row = np.zeros((H, H), dtype=np.float32)
        Sigma_col = np.zeros((W, W), dtype=np.float32)

        for n in range(N):
            E = X[n].reshape(H, W)  # (H, W)
            Sigma_row += E @ E.T    # (H, H)
            Sigma_col += E.T @ E    # (W, W)

        Sigma_row /= float(N * W)
        Sigma_col /= float(N * H)

        return cls(Sigma_row, Sigma_col, jitter=jitter)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """
        Sample ε ~ N(0, Σ_col ⊗ Σ_row).

        Using matrix-normal representation:
            G ~ N(0, I)^(H×W)
            E = L_row @ G @ L_col^T
            vec(E) ~ N(0, Σ_col ⊗ Σ_row)
        """
        G = rng.normal(loc=0.0, scale=1.0, size=(self.H, self.W)).astype(np.float32)
        E = self.L_row @ G @ self.L_col.T
        return E.reshape(-1).astype(np.float32)


# ================================================================
# 3. Low-rank OU model wrapper
# ================================================================

class LowRankOUModel:
    """
    Low-rank OU model:

        z_{t+1} = z_t A + ε_t

    where:
        A       : (D, D) low-rank transition matrix
        sigma   : noise covariance model (DiagonalSigma / LowRankSigma / KroneckerSigma)
    """
    def __init__(self,
                 A: np.ndarray,
                 sigma_model,
                 rng: np.random.Generator):
        A = np.asarray(A, dtype=np.float32)
        assert A.ndim == 2
        D1, D2 = A.shape
        assert D1 == D2, "A must be square."

        assert hasattr(sigma_model, "sample")
        assert sigma_model.D == D1, "A and sigma_model dimension mismatch."

        self.A = A
        self.sigma_model = sigma_model
        self.D = D1
        self.rng = rng

    def rollout(self,
                z0: np.ndarray,
                T_steps: int) -> np.ndarray:
        """
        Roll out the OU model for T_steps starting from z0.

        Args:
            z0      : (D,) initial latent state
            T_steps : length of the rollout

        Returns:
            Z_roll  : (T_steps, D) latent trajectory
        """
        z0 = np.asarray(z0, dtype=np.float32)
        assert z0.shape[0] == self.D

        Z = np.zeros((T_steps, self.D), dtype=np.float32)
        Z[0] = z0

        for t in range(1, T_steps):
            mean_next = Z[t - 1] @ self.A  # (D,)
            eps_t = self.sigma_model.sample(self.rng)
            Z[t] = mean_next + eps_t

        return Z


# ================================================================
# 4. Visualization pipeline: truth vs inversion vs OU (3 sigmas)
# ================================================================

def _set_colorbar_for_axes_image(im):
    """
    Attach a small horizontal colorbar above the axes of im.
    """
    ax = im.axes
    cax = inset_axes(
        ax,
        width="100%",
        height="5%",
        loc="upper center",
        bbox_to_anchor=(0, 0.18, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")


def visualize_ou_sigma_models(
    sqg_data: SQGTrajData,
    sampler: Sampler,
    A: np.ndarray,
    sigma_models: dict,
    traj_id: int = 0,
    members: int = 5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    data_std: float = 2660.0,
    seed: int = 0,
):
    """
    Visualize:
        Col 1: Truth physics
        Col 2: Physics decoded from TRUE latent (baseline inversion)
        Col 3+: Physics decoded from OU rollout with different sigma models

    Args:
        sqg_data     : SQGTrajData instance
        sampler      : Sampler instance (latent -> physics)
        A            : (D, D) low-rank OU transition matrix
        sigma_models : dict(name -> sigma_model_instance)
                       where each sigma_model has .sample() and .D
        traj_id      : which trajectory to visualize
        members      : how many time steps to show
        device       : torch.device
        data_std     : de-normalization factor for physics
        seed         : random seed for OU noise
    """
    rng = np.random.default_rng(seed)

    # ---------------------------
    # 1) Load a trajectory (truth + latent)
    # ---------------------------
    true_phy = sqg_data.get_traj(traj_id=traj_id, dataset="truth")  # (T, C, H, W)
    noise    = sqg_data.get_traj(traj_id=traj_id, dataset="noise")  # (T, C, H, W)

    T_total, C, H, W = true_phy.shape
    assert members <= T_total, "members cannot exceed trajectory length."

    true_phyN = true_phy[:members]   # (members, C, H, W)
    noiseN    = noise[:members]      # (members, C, H, W)

    # ---------------------------
    # 2) Baseline inversion: from TRUE latent noise → physics
    # ---------------------------
    x = torch.from_numpy(noiseN).to(torch.float32).to(device)         # (T, C, H, W)
    x_invert = sampler.sample(x).detach().cpu().numpy()               # (T, C, H, W)

    truth_phys  = true_phyN[:, 0, :, :]                               # (T, H, W), channel-0
    invert_phys = x_invert[:, 0, :, :] * data_std                     # (T, H, W)

    # ---------------------------
    # 3) Prepare OU rollouts for each sigma model
    # ---------------------------
    D = H * W
    lat0 = noise[0, 0, :, :]                      # (H, W), channel-0 latent
    z0 = lat0.reshape(-1).astype(np.float32)      # (D,)

    ou_results = {}   # name -> dict(phys=(T,H,W), rmse=(T,))
    all_phys_for_scale = [truth_phys, invert_phys]

    for name, sigma_model in sigma_models.items():
        assert sigma_model.D == D, f"Sigma model {name} has wrong dimension."

        # Each sigma model gets its own OU model with its own RNG (for reproducibility)
        model_rng = np.random.default_rng(seed + hash(name) % 100000)
        ou_model = LowRankOUModel(A=A, sigma_model=sigma_model, rng=model_rng)

        Z_roll = ou_model.rollout(z0=z0, T_steps=members)     # (T, D)
        rollout_lat = Z_roll.reshape(members, H, W)           # (T, H, W)

        # Build 2-channel latent tensor: channel-0 = rollout, channel-1 = zeros
        rollout_lat_2ch = np.zeros((members, 2, H, W), dtype=np.float32)
        rollout_lat_2ch[:, 0, :, :] = rollout_lat

        rollout_lat_t = torch.from_numpy(rollout_lat_2ch).to(torch.float32).to(device)
        rollout_phys = sampler.sample(rollout_lat_t).detach().cpu().numpy()[:, 0, :, :] * data_std

        # Per-frame RMSE vs truth
        rmse = np.sqrt(((rollout_phys - truth_phys) ** 2).mean(axis=(1, 2)))

        ou_results[name] = {
            "phys": rollout_phys,
            "rmse": rmse,
        }
        all_phys_for_scale.append(rollout_phys)

    # ---------------------------
    # 4) Decide global color scale across all panels
    # ---------------------------
    vmin = min(arr.min() for arr in all_phys_for_scale)
    vmax = max(arr.max() for arr in all_phys_for_scale)

    # ---------------------------
    # 5) Plot: T rows × (2 + n_sigma) columns
    # ---------------------------
    sigma_names = list(sigma_models.keys())
    n_sigma = len(sigma_names)
    n_cols = 2 + n_sigma

    fig, axs = plt.subplots(
        members, n_cols,
        figsize=(4.0 * n_cols, 2.2 * members),
        constrained_layout=True
    )
    axs = np.atleast_2d(axs)
    cmap = plt.get_cmap("viridis", 10)

    for t in range(members):
        # Col-1: Truth
        ax = axs[t, 0]
        ax.set_aspect("equal")
        ax.axis("off")
        im = ax.imshow(truth_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Truth (t={t})", fontsize=11)
        _set_colorbar_for_axes_image(im)

        # Col-2: baseline inversion from TRUE latent
        ax = axs[t, 1]
        ax.set_aspect("equal")
        ax.axis("off")
        im = ax.imshow(invert_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        # we don't compute RMSE here because it's typically small / already known
        ax.set_title("Invert from TRUE latent", fontsize=11)
        _set_colorbar_for_axes_image(im)

        # Col-3+: OU + different sigma models
        for j, name in enumerate(sigma_names, start=2):
            phys = ou_results[name]["phys"]
            rmse = ou_results[name]["rmse"][t]

            ax = axs[t, j]
            ax.set_aspect("equal")
            ax.axis("off")
            im = ax.imshow(phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"OU ({name}) RMSE={rmse:.2f}", fontsize=11)
            _set_colorbar_for_axes_image(im)

    plt.show()


# ================================================================
# 5. High-level helper to run the whole pipeline once
# ================================================================

def run_lowrank_ou_sigma_comparison(
    rank_A: int = 64,
    rank_sigma: int = 32,
    traj_id: int = 0,
    members: int = 5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    data_std: float = 2660.0,
    seed: int = 0,
):
    """
    Convenience function to:
      1) Load data and sampler
      2) Fit low-rank A via RRR
      3) Compute residuals and estimate three sigma structures
         - diagonal
         - low-rank
         - Kronecker
      4) Visualize rollouts for all three sigma models

    NOTE:
        This function assumes:
          - SQGTrajData() can be constructed without arguments
          - Sampler(...) constructor call is available to you
        Adapt the Sampler init part to your own API if needed.
    """
    # 1) Instantiate data and sampler (adapt to your paths / config)
    sqg_data = SQGTrajData()
    device = device

    # You might need to adapt these arguments to your own Sampler class:
    model_path = "../best_model.pth"
    image_shape = (2, 64, 64)
    steps = 100
    invert_steps = 100
    debug = False

    eps_func = lambda t: 0.1 * (1.0 - t)
    invert_eps_func = lambda t: 0.0 * (1.0 - t)

    sampler = Sampler(
        device=device,
        members=members,
        eps=eps_func,
        steps=steps,
        invert_eps=invert_eps_func,
        invert_steps=invert_steps,
        model_path=model_path,
        debug=debug,
    )

    # 2) Build latent transitions and fit low-rank A
    Z_t, Z_tp1, H, W = build_latent_transition_pairs(sqg_data, level=0)
    A_lr, info = fit_lowrank_A_RRR(Z_t, Z_tp1, rank_r=rank_A, ridge=1e-2)
    print("Low-rank A info:", info)

    # 3) Compute residuals and estimate different sigma structures
    eps_resid = compute_ou_residuals(Z_t, Z_tp1, A_lr)  # (N, D)
    D = eps_resid.shape[1]
    assert D == H * W

    sigma_diag = DiagonalSigma.from_residuals(eps_resid)
    sigma_lowrank = LowRankSigma.from_residuals(eps_resid, rank_r=rank_sigma, delta2=1e-6)
    sigma_kron = KroneckerSigma.from_residuals(eps_resid, H=H, W=W, jitter=1e-6)

    sigma_models = {
        "diag": sigma_diag,
        "lowrank": sigma_lowrank,
        "kron": sigma_kron,
    }

    # 4) Visualization
    visualize_ou_sigma_models(
        sqg_data=sqg_data,
        sampler=sampler,
        A=A_lr,
        sigma_models=sigma_models,
        traj_id=traj_id,
        members=members,
        device=device,
        data_std=data_std,
        seed=seed,
    )


# use case
# from HMMs import run_lowrank_ou_sigma_comparison

# run_lowrank_ou_sigma_comparison(
#     rank_A=64,
#     rank_sigma=32,
#     traj_id=0,
#     members=5,
#     data_std=2660.0,
# )
