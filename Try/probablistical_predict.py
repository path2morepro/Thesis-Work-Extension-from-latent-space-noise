import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data import SQGTrajData
from sampler import Sampler

def visualize_truth_invert_rollout(members=5, traj_id=0, sigma2=0, a=0):
    """
    Plot three columns per time step:
      Col-1: Truth physics field
      Col-2: Physics decoded from TRUE latent (inverse mapping of 'noise' via sampler.sample)
      Col-3: Physics decoded from ROLLOUT latent (your OU/other dynamics), then inverse mapping

    Globals expected:
      - new_data: provides .get_traj(traj_id, dataset="truth"/"noise")
      - sampler: provides .sample(tensor) -> inverse mapping (latent -> physics)
      - device: torch device
      - a: OU 'a' coefficient (scalar or vector of size D)
      - sigma2: OU noise variance (scalar or vector broadcastable to D)
      - data_std: scaling factor to de-standardize physics (e.g., 2660)

    Args:
      members (int): rollout length and also the number of frames shown
      traj_id (int): which trajectory to fetch

    Returns:
      None (displays a figure)
    """
    # init
    new_data = SQGTrajData()
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = lambda t: 0.1 * (1 - t)  # Noise when sampling
    invert_eps = lambda t: 0. * (1 - t)  # Noise when inverting
    steps = 100
    invert_steps = 100
    debug = False

    sampler = Sampler(device, members, eps, steps, invert_eps, invert_steps, model_path, debug)
    # ---------------------------
    # 1) Load a short clip
    # ---------------------------
    true_phy = new_data.get_traj(traj_id=traj_id, dataset="truth")  # expected (T, C, H, W)
    noise    = new_data.get_traj(traj_id=traj_id, dataset="noise")  # expected (T, C, H, W)

    # take first `members` frames for side-by-side comparison
    true_phyN = true_phy[:members, ...]    # (members, C, H, W)
    noiseN    = noise[:members, ...]       # (members, C, H, W)

    # ---------------------------
    # 2) Invert TRUE latent to physics (column-2)
    #    We assume 'noiseN' is the latent input to sampler.sample()
    # ---------------------------
    x = torch.from_numpy(noiseN).to(torch.float32).to(device)            # (members, C, H, W)
    x_invert = sampler.sample(x).detach().cpu().numpy()                  # (members, C, H, W)

    # pick channel-0 to compare (aligns with your example) and de-standardize
    truth_phys = true_phyN[:, 0, ...]                                    # (members, H, W)
    invert_phys = x_invert[:, 0, ...] * 2660                         # (members, H, W)

    # ---------------------------
    # 3) ROLLOUT latent (column-3)  <<< ROLLOUT LOGIC HERE >>>
    #    You can swap this block with a different latent dynamics later.
    #    Current logic: simple OU AR(1) per-dimension with coeff 'a' and noise var 'sigma2'.
    # ---------------------------
    # seed from the very first latent frame's channel-0 (as your example)
    lat0 = noise[0, 0, ...]                                              # (H, W)
    H, W = lat0.shape
    x0 = lat0.flatten()                                                  # (D,)
    D = x0.shape[0]

    xs = np.zeros((members, D), dtype=np.float32)
    xs[0] = x0

    # allow scalar or vector a/sigma2; compute sigma from sigma2
    sigma = np.sqrt(sigma2)

    # ---- OU rollout (you may replace from here) ----
    # AR(1): x_{t+1} = a * x_t + N(0, sigma^2 I)
    for i in range(1, members):
        noise_term = np.random.randn(D).astype(np.float32) * sigma
        xi = a * xs[i - 1] + noise_term
        xs[i] = xi
    # ---- end of ROLLOUT LOGIC ----

    # to 4D latent: (T, C=2, H, W) because sampler expects 2-ch like your code
    rollout_lat = xs.reshape(members, H, W)[:, None, ...]                # (T, 1, H, W)
    rollout_lat = np.concatenate([rollout_lat, np.zeros_like(rollout_lat)], axis=1)  # (T, 2, H, W)

    # inverse map rollout latent -> physics, pick channel-0 and de-standardize
    rollout_lat_t = torch.from_numpy(rollout_lat).to(torch.float32).to(device)
    rollout_phys = sampler.sample(rollout_lat_t).detach().cpu().numpy()  # (T, 2, H, W)
    rollout_phys = rollout_phys[:, 0, ...] * 2660                    # (T, H, W)

    # ---------------------------
    # 4) Sanity checks
    # ---------------------------
    assert truth_phys.shape == invert_phys.shape == rollout_phys.shape, \
        f"Shape mismatch: {truth_phys.shape}, {invert_phys.shape}, {rollout_phys.shape}"

    T = members

    # unified color scale across THREE columns
    vmin = min(truth_phys.min(), invert_phys.min(), rollout_phys.min())
    vmax = max(truth_phys.max(), invert_phys.max(), rollout_phys.max())

    # per-frame RMSE for columns 2 & 3 vs truth
    rmse_invert  = np.sqrt(((invert_phys  - truth_phys) ** 2).mean(axis=(1, 2)))   # (T,)
    rmse_rollout = np.sqrt(((rollout_phys - truth_phys) ** 2).mean(axis=(1, 2)))   # (T,)

    # ---------------------------
    # 5) Plot: T rows × 3 cols
    # ---------------------------
    def set_cbar(im):
        ax = im.axes
        cax = inset_axes(ax,
                         width="100%",
                         height="5%",
                         loc='upper center',
                         bbox_to_anchor=(0, 0.18, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

    fig, axs = plt.subplots(T, 3, figsize=(12, 2.2 * T), constrained_layout=True)
    axs = np.atleast_2d(axs)
    cmap = plt.get_cmap('viridis', 10)

    for t in range(T):
        # Col-1: Truth
        ax = axs[t, 0]
        ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(truth_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Truth (t={t})", fontsize=11)
        set_cbar(im)

        # Col-2: Inverted from TRUE latent
        ax = axs[t, 1]
        ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(invert_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Invert from TRUE latent (RMSE={rmse_invert[t]:.3f})", fontsize=11)
        set_cbar(im)

        # Col-3: Inverted from ROLLOUT latent
        ax = axs[t, 2]
        ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(rollout_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Rollout → Invert (RMSE={rmse_rollout[t]:.3f})", fontsize=11)
        set_cbar(im)

    plt.show()




def rollout_linear_lowrank(z0, A, T_steps, sigma2=None, random_state=None):
    """
    Linear rollout: z_{t+1} = z_t A + epsilon, epsilon ~ N(0, diag(sigma2)).
    If sigma2 is None: deterministic.
    
    Args:
      z0         : (D,) initial state
      A          : (D, D) transition matrix (low-rank)
      T_steps    : int rollout length
      sigma2     : (D,) or scalar variance; if None => 0
      random_state: np.random.RandomState or None
      
    Returns:
      Z_roll : (T_steps, D)
    """
    if random_state is None:
        random_state = np.random

    D = z0.shape[0]
    Z = np.zeros((T_steps, D), dtype=np.float32)
    Z[0] = z0.astype(np.float32)

    add_noise = sigma2 is not None
    if add_noise and np.isscalar(sigma2):
        sigma = float(np.sqrt(sigma2))
    elif add_noise:
        sigma = np.sqrt(sigma2).astype(np.float32)

    for t in range(1, T_steps):
        mean_next = Z[t-1] @ A   # (D,)
        if add_noise:
            if np.isscalar(sigma2):
                eps = random_state.randn(D).astype(np.float32) * sigma
            else:
                eps = random_state.randn(D).astype(np.float32) * sigma
            Z[t] = mean_next + eps
        else:
            Z[t] = mean_next
    return Z


def visualize_truth_invert_rollout(members, traj_id, new_data, sampler, device, sigma2, A_lr):
    """
    Col-1: Truth physics
    Col-2: Physics decoded from TRUE latent
    Col-3: Physics decoded from ROLLOUT latent (low-rank linear rollout), then inverse mapping

    Globals expected:
      - new_data.get_traj(traj_id, dataset="truth"/"noise")
      - sampler.sample(tensor)  # latent -> physics
      - device
      - A_lr: low-rank transition matrix (D x D)
      - sigma2: optional noise variance for rollout (scalar or (D,))
      - data_std: de-standardization factor (e.g., 2660)
      - rollout_linear_lowrank(z0, A, T_steps, sigma2=None) -> (T_steps, D)
    """

    # ---------------------------
    # 1) Load a short clip
    # ---------------------------
    true_phy = new_data.get_traj(traj_id=traj_id, dataset="truth")  # (T, C, H, W)
    noise    = new_data.get_traj(traj_id=traj_id, dataset="noise")  # (T, C, H, W)

    # first `members` frames
    true_phyN = true_phy[:members, ...]    # (members, C, H, W)
    noiseN    = noise[:members, ...]       # (members, C, H, W)

    # ---------------------------
    # 2) Invert TRUE latent to physics (column-2)
    # ---------------------------
    x = torch.from_numpy(noiseN).to(torch.float32).to(device)            # (members, C, H, W)
    x_invert = sampler.sample(x).detach().cpu().numpy()                  # (members, C, H, W)

    # compare channel-0
    truth_phys  = true_phyN[:, 0, ...]                                   # (members, H, W)
    invert_phys = x_invert[:, 0, ...] * 2660                         # (members, H, W)

    # ---------------------------
    # 3) ROLLOUT latent via low-rank A (REPLACED BLOCK)
    #    rollout_linear_lowrank should implement:
    #    z_{t+1} = z_t @ A_lr + eps, eps~N(0, diag(sigma2)) if sigma2 provided
    # ---------------------------
    lat0 = noise[0, 0, ...]                                              # (H, W)
    H, W = lat0.shape
    D = H * W
    z0_lat = lat0.reshape(-1)                                            # (D,)

    # rollout in latent
    Z_roll = rollout_linear_lowrank(z0_lat, A=A_lr, T_steps=members, sigma2=sigma2)  # (T, D)

    # to sampler format: (T, 2, H, W)
    rollout_lat = Z_roll.reshape(members, H, W)[:, None, ...]            # (T, 1, H, W)
    rollout_lat = np.concatenate([rollout_lat, np.zeros_like(rollout_lat)], axis=1)  # (T, 2, H, W)

    # inverse map rollout latent -> physics, channel-0
    rollout_lat_t = torch.from_numpy(rollout_lat).to(torch.float32).to(device)
    rollout_phys = sampler.sample(rollout_lat_t).detach().cpu().numpy()[:, 0, ...] * 2660  # (T, H, W)

    # ---------------------------
    # 4) Sanity checks
    # ---------------------------
    assert truth_phys.shape == invert_phys.shape == rollout_phys.shape, \
        f"Shape mismatch: {truth_phys.shape}, {invert_phys.shape}, {rollout_phys.shape}"

    T = members

    # unified color scale
    vmin = min(truth_phys.min(), invert_phys.min(), rollout_phys.min())
    vmax = max(truth_phys.max(), invert_phys.max(), rollout_phys.max())

    # per-frame RMSE vs truth
    rmse_invert  = np.sqrt(((invert_phys  - truth_phys) ** 2).mean(axis=(1, 2)))   # (T,)
    rmse_rollout = np.sqrt(((rollout_phys - truth_phys) ** 2).mean(axis=(1, 2)))   # (T,)

    # ---------------------------
    # 5) Plot: T rows × 3 cols
    # ---------------------------
    def set_cbar(im):
        ax = im.axes
        cax = inset_axes(ax, width="100%", height="5%", loc='upper center',
                         bbox_to_anchor=(0, 0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')

    fig, axs = plt.subplots(T, 3, figsize=(12, 2.2 * T), constrained_layout=True)
    axs = np.atleast_2d(axs)
    cmap = plt.get_cmap('viridis', 10)

    for t in range(T):
        # Col-1: Truth
        ax = axs[t, 0]; ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(truth_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Truth (t={t})", fontsize=11); set_cbar(im)

        # Col-2: Inverted from TRUE latent
        ax = axs[t, 1]; ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(invert_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Invert from TRUE latent (RMSE={rmse_invert[t]:.3f})", fontsize=11); set_cbar(im)

        # Col-3: Inverted from ROLLOUT latent
        ax = axs[t, 2]; ax.set_aspect('equal'); ax.axis('off')
        im = ax.imshow(rollout_phys[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Rollout → Invert (RMSE={rmse_rollout[t]:.3f})", fontsize=11); set_cbar(im)

    plt.show()


def fit_lowrank_A_RRR(Z_t, Z_tp1, rank_r=32, ridge=1e-3):
    """
    Fit A (D x D) with rank<=r via Reduced-Rank Regression (RRR).
    Minimize ||Y - X A||_F subject to rank(A) <= r.
    
    Args:
      Z_t   : (N, D) design matrix X
      Z_tp1 : (N, D) response     Y
      rank_r: target rank
      ridge : lambda for (X^T X + lambda I)^-1 (numerical stability)
      
    Returns:
      A_rrr : (D, D) low-rank transition matrix
      info  : dict with diagnostics
    """
    X = Z_t
    Y = Z_tp1
    N, D = X.shape
    assert Y.shape == (N, D)

    # G = X^T X + lambda I
    G = X.T @ X
    if ridge > 0:
        G = G + ridge * np.eye(D)

    # Solve G^{-1} X^T Y without explicit inverse
    # A_ols^T = G^{-1} (X^T Y)  -> use solve
    XtY = X.T @ Y
    # We solve G * A_ols^T = XtY  -> A_ols^T = solve(G, XtY)
    A_ols_T = np.linalg.solve(G, XtY)
    A_ols = A_ols_T.T  # (D, D)

    # Compute C = Y^T X G^{-1} X^T Y = Y^T P_X Y (ridge version)
    # We already have B = G^{-1} X^T Y = A_ols^T
    # So X G^{-1} X^T Y = X @ (A_ols^T) = X @ A_ols_T
    XAolsT = X @ A_ols_T            # (N, D)
    C = Y.T @ XAolsT                # (D, D)

    # Eigen-decomposition of C
    # We need top-r eigenvectors (largest eigenvalues)
    evals, evecs = np.linalg.eigh(C)  # symmetric
    idx = np.argsort(evals)[::-1]     # descending
    V_r = evecs[:, idx[:rank_r]]      # (D, r)

    # Reduced-Rank solution: A_rrr = A_ols V_r V_r^T
    A_rrr = A_ols @ (V_r @ V_r.T)

    info = {
        "ridge": ridge,
        "rank_r": rank_r,
        "eigvals_top": evals[idx[:rank_r]],
        "A_ols_normF": np.linalg.norm(A_ols, "fro"),
        "A_rrr_normF": np.linalg.norm(A_rrr, "fro")
    }
    return A_rrr, info
