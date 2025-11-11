import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from data import SQGData


# ======================================================
# Utilities
# ======================================================

def zscore(x):
    """Standardize a 1D array."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd, mu, sd


def acf_1d(x, max_lag=40):
    """Compute 1D autocorrelation function up to max_lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    if var <= 0 or len(x) == 0:
        return np.zeros(max_lag + 1)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2
    ac = ac[mid: mid + max_lag + 1] / (len(x) * var)
    return ac


def fit_ar2_ls(x):
    """Least-squares AR(2) fit for standardized series x."""
    x = np.asarray(x, dtype=np.float64)
    y = x[2:]
    X = np.stack([x[1:-1], x[:-2]], axis=1)  # (T-2, 2)
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi1, phi2 = phi[0], phi[1]
    return float(phi1), float(phi2)


def make_dataset_for_residual_pred(x_norm, r_norm):
    """Create supervised dataset for self-supervised residual prediction."""
    T = len(x_norm)
    X_list, Y_list = [], []
    for t in range(2, T):
        X_list.append([x_norm[t - 1], x_norm[t - 2]])
        Y_list.append(r_norm[t])
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
    return X, Y


# ======================================================
# Model
# ======================================================

class TinyMLP(nn.Module):
    """Small MLP for detecting nonlinear temporal structure."""
    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = F.gelu(self.fc1(z))
        h = F.gelu(self.fc2(h))
        return self.fc3(h)


# ======================================================
# Main experiment pipeline
# ======================================================

def run_whiten_probe(noise0, max_lag=30, epochs=200):
    """
    Perform global temporal correlation analysis + AR(2) whitening +
    nonlinear predictability probing using a small MLP.
    """

    # ----------------------------------------------
    # Extract global time series x_t from 3D field
    # ----------------------------------------------
    arr = np.array(noise0)
    assert arr.ndim == 3 and arr.shape[0] >= 64, \
        "noise0 must be (T, X, Y) with T >= 64"

    x_t = arr.mean(axis=(1, 2)).astype(np.float64)
    source_info = f"Used noise0 with shape {arr.shape}"

    # ----------------------------------------------
    # Standardize x_t
    # ----------------------------------------------
    x_norm, mu_x, sd_x = zscore(x_t)

    # ----------------------------------------------
    # Fit AR(2) and compute residual
    # ----------------------------------------------
    phi1, phi2 = fit_ar2_ls(x_norm)

    r = x_norm.copy()
    for t in range(2, len(x_norm)):
        r[t] = x_norm[t] - (phi1 * x_norm[t - 1] + phi2 * x_norm[t - 2])

    # ----------------------------------------------
    # Standardize residual
    # ----------------------------------------------
    r_norm, mu_r, sd_r = zscore(r)

    # ----------------------------------------------
    # Build self-supervised residual prediction dataset
    # ----------------------------------------------
    X, Y = make_dataset_for_residual_pred(x_norm, r_norm)
    n = len(X)
    n_train = int(0.8 * n)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    # ----------------------------------------------
    # Train tiny MLP
    # ----------------------------------------------
    torch.manual_seed(0)
    model = TinyMLP(in_dim=2, hidden=64, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    train_losses, val_losses = [], []

    for epoch in tqdm(range(epochs), desc="Training MLP", ncols=80):
        model.train()
        opt.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            vloss = loss_fn(pred_val, Y_val_t).item()
            train_losses.append(loss.item())
            val_losses.append(vloss)

    # ----------------------------------------------
    # Evaluate predictions
    # ----------------------------------------------
    with torch.no_grad():
        rhat_all = model(torch.from_numpy(X)).squeeze(1).numpy()

    acf_x = acf_1d(x_norm, max_lag=max_lag)
    acf_r = acf_1d(r_norm, max_lag=max_lag)
    acf_rhat = acf_1d(rhat_all, max_lag=max_lag)

    # ----------------------------------------------
    # Plots
    # ----------------------------------------------

    # Training curves
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Training/Validation Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ACF: x_norm
    plt.figure()
    plt.stem(np.arange(len(acf_x)), acf_x, basefmt=" ")
    plt.title("ACF of x_t (standardized)")
    plt.xlabel("lag")
    plt.ylabel("rho")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ACF: residual r_norm
    plt.figure()
    plt.stem(np.arange(len(acf_r)), acf_r, basefmt=" ")
    plt.title("ACF of residual r_t (after AR(2) whitening)")
    plt.xlabel("lag")
    plt.ylabel("rho")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ACF: predicted residual rhat + CI
    plt.figure()
    lags = np.arange(len(acf_rhat))
    plt.stem(lags, acf_rhat, basefmt=" ")

    # White-noise 95% CI
    N = len(rhat_all)
    ci = 1.96 / np.sqrt(N)
    plt.axhline(ci, color='red', linestyle='--', linewidth=1)
    plt.axhline(-ci, color='red', linestyle='--', linewidth=1)

    plt.title("ACF of predicted residual $\hat r_t$ by tiny MLP")
    plt.xlabel("lag")
    plt.ylabel("rho")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Scatter plot
    plt.figure()
    plt.scatter(r_norm[2:], rhat_all, s=6)
    plt.title("Scatter: r_t (norm) vs predicted $\hat r_t$")
    plt.xlabel("r_t (norm)")
    plt.ylabel("predicted $\hat r_t$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Summary
    summary = {
        "data_source": source_info,
        "phi1": phi1,
        "phi2": phi2,
        "train_loss_last": train_losses[-1],
        "val_loss_last": val_losses[-1],
        "acf_x_lag1": float(acf_x[1]),
        "acf_x_lag2": float(acf_x[2]),
        "acf_r_lag1": float(acf_r[1]),
        "acf_r_lag2": float(acf_r[2]),
        "acf_rhat_lag1": float(acf_rhat[1]),
        "acf_rhat_lag2": float(acf_rhat[2]),
    }

    return summary


sqg = SQGData("SQG.npy", "inverted_SQG.npy")
noise0 = sqg.get_field()

summary = run_whiten_probe(noise0)
print(summary)
