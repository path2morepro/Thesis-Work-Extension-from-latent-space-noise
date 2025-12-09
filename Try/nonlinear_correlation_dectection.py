import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from data import SQGTrajData

# --------------------
# Utilities
# --------------------
def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd, mu, sd

def acf_1d(x, max_lag=40):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    if var <= 0 or len(x) == 0:
        return np.zeros(max_lag+1)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2
    ac = ac[mid: mid + max_lag + 1] / (len(x) * var)
    return ac

def fit_ar2_ls(x):
    """
    Least-squares AR(2) fit on a 1D series x.
    x should already be standardized (z-scored).
    """
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    y = x[2:]
    X = np.stack([x[1:-1], x[:-2]], axis=1)  # (T-2, 2) for [x_{t-1}, x_{t-2}]
    # Solve least squares
    phi, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi1, phi2 = phi[0], phi[1]
    return float(phi1), float(phi2)

def make_dataset_for_residual_pred(x_norm, r_norm):
    """
    Build supervised pairs for self-supervised residual prediction:
    Input: [x_{t-1}, x_{t-2}]  (causal, no leakage)
    Target: r_t
    """
    T = len(x_norm)
    X_list = []
    Y_list = []
    for t in range(2, T):
        X_list.append([x_norm[t-1], x_norm[t-2]])
        Y_list.append(r_norm[t])
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
    return X, Y

class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, out_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
    def forward(self, z):
        h = F.gelu(self.fc1(z))
        # Gaussian Error Linear Unit
        h = F.gelu(self.fc2(h))
        out = self.fc3(h)
        return out


def prepare_global_dataset(data: SQGTrajData):
    """
    将所有 trajectory 合并成一个大 dataset:
        X = [x_{t-1}, x_{t-2}]
        Y = r_t
    """
    X_all, Y_all = [], []

    for i in range(data.num_traj):

        # ---- 1. 取 noise trajectory，并平均到 2D ----
        traj = data.get_traj(i, dataset="noise")  # (T, 2, 64, 64)
        traj = traj.mean(axis=1)                  # (T, 64, 64)

        # ---- 2. 提取全局 mean 时间序列 ----
        x_t = traj.mean(axis=(1, 2))

        # ---- 3. 标准化 ----
        x_norm, _, _ = zscore(x_t)

        # ---- 4. 拟合 AR(2) ----
        phi1, phi2 = fit_ar2_ls(x_norm)

        # ---- 5. 得到残差 ----
        residual = x_norm.copy()
        for t in range(2, len(x_norm)):
            residual[t] = x_norm[t] - (phi1 * x_norm[t-1] + phi2 * x_norm[t-2])

        r_norm, _, _ = zscore(residual)

        # ---- 6. 构造 (x_{t-1}, x_{t-2}) → r_t ----
        for t in range(2, len(x_norm)):
            X_all.append([x_norm[t-1], x_norm[t-2]])
            Y_all.append(r_norm[t])

    X_all = np.array(X_all, dtype=np.float32)
    Y_all = np.array(Y_all, dtype=np.float32).reshape(-1, 1)

    print(f"全局 dataset 构建完成: X={X_all.shape}, Y={Y_all.shape}")
    return X_all, Y_all

def run_whiten_probe_multi_epoch(data, max_lag=40, epochs=200):
    """
    使用所有 trajectories 的所有 sample 构成一个 epoch 训练。
    训练后画出原始 run_whiten_probe 的所有图，并返回
    R^2 和 corr(r_t, r_hat).
    """
    
    # --------------------------------------------
    # 1. 构建全局 Dataset
    # --------------------------------------------
    X, Y = prepare_global_dataset(data)
    N = len(X)

    n_train = int(N * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    # --------------------------------------------
    # 2. 训练 MLP
    # --------------------------------------------
    torch.manual_seed(0)
    model = TinyMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []

    for ep in tqdm(range(epochs), desc="Training FULL dataset epoch"):
        model.train()
        opt.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, Y_train_t)
        loss.backward()
        opt.step()

        # val
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), Y_val_t).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

    # --------------------------------------------
    # 3. 用所有 sample 推理，计算 ACF + R^2 + corr
    # --------------------------------------------
    with torch.no_grad():
        rhat = model(torch.from_numpy(X)).squeeze(1).numpy()  # 预测值 \hat r_t

    r_true = Y.squeeze()   # 真实残差 r_t（已经 zscore 过）

    # ACF
    acf_pred = acf_1d(rhat, max_lag=max_lag)
    acf_r = acf_1d(r_true, max_lag=max_lag)

    # 相关系数 corr(r_t, r_hat)
    corr = np.corrcoef(r_true, rhat)[0, 1]

    # 决定系数 R^2 = 1 - SS_res / SS_tot
    ss_res = np.sum((r_true - rhat) ** 2)
    ss_tot = np.sum((r_true - r_true.mean()) ** 2)
    R2 = 1.0 - ss_res / ss_tot

    print(f"corr(r_t, r_hat) = {corr:.6f}")
    print(f"R^2 = {R2:.6f}")

    # 额外：可以顺便看一下新的残差 e_t = r_t - r_hat 的 ACF（可选）
    e_t = r_true - rhat
    acf_err = acf_1d(e_t, max_lag=max_lag)

    # --------------------------------------------
    # 4. 画出所有图：和 run_whiten_probe 一样
    # --------------------------------------------

    # Loss 曲线
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Training / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # ACF of residual r_t
    plt.figure()
    plt.stem(np.arange(len(acf_r)), acf_r, basefmt=" ")
    plt.title("ACF of residual r_t (global)")
    plt.xlabel("lag")
    plt.grid(True)
    plt.tight_layout()

    # ACF of predicted residual r_hat
    plt.figure()
    plt.stem(np.arange(len(acf_pred)), acf_pred, basefmt=" ")
    ci = 1.96 / np.sqrt(len(rhat))
    plt.axhline(ci, color='red', linestyle='--')
    plt.axhline(-ci, color='red', linestyle='--')
    plt.title("ACF of predicted residual $\hat r_t$")
    plt.grid(True)
    plt.tight_layout()

    # ACF of new error e_t = r_t - r_hat（可选图，帮你压死争议）
    plt.figure()
    plt.stem(np.arange(len(acf_err)), acf_err, basefmt=" ")
    plt.title("ACF of error e_t = r_t - $\hat r_t$")
    plt.xlabel("lag")
    plt.grid(True)
    plt.tight_layout()

    # Scatter
    plt.figure()
    plt.scatter(r_true, rhat, s=4)
    plt.xlabel("True r_t (norm)")
    plt.ylabel("Pred $\hat r_t$")
    plt.title("Scatter of residual prediction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return dict(
        train_loss=train_losses[-1],
        val_loss=val_losses[-1],
        acf_r=acf_r,
        acf_pred=acf_pred,
        acf_err=acf_err,
        corr=corr,
        R2=R2,
    )
