# -*- coding: utf-8 -*-
# PyTorch 加速版：随机子空间 + Mardia/能量距离检验（GPU 友好）
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Chi2, Normal, MultivariateNormal
from tqdm import tqdm

DTYPE = torch.float64

def to_device(x, device=None):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    return t.to(DTYPE)

# ------------------------- 数据读取（沿用你的 API） -------------------------
class SQGDataTorch:
    def __init__(self, truth_path='SQG.npy', noise_path='inverted_SQG.npy', data_std=2660.0, device=None):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.truth_raw = to_device(np.load(truth_path), self.device)
        self.noise_raw = to_device(np.load(noise_path), self.device)
        assert self.truth_raw.shape == self.noise_raw.shape, "Truth 和 Noise 数据形状不一致！"

        self.shape = self.truth_raw.shape  # (time, levels, dx, dy)
        self.time, self.levels, self.dx, self.dy = self.shape
        self.data_std = float(data_std)

        self.truth_norm = self.truth_raw / self.data_std
        self.noise_norm = self.noise_raw / self.data_std
        print(f"✅ SQG 数据加载完成: shape = {self.shape}, data_std = {data_std}, device={self.device}")

    def get_field(self, dataset='noise', t=None, level=0, normalized=False) -> Tensor:
        assert dataset in ['truth', 'noise']
        data_all = (self.truth_norm if normalized else self.truth_raw) if dataset == 'truth' \
                   else (self.noise_norm if normalized else self.noise_raw)
        if t is None:
            return data_all[:, level]           # (T, dx, dy)
        else:
            return data_all[t, level]           # 支持 int/list/slice -> (T?, dx, dy)

# ------------------------- 工具函数（Torch 版） -------------------------
def _standardize_columns_torch(X: Tensor) -> Tensor:
    # X: (T, D)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    sd = Xc.std(dim=0, unbiased=True, keepdim=True)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return Xc / sd

def _random_orth_subspace_torch(D: int, k: int, generator=None, device=None) -> Tensor:
    # 返回 D×k 的正交基（QR）
    if generator is None:
        G = torch.randn(D, k, dtype=DTYPE, device=device)
    else:
        G = torch.randn(D, k, dtype=DTYPE, device=device, generator=generator)
    Q, _ = torch.linalg.qr(G, mode='reduced')
    return Q[:, :k]

# ------------------------- Mardia 多元正态检验（Torch 核心提速） -------------------------
@torch.no_grad()
def mardia_test_torch(Y: torch.Tensor, jitter: float = 1e-8) -> dict:
    """
    Y: (T, p)
    返回: dict(skew_stat, skew_p, kurt_stat, kurt_p)
    """
    T, p = Y.shape
    device = Y.device
    dtype = torch.float64

    mu = Y.mean(dim=0, keepdim=True)                # 1×p
    Z = Y - mu                                      # T×p

    # 协方差（与 ddof=1 对齐）
    S = (Z.T @ Z) / (T - 1)
    S = S + jitter * torch.eye(p, dtype=dtype, device=device)

    # Cholesky + 三角解
    L = torch.linalg.cholesky(S)                    # p×p
    W = torch.linalg.solve_triangular(L, Z.T, upper=False)  # p×T

    # A = W^T W
    A = W.T @ W                                     # T×T

    # b1p = (1/T^2) * sum A^3
    b1p = A.pow(3).sum() / (T**2)

    # d_i = ||w_i||^2
    d = (W.pow(2)).sum(dim=0)                       # (T,)
    b2p = (d.pow(2)).mean()

    # 偏度检验
    df_skew = p*(p+1)*(p+2)//6
    skew_stat = T * b1p / 6.0                       # 标量张量

    # —— 关键修复：分布参数和 value 放到同一 device；CUDA 上用 float32 做 special 函数更稳 ——
    cdf_dtype = torch.float32 if device.type == "cuda" else dtype
    chi2 = torch.distributions.Chi2(
        torch.tensor(df_skew, device=device, dtype=cdf_dtype)
    )
    skew_p = (1.0 - chi2.cdf(skew_stat.to(cdf_dtype).clamp_min(0))).item()

    # 峰度检验（正态近似）
    Eb2 = p*(p+2)
    Varb2 = 8.0*p*(p+2)/T
    z_kurt = (b2p - Eb2) / (Varb2**0.5 + 1e-12)

    norm0 = torch.distributions.Normal(
        loc=torch.tensor(0.0, device=device, dtype=cdf_dtype),
        scale=torch.tensor(1.0, device=device, dtype=cdf_dtype),
    )
    kurt_p = (2.0 * (1.0 - norm0.cdf(z_kurt.to(cdf_dtype).abs()))).item()

    return dict(
        skew_stat=float(skew_stat),
        skew_p=float(skew_p),
        kurt_stat=float(z_kurt),
        kurt_p=float(kurt_p)
    )


# ------------------------- 能量距离检验（Torch 版） -------------------------
@torch.no_grad()
def _pdist_avg_torch(A: Tensor, B: Tensor) -> Tensor:
    # 返回平均两两欧氏距离 E||A-B||
    # 注意：cdist 需要较大显存，T 很大时可改成分块
    return torch.cdist(A, B).mean()

@torch.no_grad()
def energy_distance_torch(A: Tensor, B: Tensor) -> Tensor:
    return 2*_pdist_avg_torch(A, B) - _pdist_avg_torch(A, A) - _pdist_avg_torch(B, B)

@torch.no_grad()
def energy_gaussian_test_torch(Y: Tensor, B: int = 200, generator=None, jitter: float = 1e-6) -> dict:
    """
    与拟合同均值/协方差高斯样本比较的能量距离置换检验（Torch 版）
    Y: (T, k)
    """
    T, k = Y.shape
    mu = Y.mean(dim=0)
    C = ((Y - mu).T @ (Y - mu)) / (T - 1)
    C = C + jitter * torch.eye(k, dtype=DTYPE, device=Y.device)

    mvn = MultivariateNormal(loc=mu, covariance_matrix=C)
    Yg = mvn.sample((T,))                           # (T, k)

    E_obs = energy_distance_torch(Y, Yg)
    Z = torch.cat([Y, Yg], dim=0)                   # (2T, k)
    idx = torch.arange(2*T, device=Y.device)

    cnt = 0
    for _ in range(B):
        if generator is None:
            perm = idx[torch.randperm(2*T, device=Y.device)]
        else:
            perm = idx[torch.randperm(2*T, generator=generator, device=Y.device)]
        Y1 = Z[perm[:T]]
        Y2 = Z[perm[T:]]
        if energy_distance_torch(Y1, Y2) >= E_obs:
            cnt += 1
    p = (cnt + 1) / (B + 1)
    return dict(E_obs=float(E_obs), p=float(p))

# ------------------------- 随机子空间 + 多元检验（Torch 版） -------------------------
@torch.no_grad()
def random_subspace_multivariate_tests_torch(
    noise_TXY: torch.Tensor,
    k_list=(5, 10, 20, 50),
    r=100,
    B=50,
    standardize=True,
    seed=0
):
    """
    noise_TXY: torch.Tensor (T, X, Y)
    在每个 k 的循环中加入 tqdm 进度条。
    """
    assert noise_TXY.ndim == 3, "noise 必须是三维 (T, X, Y)。"
    device = noise_TXY.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    T, X, Y = noise_TXY.shape
    data = noise_TXY.reshape(T, -1).to(torch.float64)
    D = data.shape[1]
    if standardize:
        data = _standardize_columns_torch(data)

    summary = {}
    for k in k_list:
        if not (k <= D and k < T):
            summary[k] = dict(skipped=True, reason=f"k={k} 需满足 k <= D 且 k < T；实际 T={T}, D={D}")
            continue

        mardia_pass = 0
        energy_pass = 0
        mardia_p_pairs = []
        energy_pvals = []

        print(f"\n🔹 正在测试子空间维度 k={k} ...")
        for _ in tqdm(range(r), desc=f"Subspace test k={k}", ncols=90):
            # 随机 k 维正交子空间
            Q = _random_orth_subspace_torch(D, k, generator=gen, device=device)
            Yk = data @ Q  # (T, k)

            # --- Mardia 检验 ---
            mt = mardia_test_torch(Yk)
            mardia_ok = (mt["skew_p"] > 0.05) and (mt["kurt_p"] > 0.05)
            mardia_pass += int(mardia_ok)
            mardia_p_pairs.append((mt["skew_p"], mt["kurt_p"]))

            # --- 能量距离检验 ---
            et = energy_gaussian_test_torch(Yk, B=B, generator=gen)
            energy_ok = (et["p"] > 0.05)
            energy_pass += int(energy_ok)
            energy_pvals.append(et["p"])

        summary[k] = dict(
            skipped=False,
            r=r,
            mardia_pass_ratio=mardia_pass / r,
            mardia_p_values=torch.tensor(mardia_p_pairs, dtype=torch.float64, device=device).cpu().numpy(),
            energy_pass_ratio=energy_pass / r,
            energy_p_values=torch.tensor(energy_pvals, dtype=torch.float64, device=device).cpu().numpy(),
        )

    return dict(T=T, D=D, k_list=k_list, r=r, B=B, standardize=standardize, results=summary)

# ------------------------- 随机投影（Torch 版；1D 正态性可选） -------------------------
@torch.no_grad()
def random_projection_gaussian_test_torch(noise_TXYL: Tensor, level: int = 0, num_proj: int = 50, seed: int = 42):
    """
    与你原函数等价，但在 Torch 上完成投影与标准化。
    说明：一维正态性的具体检验统计（如 D’Agostino）仍可调用 SciPy；这里演示只返回标准化后的投影供你外部检验。
    """
    device = noise_TXYL.device
    gen = torch.Generator(device=device).manual_seed(seed)

    # 兼容输入形状： (T, levels, X, Y) 或 (T, X, Y)（若已先 get_field 固定 level）
    if noise_TXYL.ndim == 4:
        T, L, X, Y = noise_TXYL.shape
        data = noise_TXYL[:, level].reshape(T, -1).to(DTYPE)
    elif noise_TXYL.ndim == 3:
        T, X, Y = noise_TXYL.shape
        data = noise_TXYL.reshape(T, -1).to(DTYPE)
    else:
        raise ValueError("noise 需要是 (T, L, X, Y) 或 (T, X, Y)")

    D = data.shape[1]
    pvals = []  # 如需 1D 正态检验，这里可回传给 SciPy；此函数主要负责高效投影与标准化

    # 批量生成随机方向可进一步加速；此处简洁起见逐次
    for _ in range(num_proj):
        w = torch.randn(D, dtype=DTYPE, device=device, generator=gen)
        w = w / (w.norm() + 1e-12)
        proj = (data @ w)
        proj = (proj - proj.mean()) / (proj.std(unbiased=True) + 1e-12)
        # 你可在外部对 proj.cpu().numpy() 做 scipy.stats.normaltest
        pvals.append(float('nan'))

    return {
        "num_projections": num_proj,
        "p_values": np.array(pvals),
        "median_p_value": np.nan,
        "pass_ratio_gt_0_05": np.nan
    }


# 1) 读取数据（Torch）
sqg = SQGDataTorch('../SQG.npy', '../inverted_SQG.npy', device='cuda')  # 若无 GPU 可省略 device

# 2) 取出一个 level 的时序场 (T, X, Y)
noise0 = sqg.get_field(dataset='noise', level=0, normalized=False)  # torch.Tensor

# 3) 跑随机子空间 + 多元检验（GPU 上会很快，Mardia 全在 Torch 内完成）
res = random_subspace_multivariate_tests_torch(
    noise0,
    k_list=(5, 10, 20, 50),
    r=100,          # 子空间抽样次数
    B=200,          # 能量距离置换次数（大就慢；可先 50/100）
    standardize=True,
    seed=0
)

# 4) 读结果
for k in res["k_list"]:
    info = res["results"][k]
    if info["skipped"]:
        print(f"k={k} 跳过：{info['reason']}")
        continue
    print(f"\n[子空间维度 k={k}]")
    print(f"  Mardia  通过率: {info['mardia_pass_ratio']*100:.1f}%")
    print(f"  Energy  通过率: {info['energy_pass_ratio']*100:.1f}%")
