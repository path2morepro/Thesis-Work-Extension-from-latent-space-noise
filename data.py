# 那就先给数据读出来再做分析
import os
import numpy as np
from natsort import natsorted

class SQGData:
    def __init__(self, truth_path='SQG.npy', noise_path='inverted_SQG.npy', data_std=2660.0):
        """
        初始化并加载SQG数据。
        参数：
            truth_path : str  真值文件路径
            noise_path : str  噪声或预测文件路径
            data_std   : float 用于标准化的标准差常数
        """
        # 读取原始数据
        self.truth_raw = np.load(truth_path)
        self.noise_raw = np.load(noise_path)
        
        # 检查形状一致性
        assert self.truth_raw.shape == self.noise_raw.shape, "Truth 和 Noise 数据形状不一致！"
        
        # 保存形状信息
        self.shape = self.truth_raw.shape   # (time, levels, dx, dy)
        self.time, self.levels, self.dx, self.dy = self.shape
        
        # 创建标准化版本
        self.data_std = data_std
        self.truth_norm = self.truth_raw / data_std
        self.noise_norm = self.noise_raw / data_std

        print(f"✅ SQG 数据加载完成: shape = {self.shape}, data_std = {data_std}")

    # ------------------------------------------------------------------
    def get_field(self, dataset='noise', t=None, level=0, normalized=False):
        """
        获取某个或多个时间步、层的数据。
        t: 可以是 int 或 slice 或 list
        """
        assert dataset in ['truth', 'noise'], "dataset 必须是 'truth' 或 'noise'"

        if dataset == 'truth':
            data = self.truth_norm if normalized else self.truth_raw
        else:
            data = self.noise_norm if normalized else self.noise_raw

        if t is None:
            return data[:, level]          # 所有时间步 (time, dx, dy)
        else:
            return data[t, level]          # 支持 int、list、slice


class SQGTrajData:
    """
    Load multiple SQG trajectories.
    Each .npy file = 1 trajectory with shape (T, levels, dx, dy)
    Typical: 100 trajectories × 100 steps each.
    """

    def __init__(
        self,
        truth_folder="../sqg_subset",
        noise_folder="../inverted_sqg_subset",
        data_std=2660.0
    ):
        self.truth_folder = truth_folder
        self.noise_folder = noise_folder
        self.data_std = data_std

        # ---------------------------------------------------------
        # 1. Collect files
        # ---------------------------------------------------------
        truth_files = self._collect_files(truth_folder)
        noise_files = self._collect_files(noise_folder)

        assert len(truth_files) == len(noise_files), \
            f"truth 和 noise 文件数不一致: {len(truth_files)} vs {len(noise_files)}"

        self.num_traj = len(truth_files)

        # ---------------------------------------------------------
        # 2. Load each trajectory
        # ---------------------------------------------------------
        truth_list = []
        noise_list = []

        for tf, nf in zip(truth_files, noise_files):
            t_arr = np.load(tf)     # shape (T, 2, N, N)
            n_arr = np.load(nf)

            assert t_arr.shape == n_arr.shape, f"{tf} vs {nf} shape mismatch"

            truth_list.append(t_arr)
            noise_list.append(n_arr)

        # ---------------------------------------------------------
        # 3. Stack -> shape (traj, time, levels, dx, dy)
        # ---------------------------------------------------------
        self.truth_raw = np.stack(truth_list, axis=0)
        self.noise_raw = np.stack(noise_list, axis=0)

        assert self.truth_raw.shape == self.noise_raw.shape

        # Save shape info
        self.shape = self.truth_raw.shape
        self.time_steps = self.shape[1]
        self.levels = self.shape[2]
        self.dx = self.shape[3]
        self.dy = self.shape[4]

        # ---------------------------------------------------------
        # 4. Normalization
        # ---------------------------------------------------------
        self.truth_norm = self.truth_raw / data_std
        self.noise_norm = self.noise_raw / data_std

        print("------------------------------------------------------")
        print(" SQG Trajectory Dataset Loaded")
        print(f"  trajectories = {self.num_traj}")
        print(f"  time steps   = {self.time_steps}")
        print(f"  levels       = {self.levels}")
        print(f"  spatial grid = {self.dx} × {self.dy}")
        print(f"  shape        = {self.shape}")
        print("------------------------------------------------------")

    # ---------------------------------------------------------
    # Helper: collect sorted .npy files
    # ---------------------------------------------------------
    def _collect_files(self, folder):
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
        assert len(files) > 0, f"文件夹 {folder} 没有 npy 文件"

        return natsorted(files)

    # ---------------------------------------------------------
    # Access functions
    # ---------------------------------------------------------
    def get_traj(self, traj_id, dataset="noise", normalized=False):
        """ Return trajectory of shape (T, levels, dx, dy) """
        assert 0 <= traj_id < self.num_traj
        assert dataset in ["truth", "noise"]

        if dataset == "truth":
            return self.truth_norm[traj_id] if normalized else self.truth_raw[traj_id]
        else:
            return self.noise_norm[traj_id] if normalized else self.noise_raw[traj_id]

    def get_frame(self, traj_id, t, dataset="noise", normalized=False):
        """ Return one frame: shape (levels, dx, dy) """
        traj = self.get_traj(traj_id, dataset=dataset, normalized=normalized)
        assert 0 <= t < self.time_steps
        return traj[t]

    def get_all(self, dataset="noise", normalized=False):
        """ Return entire dataset """
        if dataset == "truth":
            return self.truth_norm if normalized else self.truth_raw
        else:
            return self.noise_norm if normalized else self.noise_raw

