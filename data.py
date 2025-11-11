# 那就先给数据读出来再做分析

import numpy as np

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




