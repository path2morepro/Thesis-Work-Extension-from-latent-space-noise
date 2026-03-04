import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _resolve_existing_path(path_like, expect_dir=False, add_npy_suffix=False):
    path = Path(path_like)

    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            PROJECT_ROOT / path,
            DATA_ROOT / path,
            THIS_DIR / path,
        ])

    if add_npy_suffix:
        expanded = []
        for candidate in candidates:
            expanded.append(candidate)
            if candidate.suffix != ".npy":
                expanded.append(candidate.with_suffix(".npy"))
        candidates = expanded

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if expect_dir and candidate.is_dir():
            return candidate
        if not expect_dir and candidate.is_file():
            return candidate

    kind = "directory" if expect_dir else "file"
    raise FileNotFoundError(
        f"Could not resolve {kind} from {path_like!r}. Tried: "
        + ", ".join(str(c) for c in unique_candidates)
    )

class SQGDataset(Dataset):
    def __init__(self, data_path, mean, std):
        """
        Args:
            data_path (str): Path to data file.
            mean, std: normalization stats.
        """
        self.mean = mean
        self.std = std
        resolved_path = _resolve_existing_path(data_path, add_npy_suffix=True)
        self.data = np.load(resolved_path)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = (self.data[idx] - self.mean) / self.std
        return x
    
class NoiseDataset(Dataset):
    """
    ConvLSTM-ready latent dataset built from full trajectories.

    Expected raw array shape:
      (num_trajectories, time_steps, levels, height, width)

    Default behavior:
      - split trajectories into 80% train / 10% val / 10% test
      - use 10 past frames to predict the next frame
      - use stride 1 inside each trajectory
      - select one level so samples match a 1-channel ConvLSTM
    """

    def __init__(
        self,
        folder_path=DATA_ROOT / "inverted_sqg_subset",
        split="train",
        history_frames=10,
        forecast_horizon=1,
        stride=1,
        level=0,
        mean=0.0,
        std=1.0,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=0,
        shuffle_trajectories=True,
        dtype=torch.float32,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split}")
        if history_frames <= 0:
            raise ValueError("history_frames must be positive")
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1)")
        if not (0 <= val_ratio < 1):
            raise ValueError("val_ratio must be in [0, 1)")
        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")
        self.folder_path = _resolve_existing_path(folder_path, expect_dir=True)
        raw_files = self._collect_files(self.folder_path)
        self.raw_files = raw_files
        self.file_names = [os.path.basename(f) for f in raw_files]
        raw_list = []
        for f in raw_files:
            f_arr = np.load(f)
            raw_list.append(f_arr)
        raw = np.stack(raw_list, axis=0)
        
        if raw.ndim != 5:
            raise ValueError(
                f"Expected data shape (traj, time, level, H, W), got {raw.shape}"
            )

        self.data = torch.as_tensor(raw, dtype=dtype)
        self.mean = mean
        self.std = std
        self.split = split
        self.history_frames = history_frames
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.level = level
        self.dtype = dtype

        self.num_traj, self.time_steps, self.levels, self.height, self.width = self.data.shape
        if not (0 <= level < self.levels):
            raise ValueError(f"level must be in [0, {self.levels - 1}], got {level}")

        self.window_size = history_frames + forecast_horizon
        if self.time_steps < self.window_size:
            raise ValueError(
                f"time_steps={self.time_steps} is too short for window_size={self.window_size}"
            )

        split_indices = self._split_trajectories(
            num_traj=self.num_traj,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            shuffle=shuffle_trajectories,
        )
        self.trajectory_ids = split_indices[split]
        self.sample_index = self._build_sample_index()

        print(
            f"NoiseDataset[{self.split}] loaded: "
            f"{len(self.trajectory_ids)} trajectories, "
            f"{len(self.sample_index)} samples, "
            f"history={self.history_frames}, horizon={self.forecast_horizon}, "
            f"level={self.level}"
        )

    def _split_trajectories(self, num_traj, train_ratio, val_ratio, seed, shuffle):
        indices = np.arange(num_traj)
        if shuffle:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(indices)

        n_train = int(num_traj * train_ratio)
        n_val = int(num_traj * val_ratio)
        n_test = num_traj - n_train - n_val
        if min(n_train, n_val, n_test) <= 0:
            raise ValueError(
                f"Invalid split sizes for num_traj={num_traj}: "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        return {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

    def _build_sample_index(self):
        max_start = self.time_steps - self.window_size
        sample_index = []
        for traj_id in self.trajectory_ids:
            for start in range(0, max_start + 1, self.stride):
                sample_index.append((int(traj_id), start))
        return sample_index

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        traj_id, start = self.sample_index[idx]
        end = start + self.history_frames
        target_idx = end + self.forecast_horizon - 1

        x = self.data[traj_id, start:end, self.level:self.level + 1]
        y = self.data[traj_id, target_idx, self.level:self.level + 1]

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return x, y

    def get_sample_info(self, idx):
        traj_id, start = self.sample_index[idx]
        end = start + self.history_frames
        target_idx = end + self.forecast_horizon - 1
        return {
            "dataset_idx": int(idx),
            "traj_id": int(traj_id),
            "start_idx": int(start),
            "input_end_idx": int(end - 1),
            "target_idx": int(target_idx),
            "source_file": self.raw_files[traj_id],
            "source_file_name": self.file_names[traj_id],
            "noise_file": self.raw_files[traj_id],
            "noise_file_name": self.file_names[traj_id],
            "level": int(self.level),
        }
    
    def _collect_files(self, folder):
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]
        assert len(files) > 0, f"No files in floder {folder}"

        return natsorted(files)
