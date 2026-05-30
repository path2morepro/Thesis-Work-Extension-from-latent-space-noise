# Normalisation convention used throughout this project:
#   mean = 0,  std = 2660  (physical units of the SQG buoyancy field)
#
# Audit of all data-loading code — all confirmed mean=0, std=2660:
#   generate_inverted_timeseries.py       → SQGDataset(mean=0, std=2660)        ✓
#   trainCondDiffusion.ipynb              → SQGPairDataset(std=2660)             ✓
#   labelDropoutExp.ipynb                 → SQGPairDataset(std=2660)             ✓
#   evalAutoregressive.ipynb              → inline / DATA_STD=2660               ✓
#   easing/visualize_trajectories.ipynb   → inline / DATA_STD=2660               ✓
#   easing/visualize_SQG.ipynb            → inline / data_std=2660               ✓
#   easing/playground.ipynb               → SQGDataset(mean=0, std=2660)         ✓

NORM_MEAN = 0
NORM_STD  = 2660

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Convenience paths — pass either as data_dir to SQGPairDataset
DATA_1H = PROJECT_ROOT / "data"           # sqg_N64_1hrly_*.npy  (100 trajectories)
DATA_3H = PROJECT_ROOT / "3hour_data"     # sqg_N64_3hrly_*.npy  (100 trajectories)


class SQGDataset(Dataset):
    """Single-frame SQG dataset.  Normalisation: mean=0, std=2660."""

    def __init__(self, data_path, mean=NORM_MEAN, std=NORM_STD):
        self.mean = mean
        self.std  = std
        path = Path(data_path)
        if not path.suffix:
            path = path.with_suffix(".npy")
        if not path.is_absolute():
            for base in (Path.cwd(), PROJECT_ROOT, PROJECT_ROOT / "data", THIS_DIR):
                candidate = base / path
                if candidate.exists():
                    path = candidate
                    break
        self.data = torch.tensor(np.load(path).astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx] - self.mean) / self.std


class SQGPairDataset(Dataset):
    """
    Consecutive (x_t, x_{t+1}) pairs for training p(x_{t+1} | x_t).
    Normalisation: mean=0, std=2660.

    Supports both the 1-hour and 3-hour datasets — pass the appropriate
    directory as data_dir:

        from dataset import SQGPairDataset, DATA_1H, DATA_3H

        train_ds = SQGPairDataset(DATA_1H, split='train')   # 1-hour data
        train_ds = SQGPairDataset(DATA_3H, split='train')   # 3-hour data

    Each file is one trajectory of shape (T, C, H, W).  Trajectories are
    split by index (not by time) so every pair within a trajectory stays
    in the same split.

    Args:
        data_dir   : path to folder with sqg_N64_*hrly_*.npy files
        std        : normalisation std; default 2660
        split      : 'train' or 'val'
        train_frac : fraction of trajectories used for training
    """

    def __init__(self, data_dir=DATA_1H, std=NORM_STD, split='train', train_frac=0.8):
        data_dir = Path(data_dir)
        files = natsorted(data_dir.glob('sqg_N64_*hrly_*.npy'))
        assert len(files) > 0, f"No sqg_N64_*hrly_*.npy files found in {data_dir}"

        cut   = int(len(files) * train_frac)
        files = files[:cut] if split == 'train' else files[cut:]

        self.pairs = []
        self.data  = []
        for i, f in enumerate(files):
            arr = torch.tensor(np.load(f).astype(np.float32) / std)
            self.data.append(arr)
            T = arr.shape[0]
            for t in range(T - 1):
                self.pairs.append((i, t))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, t = self.pairs[idx]
        return self.data[i][t], self.data[i][t + 1]
