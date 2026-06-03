

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


class SQGLeadTimeDataset(Dataset):
    """
    Direct lead-time dataset for CEWF-style (non-autoregressive) forecasting.

    Returns (previous, current, time_label[s]) where previous is the single
    initial frame x₀ (conditioning window = 1, no static fields), current is
    the target frame(s), and time_label[s] are normalized lead times.

    Lead-time convention
    --------------------
    - k          : lead in *frames* (integer ≥ 1)
    - lead_hours : k × hours_per_frame  (physical hours ahead)
    - time_label : lead_hours / max_horizon  (dimensionless, in (0, 1])

    max_horizon should be ≥ the maximum lead_hours you will ever train or
    evaluate at, so that all time_labels land in (0, 1].

    Training mode (random_lead_time=True)
    --------------------------------------
    __len__  = number of valid (trajectory, anchor-frame) pairs where
               anchor + max_lead_frames < T (so every k ∈ [1, max_lead_frames]
               is reachable).
    __getitem__ samples k ~ Uniform{1 … max_lead_frames} independently each
    call and returns::

        previous   : torch.FloatTensor (C, H, W)  — x₀, normalised
        current    : torch.FloatTensor (C, H, W)  — x₀₊ₖ, normalised
        time_label : torch.FloatTensor scalar (0-dim) — lead_hours / max_horizon

    The DataLoader default collate stacks 0-dim tensors into (B,) — exactly
    what the FourierEmbedding map_time expects at training time.

    Eval mode (random_lead_time=False)
    -----------------------------------
    __len__  = n_init (seeded, reproducible starting points).
    __getitem__ returns all requested leads from the *same* x₀ at once::

        previous    : torch.FloatTensor (C, H, W)         — x₀
        current     : torch.FloatTensor (n_lead, C, H, W) — targets at each lead
        time_labels : torch.FloatTensor (n_lead,)          — one label per lead

    After DataLoader collation with batch_size B these become
    (B,C,H,W), (B,n_lead,C,H,W), (B,n_lead) — the predict loop accesses
    time_labels[:, lead_idx] or reshapes as needed.

    Args:
        data_dir         : directory with sqg_N64_*hrly_*.npy files
        std              : normalisation std (default NORM_STD = 2660)
        split            : 'train' or 'val'
        train_frac       : fraction of trajectory files used for training
        hours_per_frame  : physical hours between consecutive frames; inferred
                           from the filename ('1hrly' → 1, '3hrly' → 3) if None
        max_lead_frames  : maximum lead k (frames) for training sampling and
                           for anchor validity checks in training mode
        max_horizon      : normalisation constant (hours); set ≥ the largest
                           lead_hours you will use so time_labels ≤ 1
        random_lead_time : True → training mode; False → eval mode
        forecasting_leads: (eval only) list of k values (frames) to return;
                           defaults to list(range(1, max_lead_frames + 1))
        n_init           : (eval only) number of seeded starting points
        seed             : (eval only) RNG seed for reproducible init points
    """

    def __init__(
        self,
        data_dir=DATA_1H,
        std=NORM_STD,
        split='train',
        train_frac=0.8,
        hours_per_frame=None,
        max_lead_frames=10,
        max_horizon=10,
        random_lead_time=True,
        forecasting_leads=None,
        n_init=20,
        seed=42,
    ):
        data_dir = Path(data_dir)
        files = natsorted(data_dir.glob('sqg_N64_*hrly_*.npy'))
        assert len(files) > 0, f"No sqg_N64_*hrly_*.npy files in {data_dir}"

        if hours_per_frame is None:
            fname = Path(files[0]).name
            if '1hrly' in fname:
                hours_per_frame = 1
            elif '3hrly' in fname:
                hours_per_frame = 3
            else:
                raise ValueError(
                    f"Cannot infer hours_per_frame from '{fname}'; pass it explicitly."
                )

        cut   = int(len(files) * train_frac)
        files = files[:cut] if split == 'train' else files[cut:]
        assert len(files) > 0, f"No files for split='{split}' (train_frac={train_frac})"

        # load every trajectory once; __getitem__ indexes into these tensors
        self.data = [
            torch.tensor(np.load(f).astype(np.float32) / std)
            for f in files
        ]

        self.hours_per_frame  = hours_per_frame
        self.max_lead_frames  = max_lead_frames
        self.max_horizon      = max_horizon
        self.random_lead_time = random_lead_time

        if random_lead_time:
            # pre-compute all valid (traj_idx, t) anchors
            # valid iff t + max_lead_frames < T  →  t ∈ [0, T - max_lead_frames - 1]
            self._anchors = []
            for i, traj in enumerate(self.data):
                T = traj.shape[0]
                for t in range(T - max_lead_frames):
                    self._anchors.append((i, t))
        else:
            if forecasting_leads is None:
                forecasting_leads = list(range(1, max_lead_frames + 1))
            self.forecasting_leads = list(forecasting_leads)
            max_k = max(self.forecasting_leads)

            # anchor validity: t + max_k < T for every trajectory
            min_T = min(traj.shape[0] for traj in self.data)
            max_valid_t = min_T - max_k  # t ∈ [0, max_valid_t - 1]
            assert max_valid_t > 0, (
                f"Trajectories (min T={min_T}) too short for max lead {max_k} frames."
            )

            rng          = np.random.default_rng(seed)
            traj_indices = rng.integers(0, len(self.data), size=n_init)
            t_indices    = rng.integers(0, max_valid_t,    size=n_init)
            self._eval_points = list(zip(traj_indices.tolist(), t_indices.tolist()))

    def __len__(self):
        if self.random_lead_time:
            return len(self._anchors)
        return len(self._eval_points)

    def __getitem__(self, idx):
        if self.random_lead_time:
            traj_idx, t = self._anchors[idx]
            k = int(torch.randint(1, self.max_lead_frames + 1, ()).item())
            previous   = self.data[traj_idx][t]          # (C, H, W)
            current    = self.data[traj_idx][t + k]      # (C, H, W)
            time_label = torch.tensor(
                k * self.hours_per_frame / self.max_horizon, dtype=torch.float32
            )                                            # scalar 0-dim tensor → collates to (B,)
            return previous, current, time_label
        else:
            traj_idx, t = self._eval_points[idx]
            previous = self.data[traj_idx][t]            # (C, H, W)
            current  = torch.stack(
                [self.data[traj_idx][t + k] for k in self.forecasting_leads]
            )                                            # (n_lead, C, H, W)
            time_labels = torch.tensor(
                [k * self.hours_per_frame / self.max_horizon
                 for k in self.forecasting_leads],
                dtype=torch.float32,
            )                                            # (n_lead,)
            return previous, current, time_labels
