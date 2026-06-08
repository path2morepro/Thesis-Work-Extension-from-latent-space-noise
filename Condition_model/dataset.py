

NORM_MEAN = 0
NORM_STD  = 2660

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted


THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Convenience paths — pass either as data_dir to SQGPairDataset
DATA_100 = PROJECT_ROOT / "data"           # (100 trajectories)
DATA_500 = PROJECT_ROOT / "data_500traj"     #   (500 trajectories)


class SQGLeadTimeDataset(Dataset):
    """
    Direct lead-time dataset for CEWF-style (non-autoregressive) forecasting.

    Returns (previous, current, time_label[s]) where previous is the single
    initial frame x_0 for each trajectory (conditioning window = 1, no static fields), 
    NOT SURE wether fixed previous frame would affact the performance, but it supports the max lead time for forecasting
    current is the target frame(s), and time_label[s] are normalized lead times.

    Lead-time convention
    --------------------
    - k          : lead in *frames* (integer ≥ 1)
    - lead_hours : k × hours_per_frame  (physical hours ahead)
    - time_label : lead_hours / max_horizon  (dimensionless, in (0, 1])

    max_horizon should be ≥ the maximum lead_hours you will ever train or
    evaluate at, so that all time_labels land in (0, 1].

    Training mode (random_lead_time=True)
    --------------------------------------
    The frame-0 anchor of every trajectory is paired with EVERY lead
    k ∈ {1 … max_lead_frames}; each (trajectory, k) is one deterministic
    dataset item.
    __len__  = n_trajectories × max_lead_frames.
    __getitem__ returns the deterministic pair::

        previous   : torch.FloatTensor (C, H, W)  — x₀ (frame 0), normalised
        current    : torch.FloatTensor (C, H, W)  — x_k (k frames ahead), normalised
        time_label : torch.FloatTensor scalar (0-dim) — lead_hours / max_horizon

    Leads are enumerated as separate indices (not sampled randomly); with
    DataLoader shuffle=True every lead is seen exactly n_trajectories times
    per epoch. The default collate stacks 0-dim tensors into (B,) — exactly
    what the FourierEmbedding map_time expects at training time.
    Requires max_lead_frames ≤ T-1 so the frame-0 anchor can reach every lead.

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
        data_dir=DATA_500,
        std=NORM_STD,
        split='train',
        train_frac=0.8,
        hours_per_frame=1,
        max_lead_frames=99,
        max_horizon=99,
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
            # you can also call this as train mode

            # Training: anchor = first frame (t=0) of every trajectory, paired
            # with EVERY lead k in [1, max_lead_frames]. Each (trajectory, k) is
            # a distinct, deterministic sample  ->  len = n_traj * max_lead_frames.
            # (Leads are enumerated as separate indices, not drawn randomly; with

            # DataLoader shuffle=True they are shuffled across the epoch, so every
            # lead is seen exactly n_traj times per epoch.)
            min_T = min(traj.shape[0] for traj in self.data)
            assert max_lead_frames <= min_T - 1, (
                f"max_lead_frames={max_lead_frames} exceeds T-1={min_T - 1}; "
                f"the frame-0 anchor cannot reach lead {max_lead_frames}."
            )
            self._pairs = [
                (i, k)
                for i in range(len(self.data))
                for k in range(1, max_lead_frames + 1)
            ]
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
            # is this duplicatel logic
            # but it's for eval mode

            rng          = np.random.default_rng(seed)
            traj_indices = rng.integers(0, len(self.data), size=n_init)
            t_indices    = rng.integers(0, max_valid_t,    size=n_init)
            self._eval_points = list(zip(traj_indices.tolist(), t_indices.tolist()))

    def __len__(self):
        if self.random_lead_time:
            return len(self._pairs)
        return len(self._eval_points)

    def __getitem__(self, idx):
        if self.random_lead_time:
            traj_idx, k = self._pairs[idx]
            previous   = self.data[traj_idx][0]          # (C, H, W) — frame-0 IC
            current    = self.data[traj_idx][k]          # (C, H, W) — k frames ahead
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
        


