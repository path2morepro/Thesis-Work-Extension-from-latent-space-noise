NORM_STD  = 2660

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted


THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATA_PATH = PROJECT_ROOT / "data"

# Convenience paths — pass either as data_dir to SQGPairDataset
DATA_100 = PROJECT_ROOT / "data" / "data_100traj"          # (100 trajectories)
DATA_500 = PROJECT_ROOT /"data" / "data_500traj"     #   (500 trajectories)


class SQGLeadTimeDataset(Dataset):
    """
    Args:
        data_dir         : directory containing sqg_N64_*hrly_*.npy files
        std              : normalisation std  (default NORM_STD = 2660)
        split            : 'train' or 'val'
        train_frac       : fraction of trajectory files used for training
        max_lead         : maximum lead k (in frames) for predict target, normally set as 24
        random_lead_time : True → training mode; False → eval mode
        n_init           : (eval only) it is for restrict the size of eval dataset
        seed             : (eval only) RNG seed for reproducible anchor selection
    """

    def __init__(
        self,
        data_dir=DATA_100,
        std=NORM_STD,
        split='train',
        train_frac=0.8,
        max_lead=24,
        random_lead_time=True,
        eval_traj_num=20,
        seed=42,
        max_frames = 100,
        size = 1.0
    ):
        data_dir = Path(data_dir)
        files = natsorted(data_dir.glob('sqg_N64_*hrly_*.npy'))
        assert len(files) > 0, f"No sqg_N64_*hrly_*.npy files in {data_dir}"

        cut   = int(len(files)* size * train_frac)
        files = files[:cut] if split == 'train' else files[cut:]
        assert len(files) > 0, f"No files for split='{split}' (train_frac={train_frac})"

        # Load every trajectory once; __getitem__ indexes into these tensors
        self.data = [
            torch.tensor(np.load(f).astype(np.float32) / std)
            for f in files
        ]
        
        self.forecasting_leads = list(range(max_lead+1)) # 0...24
        self.max_frames = max_frames
        self.max_lead  = max_lead
        self.random_lead_time = random_lead_time
        # forecasting_leads should be fixed
        # determined by hours per lead and max lead time
        # does forecasting leads equal to k? if so we don't need 2 names
        # k is iterative variables, forecasting_leads should be a vector
        
        n_valid_anchors = max_frames - max_lead  # number of valid t values
        if random_lead_time:
            # Training mode.
            # The anchor can be ANY frame t in [0, (number of frames per traj - 1) - max_lead],
            # For a trajectory of length T:
            #   valid anchors: t ∈ {0, 1, …, (number of frames per traj - 1)- max_lead}
            #   valid leads:   k ∈ {0, …, max_lead}
            #
            # all the trajectory has the same number of frames 100
            
            assert n_valid_anchors > 0, (
                f"max_lead_frames={max_lead} too large for trajectory length "
                f"T={max_frames}; need T > max_lead_frames."
            )

            self._pairs = [
                (traj_idx, t, k)
                for traj_idx in range(len(self.data))
                for t in range(n_valid_anchors)           # anchor: any valid frame
                for k in self.forecasting_leads   # train on [0, 24] hour, 25 step in total   
            ]
            # what's the pairs inside?
            

        else:
            # Eval mode.
            rng = np.random.default_rng(seed)
            traj_indices = rng.integers(0, len(self.data), size=eval_traj_num)
            # Anchor t drawn from all valid positions — not just frame 0
            t_indices    = rng.integers(0, n_valid_anchors, size=eval_traj_num)
            self._eval_points = list(zip(traj_indices.tolist(), t_indices.tolist()))

    def __len__(self):
        if self.random_lead_time:
            return len(self._pairs)
        return len(self._eval_points)

    def __getitem__(self, idx):
        # out of curiosity: how we could pass idx into this function?
        # by dataloader? 
        # and how we could keep idx random?
        if self.random_lead_time:
            # Training: return one (anchor, target) pair with its lead label.
            # `initial` is traj[t] (the anchor, which can be any valid frame),
            # `target` is traj[t + k] (k frames later).
            traj_idx, t, k = self._pairs[idx]
            initial   = self.data[traj_idx][t]           # (C, H, W) — anchor frame
            target    = self.data[traj_idx][t + k]       # (C, H, W) — target frame
            time_label = torch.tensor(
                k  / self.max_lead, dtype=torch.float32
            )  # scalar 0-dim tensor → collates to (B,)
            return initial, target, time_label

        else:
            # Eval mode: return all requested leads from the same anchor at once.
            # `initial` is traj[t], `target[j]` is traj[t + forecasting_leads[j]].
            # `time_labels[j]` is the normalised lead for lead j, accounting for
            traj_idx, t = self._eval_points[idx]
            initial = self.data[traj_idx][t]             # (C, H, W) — anchor frame
            # forecasting_leads actually includes 0
            # to avoid the situation that in the eval mode the dataset query lead time = 0 so initial=target
            # the predict performance could be affected
            # so exclude element 1 in eval mode
            target  = torch.stack(
                [self.data[traj_idx][t + k] for k in self.forecasting_leads[1:]] 
            )                                             # (max_lead-1, C, H, W)
            time_labels = torch.tensor(
                [k / self.max_lead for k in self.forecasting_leads[1:]],
                dtype=torch.float32,
            )                                        # (max_lead,) forecast [1, 24] hour

            return initial, target, time_labels, torch.tensor(traj_idx, dtype=torch.long), torch.tensor(t, dtype=torch.long)
        


class SQGDataset(Dataset):
    def __init__(self, data_path, mean=0, std=2660):
        """
        Args:
            data_path (str): Path to data file.
            mean, std: normalization stats.
        """
        self.mean = mean
        self.std = std
        files = natsorted(data_path.glob('*.npy'))
        self.data = [
            np.load(f).astype(np.float32) / std
            for f in files
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = (self.data[idx] - self.mean) / self.std
        return x


