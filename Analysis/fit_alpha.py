import sys
import math
from pathlib import Path
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from natsort import natsorted
from matplotlib.animation import FuncAnimation 
from IPython.display import HTML
from utils import *
from kernels import *
import copy
from tqdm.auto import tqdm 

import re
from torch.utils.data import Dataset, DataLoader
from Condition_model import samplers, dataset, forecasting, utils
from Condition_model.dataset import SQGDataset, SQGLeadTimeDataset, DATA_PATH, DATA_500, DATA_test

invert_data_path = DATA_PATH / 'invert' / 'invert_steps=100'
test_invert_data_path = DATA_PATH / 'invert' / 'test'


class MIMODataset(Dataset):
    """One item = one (window, step) pair.

    x0     : (2, 64, 64) anchor state, model units
    s_norm : scalar s / S, s = 1..S
    z_prev : Z[w, s-1]   (lead s)
    z_next : Z[w, s]     (lead s+1)
    """

    def __init__(self, Zn, traj_ids, anchor_ids, data_path, split='train', std=2660):
        # QUESTION: how should I normalize x0 and Zn?
        self.split = split
        assert len(Zn.shape) == 5, f"I think there is also ensemble {Zn.shape}"
        N, L, C, H, W = Zn.shape

        # it also would store the meta data of the window
        meta = torch.empty((N, 2))
        meta[:, 0] = torch.as_tensor(traj_ids)
        meta[:, 1] = torch.as_tensor(anchor_ids)

        perm = torch.randperm(N)
        # split the dataset
        # initial latent space data
        train_size = round(N * 0.8)
        train_idx = perm[:train_size]
        val_idx = perm[train_size: ]
        Zn = torch.as_tensor(Zn)
        if split == 'train':
            self.idx = train_idx
            self.Zn = Zn[self.idx, ...]
            self.meta = meta[self.idx, ]
            
        elif split == 'val':
            self.idx = val_idx
            self.Zn = Zn[self.idx, ...]
            self.meta = meta[self.idx, ]

        elif split == 'test':
            self.idx = np.arange(N)
            self.Zn = Zn
            self.meta = meta

        
        self.size = len(self.idx)

        # extract x0 from the physcial data
        anchors = np.empty((self.size, C, H, W), dtype=np.float32)        
        for i, idx in enumerate(self.idx):
            traj_id = traj_ids[idx]
            anchor_id = anchor_ids[idx]
            file_path = f'sqg_N64_1hrly_{traj_id}.npy'
            phy_traj =(np.load(data_path / file_path))
            anchors[i, ...] = phy_traj[anchor_id, ...]

 
        # alpha is actually only dependent on x0
        # alpha is different even in one trajectory
        self.anchors = torch.as_tensor(anchors / std)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        What you want to return?
        At first the anchor is the input
        Then when we compute loss, eps_t and eps_{t+1} for each t or for given t
        I think in the training mode that would be for given t randomly
        in the forecasting it would be for all t
        
        '''
        x0 = self.anchors[idx]
        z = self.Zn[idx, ...]
        meta = self.meta[idx, ...]
        return x0, z, meta


class RandomTimeDataset(Dataset):
    """One item = one (window, step) pair.

    x0     : (2, 64, 64) anchor state, model units
    s_norm : scalar s / S, s = 1..S
    z_prev : Z[w, s-1]   (lead s)
    z_next : Z[w, s]     (lead s+1)
    """

    def __init__(self, Zn, traj_ids, anchor_ids, data_path, split='train', random_t=True):
        self.split = split
        self.random_t = random_t
        assert len(Zn.shape) == 5, f"I think there is also ensemble {Zn.shape}"
        N, L, C, H, W = Zn.shape[0] 
        perm = torch.randperm(N)
        train_size = round(N * 0.8)

        train_idx = perm[:train_size]
        val_idx = perm[train_size: ]
        Zn = torch.as_tensor(Zn)

        self.idx = train_idx if split == 'train' else val_idx
        self.Zn = Zn[self.idx, ...]
        self.size = len(self.idx)

        anchors = np.empty((self.size, C, H, W), dtype=np.float32)        
        for i, idx in enumerate(self.idx):
            traj_id = traj_ids[idx]
            anchor_id = anchor_ids[idx]
            file_path = f'sqg_N64_hrly_{traj_id}.npy'
            phy_traj =(np.load(data_path / file_path))
            anchors[i, ...] = phy_traj[anchor_id, ...]
 
        # alpha is actually only dependent on x0
        # alpha is different even in one trajectory
        self.anchors = torch.as_tensor(anchors)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        What you want to return?
        At first the anchor is the input
        Then when we compute loss, eps_t and eps_{t+1} for each t or for given t
        I think in the training mode that would be for given t randomly
        in the forecasting it would be for all t
        
        '''
        if self.random_t:
            # t should be input
            # take the eps_t and eps_{t-1}
            # take the x0 
            t = torch.randint(1, 23, (1, )).item()
            t_norm = t / 23.0
            x0 = self.anchors[idx]
            zt = self.Zn[idx, t, ...]
            zt_sub_1 = self.Zn[idx, t-1, ...]

            return t_norm, x0, zt, zt_sub_1
        else:
            x0 = self.anchor[idx]
            z = self.Zn[idx, ...]
            ts = torch.arange(1, 24) / 23.0
            return x0, z, ts



def alpha_loss(alpha, z):
    """
    MSE loss
    x0: (bs, 2, 64, 64)
    z: (bs, 24, 2, 64, 64)
    """
    resid = z[:, 1:, ...] - alpha[:, :, None, None, None] * z[:, :-1, ...]
    # BTW: what's the difference between the square distance and the negative lld?
    return resid.pow(2).mean()



def train(model, train_loader, val_loader, optimizer, device, epochs=10, patience=10):
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    val_losses = []
    train_losses = []
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # in-place update progress bar
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0
        for x0, z, _ in train_loader:
            x0, z = x0.to(device), z.to(device)
            # it's faster than set as 0
            optimizer.zero_grad(set_to_none=True)            
            alpha = model(x0)
            loss = alpha_loss(alpha, z)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()     
        # Validation
        # the size of each batch could be different in the validation dataloader
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for x0, z, _ in val_loader:
                x0, z = x0.to(device), z.to(device)
                batch_size = x0.shape[0]
                alpha = model(x0)
                loss = alpha_loss(alpha, z)

                val_loss += loss.item() * batch_size
                val_samples += batch_size

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / val_samples
        pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{avg_val:.4f}")

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        # Early stopping

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')  # Save best
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                # Load best model
                model.load_state_dict(torch.load('best_model.pt'))
                break
    
    # Plot the loss 
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    return model, train_losses, val_losses