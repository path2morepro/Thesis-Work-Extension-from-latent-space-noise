import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # headless: save figures instead of showing them
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .diffusion_networks import SongUNet
from .dataset import SQGLeadTimeDataset, DATA_100, DATA_500
from .loss import flow_matching_loss


"""
Conditional Flow-Matching Training.

Trains p(x_{t+k} | x_t, lead=k) via linear stochastic interpolant.

Key changes from original:
- Dataset now samples anchor t from any valid frame (not just frame 0).
- max_lead=24, leads k in {0,1,...,24}; k=0 included for embedding coverage.
- LABEL_DROPOUT=0.0 (CFG not used in this project).
- FILTERS=32, DATA_100 for fast iteration; swap to DATA_500/FILTERS=64 for real runs.
"""


# ============================================================================
# Config
# ============================================================================

# ── data ──────────────────────────────────────────────────────────────────
DATA_DIR       = DATA_100    # swap to DATA_500 for full training run
DATA_STD       = 2660.0
MAX_LEAD       = 24          # direct leads: k in {0, 1, ..., 24}
MAX_FRAMES     = 100         # fixed by sqg_generate.py

# ── model ─────────────────────────────────────────────────────────────────
IMG_CHANNELS   = 2
IMG_RESOLUTION = 64
FILTERS        = 32          # small for quick sanity runs; use 64 for real training
LABEL_DROPOUT  = 0.0         # CFG not used — must be 0.0

# ── training ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
NUM_EPOCHS   = 500
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 500

# ── misc ──────────────────────────────────────────────────────────────────
# Path management: 
# all the model path should import from here
ROOT = Path(__file__).resolve().parent.parent
SAVE_PATH = ROOT / 'models' / 'randomAnchor_n_24pred_formal.pth'
LOG_PATH  = ROOT / 'models' / 'randomAnchor_n_24pred_formal_training_log.csv'
PLOT_PATH = ROOT / 'models' / 'randomAnchor_n_24pred_formal_loss_curves.png'


# ============================================================================
# Builders
# ============================================================================

def build_datasets():
    """
    Build train/val datasets.

    Training mode returns (initial, target, time_label):
        initial    : (C, H, W)  — anchor frame traj[t], t drawn from any valid position
        target     : (C, H, W)  — traj[t + k], k in {0, 1, ..., MAX_LEAD}
        time_label : scalar     — k / MAX_LEAD in [0, 1]

    k=0 pairs (target == initial, label=0.0) are included to anchor the
    time embedding at zero; they contribute near-zero loss.
    """
    common = dict(
        std            = DATA_STD,
        max_lead       = MAX_LEAD,
        max_frames     = MAX_FRAMES,
        random_lead_time = True,    # training mode
        size            = 1
    )

    train_dataset = SQGLeadTimeDataset(DATA_DIR, split='train', **common)
    val_dataset   = SQGLeadTimeDataset(DATA_DIR, split='val',   **common)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f'Leads     : {train_dataset.forecasting_leads}')
    print(f'Pairs     — train: {len(train_dataset):,}, val: {len(val_dataset):,}')
    return train_loader, val_loader


def build_model(device):
    """
    SongUNet with in_channels=4: [z_s (2ch)] cat [x_t (2ch)].
    time_emb=1 enables the lead-time positional embedding (map_time).
    label_dropout=0.0: no CFG dropout.
    """
    model = SongUNet(
        img_resolution     = IMG_RESOLUTION,
        in_channels        = IMG_CHANNELS * 2,  # 2 (z_s) + 2 (x_t conditioning)
        out_channels       = IMG_CHANNELS,
        embedding_type     = 'positional',
        encoder_type       = 'residual',
        decoder_type       = 'standard',
        channel_mult_noise = 2,
        resample_filter    = [1, 3, 3, 1],
        model_channels     = FILTERS,
        channel_mult       = [2, 2, 2],
        attn_resolutions   = [32],
        label_dropout      = LABEL_DROPOUT,     # 0.0 — no CFG
        time_emb           = 1,                 # must be 1 for lead-time conditioning
    ).to(device)

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    return model


# ============================================================================
# Training
# ============================================================================

def run_epoch(model, loader, device, optimizer=None, warmup=None, desc=''):
    """One epoch. Trains when optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train() if train else model.eval()

    total = 0.0
    torch.set_grad_enabled(train)
    iterator = tqdm(loader, desc=desc, leave=False) if train else loader

    for initial, target, lead_time in iterator:
        # initial  : (B, C, H, W) — conditioning anchor
        # target   : (B, C, H, W) — generation target
        # lead_time: (B,)         — normalised lead in [0, 1]
        initial   = initial.to(device)
        target    = target.to(device)
        lead_time = lead_time.to(device)

        if train:
            optimizer.zero_grad()
            loss = flow_matching_loss(model, initial, target, lead_time)
            loss.backward()
            optimizer.step()
            warmup.step()
        else:
            loss = flow_matching_loss(model, initial, target, lead_time)

        total += loss.item()

    torch.set_grad_enabled(True)
    return total / len(loader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Data   : {DATA_DIR}')

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_datasets()
    model = build_model(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    warmup    = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_ITERS
    )

    train_losses, val_losses = [], []
    best_val_loss  = float('inf')
    patience       = 20
    patience_counter = 0

    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(NUM_EPOCHS):
        avg_train = run_epoch(
            model, train_loader, device, optimizer, warmup,
            desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [train]',
        )
        avg_val = run_epoch(model, val_loader, device)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            tag = '  <- best'
            patience_counter = 0
        else:
            tag = ''
            patience_counter += 1

        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train, avg_val])
        print(f'Epoch {epoch+1:3d}  train={avg_train:.4f}  val={avg_val:.4f}{tag}')

        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}.')
            break

    print(f'\nTraining done. Best val loss: {best_val_loss:.4f}')
    plot_losses(train_losses, val_losses)
    return train_losses, val_losses


# ============================================================================
# Loss curves
# ============================================================================

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses,   label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Flow-matching loss')
    plt.title(f'p(x_{{t+k}} | x_t, lead=k)  —  max_lead={MAX_LEAD}, filters={FILTERS}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f'Loss curves saved to {PLOT_PATH}')


if __name__ == '__main__':
    train()