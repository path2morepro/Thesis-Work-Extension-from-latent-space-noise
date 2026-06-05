"""
Conditional Flow-Matching Training.

Trains a conditional diffusion model to learn p(x_{t+1} | x_t).

What changes from the unconditional model:
- Dataset returns consecutive pairs (x_t, x_{t+1}) plus a lead-time label.
- `SongUNet` uses `in_channels=4` (2 channels for the noisy z_s + 2 channels
  for the conditioning x_t concatenated spatially).
- x_t is passed as `class_labels` — concatenated inside `SongUNet.forward`
  before the encoder; the lead time is passed as `time_labels`.
- Training target is x_{t+1} - z_0 (velocity), same formula as before.
- `label_dropout=0.1` enables classifier-free guidance at inference if needed.

Everything else (loss structure, optimiser) is identical to the unconditional case.

Run:  python train.py
"""

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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from diffusion_networks import SongUNet
from dataset import SQGLeadTimeDataset, DATA_1H, DATA_3H
from loss import flow_matching_loss


# ============================================================================
# Config
# ============================================================================

# ── data ──────────────────────────────────────────────────────────────────
DATA_DIR = DATA_1H   # change to DATA_3H for 3-hour data
DATA_STD = 2660.0    # normalisation std (mean assumed 0)

# ── model ─────────────────────────────────────────────────────────────────
IMG_CHANNELS   = 2    # SQG has 2 vertical levels
IMG_RESOLUTION = 64
FILTERS        = 32
LABEL_DROPOUT  = 0.1

# ── training ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
NUM_EPOCHS   = 300
LR           = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 500    # linear LR warmup steps at the start of training

# ── misc ──────────────────────────────────────────────────────────────────
SAVE_PATH = THIS_DIR.parent / 'models' / 'best_model_conditional.pth'
LOG_PATH  = THIS_DIR.parent / 'models' / 'training_log.csv'
PLOT_PATH = THIS_DIR.parent / 'models' / 'loss_curves.png'


# ============================================================================
# Builders
# ============================================================================

def build_datasets():
    """Build train/val datasets and their DataLoaders."""
    train_dataset = SQGLeadTimeDataset(DATA_DIR, std=DATA_STD, split='train')
    val_dataset   = SQGLeadTimeDataset(DATA_DIR, std=DATA_STD, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f'Pairs — train: {len(train_dataset)}, val: {len(val_dataset)}')
    return train_loader, val_loader


def build_model(device):
    """
    Build the conditional SongUNet.

    The only architectural change from the unconditional model is `in_channels=4`:
    inside `SongUNet.forward`, when `class_labels` is provided it is concatenated
    channel-wise onto the noisy input z_s before the encoder:

        [z_s (2ch)] cat [x_t (2ch)]  ->  [4ch input to encoder]
    """
    model = SongUNet(
        img_resolution     = IMG_RESOLUTION,
        in_channels        = IMG_CHANNELS * 2,  # 2 (noisy z_s) + 2 (x_t conditioning)
        out_channels       = IMG_CHANNELS,      # velocity has same shape as x_{t+1}
        embedding_type     = 'fourier',
        encoder_type       = 'residual',
        decoder_type       = 'standard',
        channel_mult_noise = 2,
        resample_filter    = [1, 3, 3, 1],
        model_channels     = FILTERS,
        channel_mult       = [2, 2, 2],
        attn_resolutions   = [32],
        label_dropout      = LABEL_DROPOUT,
        time_emb           = 1,
    ).to(device)

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    return model


# ============================================================================
# Training
# ============================================================================

def run_epoch(model, loader, device, optimizer=None, warmup=None, desc=''):
    """Run one epoch. Trains when `optimizer` is given, otherwise evaluates."""
    train = optimizer is not None
    model.train() if train else model.eval()

    total = 0.0
    torch.set_grad_enabled(train)
    iterator = tqdm(loader, desc=desc, leave=False) if train else loader
    for x_t, x_t1, lead_time in iterator:
        x_t       = x_t.to(device)
        x_t1      = x_t1.to(device)
        lead_time = lead_time.to(device)

        if train:
            optimizer.zero_grad()
            loss = flow_matching_loss(model, x_t, x_t1, lead_time)
            loss.backward()
            optimizer.step()
            warmup.step()   # warmup steps per batch, not per epoch
        else:
            loss = flow_matching_loss(model, x_t, x_t1, lead_time)

        total += loss.item()

    torch.set_grad_enabled(True)
    return total / len(loader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Data:   {DATA_DIR}')

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_datasets()
    model = build_model(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # cosine decay: learning rate decreases smoothly over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    # linear warmup for the first WARMUP_ITERS steps: avoids large unstable
    # updates at the start of training
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                         end_factor=1.0, total_iters=WARMUP_ITERS)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    with open(LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    # for epoch in range(NUM_EPOCHS):
    #     avg_train = run_epoch(model, train_loader, device, optimizer, warmup,
    #                           desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [train]')
    #     avg_val = run_epoch(model, val_loader, device)

    #     train_losses.append(avg_train)
    #     val_losses.append(avg_val)
    #     scheduler.step()   # cosine decay steps per epoch

    #     if avg_val < best_val_loss:
    #         best_val_loss = avg_val
    #         torch.save(model.state_dict(), SAVE_PATH)
    #         tag = '  <- best'
    #     else:
    #         tag = ''

    #     with open(LOG_PATH, 'a', newline='') as f:
    #         csv.writer(f).writerow([epoch + 1, avg_train, avg_val])

    #     print(f'Epoch {epoch+1:3d}  train={avg_train:.4f}  val={avg_val:.4f}{tag}')
    # # there is no early stoppping
    # print(f'\nTraining done. Best val loss: {best_val_loss:.4f}')

    patience = 20  # stop if val loss doesn't improve for N epochs
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        avg_train = run_epoch(model, train_loader, device, optimizer, warmup,
            desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [train]')
        avg_val = run_epoch(model, val_loader, device)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()   # cosine decay steps per epoch
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), SAVE_PATH)
            tag = '  <- best'
            patience_counter = 0  # reset counter on improvement
        else:
            tag = ''
            patience_counter += 1  # increment on no improvement
        
        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train, avg_val])
        print(f'Epoch {epoch+1:3d}  train={avg_train:.4f}  val={avg_val:.4f}{tag}')
        
        # early stopping check
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1}. No improvement for {patience} epochs.')
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
    plt.title('Conditional flow-matching: p(x_{t+1} | x_t)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f'Loss curves saved to {PLOT_PATH}')


if __name__ == '__main__':
    train()
