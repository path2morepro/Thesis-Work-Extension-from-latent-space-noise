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
from .dataset import SQGLeadTimeDataset, DATA_TRAIN, DATA_VAL
from .loss import flow_matching_loss


"""
Conditional Flow-Matching Training.

Trains p(x_{t+k} | x_t, lead=k) via linear stochastic interpolant.

Key changes from original:
- Dataset now samples anchor t from any valid frame (not just frame 0).
- max_lead=24, leads 1..24; lead 0 dropped — the forecast path never queries it.
- Train/val are separate DIRECTORIES (data/train, data/val); no in-folder split.
- Anchors are redrawn every epoch via dataset.set_epoch(epoch).
- LABEL_DROPOUT=0.0 (CFG not used in this project).
"""


# ============================================================================
# Config
# ============================================================================

# ── data ──────────────────────────────────────────────────────────────────
DATA_STD       = 2660.0
MAX_LEAD       = 24          # direct leads: 1..24 (lead 0 is not trained)

# Anchors drawn per trajectory per epoch. Epoch size is
#   n_files * anchors_per_file * MAX_LEAD
# so these two numbers set how long an epoch is:
#   train  1200 files * 4 * 24 = 115,200 pairs  -> 3,600 batches @ 32
#   val     150 files * 1 * 24 =   3,600 pairs  ->   112 batches @ 32
# Trajectory length is read per file by the dataset, so no MAX_FRAMES here.
#
# These are an EPOCH-SIZE knob, not a data knob. Anchors are redrawn every
# epoch by set_epoch, so the pool they are drawn from is fixed and complete:
# 600 files of 100 frames (76 anchors) + 600 of 101 (77) = 91,800 anchors,
# x 24 leads = 2,203,200 distinct pairs. Raising TRAIN_ANCHORS at a fixed
# NUM_EPOCHS therefore buys nothing but more compute -- it is the same lever
# as raising NUM_EPOCHS, and the short epoch is preferable because the LR
# step, the val pass and the best-checkpoint test all happen per epoch.
TRAIN_ANCHORS  = 4
VAL_ANCHORS    = 1

# ── model ─────────────────────────────────────────────────────────────────
IMG_CHANNELS   = 2
IMG_RESOLUTION = 64
FILTERS        = 32        
LABEL_DROPOUT  = 0.0         # CFG not used — must be 0.0

# ── training ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 32           # 3432 MiB peak on a 6 GB card; see USE_COMPILE
NUM_EPOCHS   = 300
LR           = 7e-4         # 1e-3 was tuned at batch 64; sqrt(32/64) = 0.71
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 500           # in BATCHES, not epochs
COSINE_EPOCHS = NUM_EPOCHS
USE_COMPILE = True
USE_AMP     = True          # Automatic Mixed Precision

# ── misc ──────────────────────────────────────────────────────────────────
# Path management: 
# all the model path should import from here
ROOT = Path(__file__).resolve().parent.parent

SAVE_PATH = ROOT / 'models' / 'randomAnchor_n_24pred_formal_v2.pth'
LOG_PATH  = ROOT / 'models' / 'randomAnchor_n_24pred_formal_v2_training_log.csv'
PLOT_PATH = ROOT / 'models' / 'randomAnchor_n_24pred_formal_v2_loss_curves.png'
OVERWRITE = False

# Used to resume the training from interruption
LAST_PATH = ROOT / 'models' / 'randomAnchor_n_24pred_formal_v2_last.pth'


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
    train_dataset = SQGLeadTimeDataset(
        DATA_TRAIN, anchors_per_file=TRAIN_ANCHORS,
        max_lead=MAX_LEAD, std=DATA_STD,
    )
    val_dataset = SQGLeadTimeDataset(
        DATA_VAL, anchors_per_file=VAL_ANCHORS,
        max_lead=MAX_LEAD, std=DATA_STD,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    print(f'Leads     : {train_dataset.leads}')
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

def run_epoch(model, loader, device, optimizer=None, warmup=None,
              scaler=None, desc=''):
    """One epoch. Trains when optimizer is given, otherwise evaluates.

    Only the training path runs under autocast. Validation stays in fp32 so the
    best-checkpoint comparison across epochs is not made on fp16-noisy losses;
    it is 3,600 pairs, so the extra cost is a rounding error. Under
    torch.compile this means two graphs get compiled (one per dtype/mode), i.e.
    roughly 200 s of warmup once at startup rather than 100 s.
    """
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
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                loss = flow_matching_loss(model, initial, target, lead_time)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            warmup.step()
        else:
            loss = flow_matching_loss(model, initial, target, lead_time)

        total += loss.item()

    torch.set_grad_enabled(True)
    return total / len(loader)


def train(resume=False):
    # Resuming is *supposed* to find SAVE_PATH there, so the collision guard
    # only applies to a fresh run.
    if resume and not LAST_PATH.exists():
        raise FileNotFoundError(
            f'--resume needs {LAST_PATH}, which does not exist. Drop --resume '
            f'to start a fresh run.'
        )
    # Checked before anything expensive, so a name collision fails in a second
    # rather than after the compile warmup or the first epoch.
    if not resume and SAVE_PATH.exists() and not OVERWRITE:
        raise FileExistsError(
            f'{SAVE_PATH} already exists and would be overwritten at the end '
            f'of training. Move it aside, point SAVE_PATH somewhere new, or '
            f'set OVERWRITE = True.'
        )

    # Shapes are fixed by drop_last, so autotuning pays off once and then wins.
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    print(f'Data   : {DATA_TRAIN}  |  {DATA_VAL}')

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_datasets()

    # `raw_model` must stay in scope. torch.compile returns a wrapper whose
    # state_dict prefixes every key with '_orig_mod.', and samplers.py loads
    # strictly -- saving the wrapper would write a checkpoint that nothing
    # downstream (arci, generate_invert_data, round_trip) can read, and the
    # failure would only surface after the whole run finished.
    raw_model = build_model(device)
    model     = torch.compile(raw_model) if USE_COMPILE else raw_model
    if USE_COMPILE:
        print('torch.compile enabled — expect ~200 s of warmup before epoch 1')

    optimizer = optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=COSINE_EPOCHS)
    warmup    = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_ITERS
    )

    train_losses, val_losses = [], []
    best_val_loss  = float('inf')
    start_epoch    = 0

    if resume:
        # weights_only=False because this carries optimizer/scheduler state and
        # the loss history, not just tensors. It is a file this script wrote.
        ck = torch.load(LAST_PATH, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        warmup.load_state_dict(ck['warmup'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch      = ck['epoch']
        best_val_loss    = ck['best_val_loss']
        train_losses     = ck['train_losses']
        val_losses       = ck['val_losses']
        print(f'Resumed from {LAST_PATH.name} at epoch {start_epoch}/{NUM_EPOCHS}  '
              f'best_val={best_val_loss:.4f}  lr={optimizer.param_groups[0]["lr"]:.2e}')
        if start_epoch >= NUM_EPOCHS:
            print('Nothing to do: the saved run already reached NUM_EPOCHS.')
            return train_losses, val_losses
    else:
        # Truncate only for a fresh run; a resume appends to the same history.
        with open(LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Redraw this epoch's anchors. Must happen in this (parent) process
        # before the loaders are iterated, or the workers inherit the old index.
        train_loader.dataset.set_epoch(epoch)
        val_loader.dataset.set_epoch(epoch)

        avg_train = run_epoch(
            model, train_loader, device, optimizer, warmup, scaler,
            desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [train]',
        )
        avg_val = run_epoch(model, val_loader, device)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            tag = '  <- best'
        else:
            tag = ''

        with open(LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, avg_train, avg_val])

        # Written via a temporary file and renamed, because rename is atomic on
        # the same filesystem: a crash *during* the write leaves the previous
        # epoch's state intact instead of a half-written file that cannot load.
        tmp = LAST_PATH.with_suffix('.tmp')
        torch.save({
            'epoch'            : epoch + 1,   # epochs COMPLETED, so resume starts here
            'model'            : raw_model.state_dict(),
            'optimizer'        : optimizer.state_dict(),
            'scheduler'        : scheduler.state_dict(),
            'warmup'           : warmup.state_dict(),
            'scaler'           : scaler.state_dict(),
            'best_val_loss'    : best_val_loss,
            'train_losses'     : train_losses,
            'val_losses'       : val_losses,
        }, tmp)
        tmp.replace(LAST_PATH)

        print(f'Epoch {epoch+1:3d}  train={avg_train:.4f}  val={avg_val:.4f}{tag}')

    torch.save(raw_model.state_dict(), SAVE_PATH)
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
    import argparse

    p = argparse.ArgumentParser(description='Conditional flow-matching training.')
    p.add_argument('--resume', action='store_true',
                   help=f'continue from {LAST_PATH.name} (model, optimizer, cosine '
                        f'position, warmup counter, loss history and best-val state)')
    train(resume=p.parse_args().resume)