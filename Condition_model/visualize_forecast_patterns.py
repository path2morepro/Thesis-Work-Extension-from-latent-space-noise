"""
Visualize autoregressive forecasting patterns across multiple cases.
Run this in your notebook environment or as: python visualize_forecast_patterns.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path('SQG')))
from diffusion_networks import SongUNet
from natsort import natsorted

# ── Config ────────────────────────────────────────────────────────────────
MODEL_PATH = Path('../models/results/best_model_conditional.pth')
DATA_STD = 2660.0
TRAIN_FRAC = 0.8
IMG_CHANNELS = 2
IMG_RESOLUTION = 64
FILTERS = 32
LABEL_DROPOUT = 0.1
N_ENSEMBLE = 10
ROLLOUT_STEPS = 20
ODE_STEPS = 100
N_INIT = 20
N_CASES = 6  # number of cases to visualize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ── Load Data ─────────────────────────────────────────────────────────────
hour = 3
DATA_DIR = Path('../3hour_data/')
files = natsorted(DATA_DIR.glob(f'sqg_N64_{hour}hrly_*.npy'))
assert len(files) > 0, f"No files found in {DATA_DIR}"

cut = int(len(files) * TRAIN_FRAC)
val_files = files[cut:]

val_trajs = []
for f in val_files:
    arr = torch.tensor(np.load(f).astype(np.float32) / DATA_STD)
    val_trajs.append(arr)

T_per_traj = val_trajs[0].shape[0]
max_t = T_per_traj - ROLLOUT_STEPS - 1
rng = np.random.default_rng(42)
traj_indices = rng.integers(0, len(val_trajs), size=N_INIT)
time_indices = rng.integers(0, max_t, size=N_INIT)
init_points = list(zip(traj_indices.tolist(), time_indices.tolist()))

print(f'Loaded {len(val_trajs)} validation trajectories')

# ── Load Model ────────────────────────────────────────────────────────────
model = SongUNet(
    img_resolution=IMG_RESOLUTION,
    in_channels=IMG_CHANNELS * 2,
    out_channels=IMG_CHANNELS,
    embedding_type='fourier',
    encoder_type='residual',
    decoder_type='standard',
    channel_mult_noise=2,
    resample_filter=[1, 3, 3, 1],
    model_channels=FILTERS,
    channel_mult=[2, 2, 2],
    attn_resolutions=[32],
    label_dropout=LABEL_DROPOUT,
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.eval()
print('Model loaded')

# ── Sampling Functions ────────────────────────────────────────────────────
@torch.no_grad()
def sample_one_step(model, x_t, steps=100):
    B = x_t.shape[0]
    dt = 1.0 / steps
    z = torch.randn_like(x_t)
    for i in range(steps):
        s = torch.full((B,), i * dt, device=x_t.device)
        b = model(z, s, class_labels=x_t)
        z = z + b * dt
    return z

@torch.no_grad()
def autoregressive_ensemble_rollout(model, x0, rollout_steps, n_ensemble, ode_steps):
    x_current = x0.unsqueeze(0).expand(n_ensemble, -1, -1, -1).clone().to(device)
    preds = []
    for _ in range(rollout_steps):
        x_next = sample_one_step(model, x_current, steps=ode_steps)
        preds.append(x_next.cpu().numpy())
        x_current = x_next
    return np.stack(preds, axis=0)

# ── Visualization ─────────────────────────────────────────────────────────
print(f'\nVisualizing {N_CASES} forecast cases...')
n_cases = N_CASES
show_steps = [1, 5, 10, 15, 20]
show_steps = [s for s in show_steps if s <= ROLLOUT_STEPS]

fig, axes = plt.subplots(n_cases, len(show_steps) * 2, figsize=(4 * len(show_steps), 3 * n_cases))

for case_idx in tqdm(range(n_cases), desc='Processing cases'):
    ti, t = init_points[case_idx]
    x0 = val_trajs[ti][t]
    truth = val_trajs[ti][t + 1 : t + 1 + ROLLOUT_STEPS].numpy()

    preds = autoregressive_ensemble_rollout(model, x0, ROLLOUT_STEPS, N_ENSEMBLE, ODE_STEPS)
    preds_mean = preds.mean(axis=1)
    preds_std = preds.std(axis=1)

    for col, step in enumerate(show_steps):
        step_idx = step - 1

        pred_val = preds_mean[step_idx, 0]
        truth_val = truth[step_idx, 0]
        vmax = max(np.abs(pred_val).max(), np.abs(truth_val).max())
        std_val = preds_std[step_idx, 0]

        # Prediction with uncertainty
        ax_pred = axes[case_idx, col * 2]
        ax_pred.imshow(pred_val, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax_pred.contour(std_val, levels=3, colors='black', alpha=0.3, linewidths=0.5)
        ax_pred.set_title(f'Pred (L={step})\nstd∈[{std_val.min():.3f},{std_val.max():.3f}]', fontsize=8)
        ax_pred.axis('off')

        # Ground truth
        ax_truth = axes[case_idx, col * 2 + 1]
        ax_truth.imshow(truth_val, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        rmse_val = np.sqrt(((pred_val - truth_val) ** 2).mean())
        ax_truth.set_title(f'Truth (L={step})\nRMSE={rmse_val:.4f}', fontsize=8)
        ax_truth.axis('off')

    # Label cases
    axes[case_idx, 0].text(-0.5, 0.5, f'Case {case_idx}: traj={ti}, t={t}',
                           transform=axes[case_idx, 0].transAxes,
                           fontsize=9, ha='right', va='center', fontweight='bold')

plt.suptitle(f'Autoregressive Forecast Patterns — {n_cases} cases (left=pred, right=truth)',
             fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig('forecast_patterns.png', dpi=150, bbox_inches='tight')
print('✓ Saved forecast_patterns.png')
plt.show()

# ── Summary Statistics ────────────────────────────────────────────────────
print(f"\n{'Case':<6} {'Traj':<6} {'t':<6}", end='')
for step in show_steps:
    print(f"{'L' + str(step):<8}", end='')
print()
print("-" * (6 + 6 + 6 + 8 * len(show_steps)))

for case_idx in range(n_cases):
    ti, t = init_points[case_idx]
    x0 = val_trajs[ti][t]
    truth = val_trajs[ti][t + 1 : t + 1 + ROLLOUT_STEPS].numpy()

    preds = autoregressive_ensemble_rollout(model, x0, ROLLOUT_STEPS, N_ENSEMBLE, ODE_STEPS)
    preds_mean = preds.mean(axis=1)
    preds_std = preds.std(axis=1)

    print(f'{case_idx:<6} {ti:<6} {t:<6}', end='')
    for step in show_steps:
        step_idx = step - 1
        rmse = np.sqrt(((preds_mean[step_idx] - truth[step_idx]) ** 2).mean())
        spread = preds_std[step_idx].mean()
        print(f'{rmse:.2f}/{spread:.2f}', end='  ')
    print()

print("\n📊 Format: RMSE/ensemble_spread for each lead time")
print("\n✓ Analysis complete!")
