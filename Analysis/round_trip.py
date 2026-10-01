"""Test 3 — does the mapping's bias accumulate, and does it matter?

Sigma != I means the inversion dumps model error into eps. Push the latents back through
the forward ODE and compare to the truth they came from.

Nothing is chained: each lead is its own map Phi(x0, tau) from the SAME anchor, so lead
24 is not built from lead 23. A rising curve is therefore the integrator accumulating
error along one ODE solve, NOT error feeding back on itself.

Two comparisons on one axis:
  * every step count, PAIRED on the same windows — isolates the integrator
  * a real ens=20 forecast — the error the mapping bias would have to sit inside

Run:  python round_trip.py [--windows 10] [--no-forecast]
"""
import argparse
import re

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
from utils import (INVERT_DIRS, DATA_PATH, L, leads, cmap,
                   shared_windows, save_fig)
from Condition_model.samplers import CondSampler
from Condition_model.train import SAVE_PATH
from Condition_model.dataset import SQGLeadTimeDataset, DATA_500, NORM_STD
from Condition_model.utils import compute_rmse, compute_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = torch.tensor([k / L for k in range(1, L + 1)], dtype=torch.float32)


def roundtrip(paths, steps, ds, save_dir=None):
    """Push saved latents back through the forward ODE and pair them with the truth.

    Args:
        paths    : list of Path to invert_traj{T}initial{A}.npy files
        steps    : Euler steps for the FORWARD integration. MUST match the directory the
                   latents came from, or the error grows with lead for the wrong reason.
        ds       : SQGLeadTimeDataset(split='val') — supplies the anchor and the truth
        save_dir : optional directory to write the (pred, truth) pair per window

    Returns:
        (pred, truth) : both (n_files, L, C, H, W) float32, in PHYSICAL units. ds.data
        was divided by NORM_STD at load, so both are multiplied back by it.
    """
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    smp, Pr, Tr = CondSampler(model_path=SAVE_PATH, device=DEVICE, steps=steps), [], []
    for f in paths:
        tj, a = map(int, re.search(r'invert_traj(\d+)initial(\d+)', f.name).groups())
        eps   = torch.from_numpy(np.load(f).astype(np.float32))       # (L, C, H, W)
        pred  = smp.sample(z0=eps, x_t=ds.data[tj][a].expand(L, -1, -1, -1), lead_times=LABELS)
        truth = ds.data[tj][a + 1: a + 1 + L]                         # leads 1..L, NOT a:a+L
        Pr.append(pred.numpy()); Tr.append(truth.numpy())
        if save_dir is not None:
            np.save(save_dir / f'traj{tj}initial{a}.npy', np.stack([pred, truth], axis=0))
    return NORM_STD * np.stack(Pr), NORM_STD * np.stack(Tr)


def forecast_reference():
    """RMSE of the ens=20 alpha=0.5 forecast perform_forecasting already wrote to disk.

    Returns (rmse, n_samples) in NORMALISED units, or (None, 0) if the run is absent.
    """
    fc_dir = DATA_PATH / 'predit_alpha=0.5'
    files  = sorted(fc_dir.glob('*_predicts.npy'))
    if not files:
        return None, 0
    fc = [(torch.from_numpy(np.load(fc_dir / f'sample{i:02d}_predicts.npy')),
           torch.from_numpy(np.load(fc_dir / f'sample{i:02d}_truth.npy')))
          for i in range(len(files))]
    return compute_metrics(fc)['rmse'], len(fc)


def main(windows=10, no_forecast=False):
    ds    = SQGLeadTimeDataset(DATA_500, split='val', random_lead_time=False,
                               eval_traj_num=100, seed=42)
    names = shared_windows()
    use   = names[:windows]
    print(f'{len(names)} windows shared by all step counts; using {len(use)}   device={DEVICE}')

    RT = {}                       # step count -> (n, L) RMSE per window per lead, PHYSICAL
    for steps, d in sorted(INVERT_DIRS.items()):
        P, T = roundtrip([d / n for n in use], steps,
                         ds, save_dir=DATA_PATH / 'round_trip' / f'steps={steps}')
        RT[steps] = np.stack([compute_rmse(p, t) for p, t in zip(P, T)])
        sd_field  = T.std()       # identical across arms — same windows, same truth
        print(f'  steps={steps:>4}:  RMSE t=1 {RT[steps][:, 0].mean():7.2f}   '
              f't=24 {RT[steps][:, 23].mean():7.2f}   (physical)')
    print(f'field std {sd_field:.1f} (physical units)\n')

    # UNITS: RT is PHYSICAL, compute_metrics is NORMALISED (ds.data is pre-divided by
    # NORM_STD). Dividing RT puts both on one axis; comparing 3.5 to 0.3 without this
    # conversion is a factor-2660 error.
    fc_rmse, n_fc = (None, 0) if no_forecast else forecast_reference()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    order = sorted(RT)
    for i, steps in enumerate(order):
        ax.plot(leads, RT[steps].mean(0) / NORM_STD, 'o-', ms=4, lw=1.8,
                color=cmap(0.05 + 0.7 * i / max(len(order) - 1, 1)),
                label=f'round trip, {steps} steps')
    if fc_rmse is not None:
        ax.plot(np.arange(1, L + 1), fc_rmse[:L], '-', lw=2.4, color='k',
                label=f'forecast, ens=20 alpha=0.5 ({n_fc} samples)')
    ax.set_yscale('log')          # round trip and forecast are ~100x apart
    ax.set_xlim(0, 24)
    ax.set_xlabel('lead time [h]'); ax.set_ylabel('RMSE  [normalised units]')
    ax.set_title('round-trip error vs forecast error, first 24 h', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()

    hdr = f"{'steps':>7}{'RMSE t=1':>12}{'RMSE t=24':>12}{'growth 24/2':>14}"
    print(hdr + (f"{'% of forecast t=24':>21}" if fc_rmse is not None else ''))
    for steps in order:
        r = RT[steps].mean(0) / NORM_STD
        row = (f'{steps:>7}{r[0]:>12.6f}{r[23]:>12.6f}'
               f'{(RT[steps][:, 23] / RT[steps][:, 1]).mean():>13.2f}x')
        print(row + (f'{r[23]/fc_rmse[23]:>20.2%}' if fc_rmse is not None else ''))
    if fc_rmse is not None:
        print(f'{"forecast":>7}{fc_rmse[0]:>12.6f}{fc_rmse[23]:>12.6f}')
    lo, hi = min(order), max(order)
    print(f'\nx{hi//lo} steps ({lo} -> {hi}): RMSE falls x{RT[lo].mean()/RT[hi].mean():.1f}'
          f'   => first-order Euler, a DISCRETISATION floor, not error feeding back')
    save_fig(fig, 'round_trip_rmse')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--windows', type=int, default=10,
                   help='windows per arm; steps=1000 costs ~2.5 min each')
    p.add_argument('--no-forecast', action='store_true',
                   help='skip the forecast reference curve')
    main(**vars(p.parse_args()))
