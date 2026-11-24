import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from natsort import natsorted

from utils import *  

# Condition_model lives next to Analysis/, so put the Thesis repo root on the path.
if str(ROOT.parent) not in sys.path:
    sys.path.append(str(ROOT.parent))

import Condition_model.utils as cmu   
from Condition_model.utils import (compute_rmse, compute_crps, compute_ssr,
                                   compute_metrics, plot_metrics, print_metrics, MetricAccumulator)

forecasting_path = ROOT.parent / 'data' / 'predict_constant_alpha'

pred_files  = natsorted(forecasting_path.glob('sample*_predicts.npy'))
truth_files = natsorted(forecasting_path.glob('sample*_truth.npy'))
assert len(pred_files) == len(truth_files) > 0, (len(pred_files), len(truth_files))

preds    = np.stack([np.load(f, mmap_mode='r') for f in pred_files]).astype(np.float32, copy=False)   # (n_traj, lead, ens, ...)
truth    = np.stack([np.load(f, mmap_mode='r') for f in truth_files]).astype(np.float32, copy=False)  # (n_traj, lead, ...)
preds24 = preds[:, :24, ...]
truth24 = truth[:, :24, ...]

# visualize SQG dataset
# visualize the inversion as well
visual_basepath = Path('/home/trace/Documents/Msc thesis writing/liuthesis-master/figures')
X_sample = truth[0, ...]
Z, _, _ = load_invert(ROOT.parent / 'data' / 'invert' / 'from_val')
Z_sample = Z[0, ...]

ts = [0, 1, 2, 3, 23]

def visualize_frames(data, ts, name):
    vmax = np.abs(data[ts, 0]).max()          # one colour scale for all frames of this dataset
    for t in ts:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(data[t, 0, ...], cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        ax.set_axis_off()                      # no label, no title
        fig.savefig(visual_basepath / f'{name}_{t}.pdf',
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

visualize_frames(Z_sample, ts, 'latent')
visualize_frames(X_sample, ts, 'sqg')


results = [(torch.from_numpy(preds24[i]), torch.from_numpy(truth24[i])) for i in range(len(preds24))]
out_path = ROOT / 'Result' 
metrics = cmu.compute_metrics(results)
cmu.plot_metrics(metrics, out_path / 'metrics24h.png')



preds24_10 = preds24[:, :, :10, ...]   # first 10 members, to match the 10-member runs below
fixed_noise_path = ROOT.parent / 'data' / 'forecasting' / 'fixed_noise'
uncorrelated_noise_path = ROOT.parent / 'data' / 'forecasting' / 'uncorrelated_noise'
ou_x0_path = ROOT.parent / 'data' / 'forecasting' / 'ou_x0'
ou_no_x0_path =  ROOT.parent / 'data' / 'forecasting' / 'ou_no_x0'

def load_data(path, type='predicts'):
    files = natsorted(path.glob(f'traj*initial*_{type}.npy'))
    x = np.stack([np.load(f).astype(np.float32) for f in files])
    return x

fnf = load_data(fixed_noise_path)
ucnf = load_data(uncorrelated_noise_path)
ou_x0f = load_data(ou_x0_path)
ou_no_x0f = load_data(ou_no_x0_path)

truth = load_data(ou_no_x0_path, 'truth')


td_fnf = temporal_difference(fnf)
td_ucnf = temporal_difference(ucnf)
td_x0 = temporal_difference(ou_x0f)
td_nox0 = temporal_difference(ou_no_x0f)
td_tru = temporal_difference(truth)
td_cons = temporal_difference(preds24_10)



diff1 = {
    'Time-varying OU process': td_nox0,
    'Ground truth': td_tru,
    'Constant OU process': td_cons
}

diff2 = {
    'Fixed noise': td_fnf,
    'Independent noise': td_ucnf,
    'Ground truth': td_tru
    }

plot_temporal_difference(diff1 | diff2, filename='overview')
plot_temporal_difference(diff1, filename='fits')


