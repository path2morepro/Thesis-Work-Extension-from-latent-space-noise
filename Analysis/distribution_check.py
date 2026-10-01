"""Test 1 — are the inverted latents N(0, I)?

The flow map is deterministic and invertible, so eps = Phi^-1(x) is a fact about the
data, not a draw. Under the training assumption it should be standard normal: per-lead
mean 0, std 1, excess kurtosis 0, and the two SQG levels uncorrelated.

This tests the PER-PIXEL marginal only. It says nothing about spatial structure — a
perfectly red field passes every check here. That is what power_spectrum.py is for.

Run:  python distribution_check.py [--steps 1000]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import utils
from utils import INVERT_DIRS, L, leads, cmap, load_invert, save_fig


def marginal_stats(Z):
    """(N, L, C, H, W) -> per-lead moments and cross-channel correlation.

    Each lead is pooled over windows, channels and pixels, so every number is one
    marginal per lead rather than a single number for the whole set.

    Returns:
        dict of (L,) arrays: mean, std, kurt (EXCESS, 0 for Gaussian), xchan.
    """
    Zl = Z.transpose(1, 0, 2, 3, 4).reshape(Z.shape[1], -1)      # (L, N*C*H*W)
    c0 = Z[:, :, 0].reshape(len(Z), Z.shape[1], -1)
    c1 = Z[:, :, 1].reshape(len(Z), Z.shape[1], -1)
    return dict(
        mean  = Zl.mean(1),
        std   = Zl.std(1),
        kurt  = stats.kurtosis(Zl, axis=1),
        xchan = np.array([np.corrcoef(c0[:, t].ravel(), c1[:, t].ravel())[0, 1]
                          for t in range(Z.shape[1])]),
    )


def plot(st, steps):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))
    panels = [(axes[0], st['std'],   'std  (1 expected)',               1.0),
              (axes[1], st['kurt'],  'excess kurtosis  (0 expected)',   0.0),
              (axes[2], st['xchan'], 'cross-channel corr  (0 expected)', 0.0)]
    for ax, y, ttl, ref in panels:
        ax.plot(leads, y, 'o-', ms=4, color=cmap(0.2))
        ax.axhline(ref, color='k', lw=0.9, ls='--')
        ax.set_xlabel('lead tau [h]'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
    axes[0].plot(leads, st['mean'], 's-', ms=3, color=cmap(0.7), label='mean (0 expected)')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'marginal of the inverted latents — {steps}-step inversion', fontsize=10)
    fig.tight_layout()
    return fig


def main(steps=1000):
    Z, _, _, _ = load_invert(INVERT_DIRS[steps])
    st = marginal_stats(Z)

    print(f'{len(Z)} windows, {steps}-step inversion')
    print(f'pooled over all leads:  mean {Z.mean():+.4f}   std {Z.std():.4f}   '
          f'excess kurtosis {stats.kurtosis(Z.ravel()):+.4f}')
    print(f"per-lead std   range [{st['std'].min():.4f}, {st['std'].max():.4f}]")
    print(f"per-lead kurt  range [{st['kurt'].min():+.4f}, {st['kurt'].max():+.4f}]")
    print(f"cross-channel  range [{st['xchan'].min():+.4f}, {st['xchan'].max():+.4f}]")
    print('NOTE cross-channel correlation is small but consistently POSITIVE at every '
          'lead;\n     it is reported, not tested — no error bar is computed for it here.')
    save_fig(plot(st, steps), f'distribution_check_steps={steps}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=1000, choices=sorted(INVERT_DIRS))
    main(**vars(p.parse_args()))
