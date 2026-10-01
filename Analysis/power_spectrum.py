"""Test 2 — is Sigma the identity, and is it the same at every lead?

Per-pixel marginals say nothing about Sigma. The ring spectrum does: under Sigma = I the
per-mode density lambda(k)/Var is FLAT at 1. It is not — the latents are red.

Second question on the same figure: does lambda_t(k) depend on the lead? If Sigma_t were
one fixed Sigma scaled by a scalar sigma(t)^2, every ring would move by the SAME factor
between two leads. Comparing the total-energy factor against the per-ring spread answers
whether the SHAPE changes or only the level.

Run:  python power_spectrum.py [--steps 1000]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils
from utils import (INVERT_DIRS, K_ISO, ks, dof_b, cmap, B,
                   load_invert, ring_spectrum, save_fig)


def plot(lam_pool, lam_lead, var_f, steps):
    share = lam_pool * dof_b / (lam_pool * dof_b).sum()     # ring share of tr(Sigma)
    cum   = np.cumsum(share)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0))
    for i, t in enumerate([1, 6, 12, 24]):
        axes[0].loglog(ks, lam_lead[t - 1, 1:K_ISO + 1] / var_f, '-', lw=1.1,
                       color=cmap(0.8 * i / 3), label=f'lead tau = {t} h')
    axes[0].loglog(ks, lam_pool[1:K_ISO + 1] / var_f, 'k-', lw=2.2, label='pooled')
    axes[0].axhline(1.0, color='gray', lw=1.0, ls='--', label='white noise (Sigma = I)')
    axes[0].set_xlabel('ring k'); axes[0].set_ylabel('lambda(k) / field variance')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which='both')

    axes[1].plot(np.arange(B), cum, 'k.-', ms=4)
    for q in (0.5, 0.9):
        kq = int(np.searchsorted(cum, q))
        axes[1].axvline(kq, color='gray', lw=0.8, ls=':')
        axes[1].annotate(f'{q:.0%} by k={kq}', (kq, q), textcoords='offset points',
                         xytext=(8, -12), fontsize=9)
    axes[1].set_xlabel('ring k'); axes[1].set_ylabel('cumulative share of tr(Sigma)')
    axes[1].grid(alpha=0.3)
    fig.suptitle(f'ring spectrum of the inverted latents — {steps}-step inversion', fontsize=10)
    fig.tight_layout()
    return fig, cum


def main(steps=1000):
    Z, _, _, _ = load_invert(INVERT_DIRS[steps])
    lam_pool, lam_lead = ring_spectrum(Z)
    var_f = Z.var()

    fig, cum = plot(lam_pool, lam_lead, var_f, steps)

    print(f'{len(Z)} windows, {steps}-step inversion')
    print(f'lambda(k)/var:  k=1 {lam_pool[1]/var_f:.2f}   k=5 {lam_pool[5]/var_f:.2f}   '
          f'k=16 {lam_pool[16]/var_f:.2f}   k=32 {lam_pool[32]/var_f:.2f}'
          f'    (all 1.00 if Sigma = I)')
    print(f'tr(Sigma) share:  k<=2 {cum[2]:.1%}   k<=8 {cum[8]:.1%}')

    # Same ring, different leads. A scalar sigma(t)^2 * Sigma would move every ring by
    # the same factor; a spread of per-ring factors means the SHAPE moves too.
    E_lead = (lam_lead * dof_b[None, :]).sum(1)
    spread = lam_lead[:, 1:K_ISO + 1].max(0) / lam_lead[:, 1:K_ISO + 1].min(0)
    print(f'\nacross leads: total energy varies x{E_lead.max()/E_lead.min():.2f}, '
          f'but per-ring max/min spans x{spread.min():.2f} .. x{spread.max():.2f}')
    print('=> the rings move by DIFFERENT factors, so Sigma_t is not one Sigma scaled by')
    print('   a scalar sigma(t)^2 — the SHAPE changes with lead, not just the level.')
    save_fig(fig, f'power_spectrum_steps={steps}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=1000, choices=sorted(INVERT_DIRS))
    main(**vars(p.parse_args()))
