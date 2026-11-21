"""Step 2: direct forecast with latents from the fitted time-varying OU models.

Same configuration as step 1 -- test split, 50 trajectories, 1 anchor each,
ens 10, leads 1..24, Euler 100 steps, seed 0 -- and the same windows, the same
per-window seed and therefore the same innovation stream. Only the latent chain
differs:

    step 1   z_k = a   z_{k-1} + sqrt(1 - a^2)   eps_k    a = 0 or 1, constant
    step 2   z_k = a_k z_{k-1} + sqrt(1 - a_k^2) eps_k    a_k = exp(-c2_k / 2)
                                                          from a fitted net

Both arms are c2-only fits: the latents are treated as unit variance, so only
alpha is modelled (Analysis/fit.py, c2_mle). Both were fitted on the Euler
latents of data/invert/from_val (the VAL split), so forecasting on test is
honest.

    ou_no_x0   Analysis/Models/time_varyingOU_withoutx0.pt   TimeVaryingOU
               c2(lead, ch, y). alpha varies with lead but is flat across the
               box to within 1%, so this is effectively a per-lead scalar.
               val nll 0.0870.
    ou_x0      Analysis/Models/c2_euler.pt                   X0ConditionedOU
               c2(lead, ch, y, x0). The anchor state goes through a whole-box
               dilated CNN and tilts a per-pixel linear readout (FiLM-style:
               a conditional linear layer, base + A(lead,ch,y) . F(x0)).
               val nll 0.0115, a 7.5x reduction. Box-mean alpha is the SAME as
               the x0-free arm; what x0 buys is the spatial pattern (per-pixel
               alpha spans 0.35..0.92 against 0.767..0.774).

Usage:
    python -m Condition_model.ou_forecast --arm ou_x0
    python -m Condition_model.ou_forecast --arm ou_no_x0

The metric for these arms is the TEMPORAL DIFFERENCE, mean |x_k - x_k-1| per lag
over windows, members and pixels (Analysis/utils.temporal_difference, the CRPI
appendix diagnostic). It goes to temporal_difference.npz; nothing is plotted.
rmse / crps / ssr are not computed -- pass --metrics for those as well.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

from .dataset import DATA_PATH
from .forecasting import _sha256, perform_direct_forecasting

# Analysis/ is not a package and fit.py does `from utils import *`, so it has to
# be imported with Analysis on sys.path. Put it FIRST: `utils` must resolve to
# Analysis/utils.py, not Condition_model/utils.py (which is only ever imported
# as Condition_model.utils, so the two never collide in sys.modules).
ANALYSIS = Path(__file__).resolve().parents[1] / 'Analysis'
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
os.environ.setdefault('MPLBACKEND', 'Agg')          # Analysis/utils.py imports pyplot

from fit import CNNFeatures, TimeVaryingOU, X0ConditionedOU   # noqa: E402
from utils import get_latents as ou_latents                   # noqa: E402

MODELS = ANALYSIS / 'Models'

# Architectures must match the fits exactly, or the state dicts will not load.
# ou_no_x0 : TimeVaryingOU defaults (emb 64, hidden 256, 1 layer, x_dim 0, y_dim 2)
#            as used in Analysis/time-varingOU.ipynb; saved from history['best_state'].
# ou_x0    : the CONFIG of Analysis/time-varyingOU.py; saved at the last epoch.
ARMS = {
    'ou_no_x0': dict(
        ckpt=MODELS / 'time_varyingOU_withoutx0.pt',
        build=lambda: TimeVaryingOU(emb_dim=64, hidden_dim=256, n_hidden=1,
                                    x_dim=0, y_dim=2, fit_sigma=False),
        needs_x0=False,
        note='c2(lead, ch, y): no x0 anywhere; alpha is flat across the box to ~1%',
        val_nll=0.08704,
    ),
    'ou_x0': dict(
        ckpt=MODELS / 'c2_euler.pt',
        build=lambda: X0ConditionedOU(
            CNNFeatures(r=32, width=32, dilations=(1, 2, 4, 8, 16, 1),
                        hand=False, dropout=0.1),
            emb_dim=64, hidden_dim=256, n_hidden=1, x_dim=0, y_dim=2,
            fit_sigma=False),
        needs_x0=True,
        note='c2(lead, ch, y, x0): whole-box dilated CNN on the anchor state',
        val_nll=0.01154,
    ),
}


def load_arm(arm, device):
    """Build the architecture, load the checkpoint strictly, return it in eval mode."""
    spec = ARMS[arm]
    assert spec['ckpt'].exists(), f"missing checkpoint: {spec['ckpt']}"
    model = spec['build']()
    model.load_state_dict(torch.load(spec['ckpt'], map_location='cpu'), strict=True)
    model = model.to(device).eval()
    print(f"{arm}: {spec['ckpt'].name}, "
          f"{sum(p.numel() for p in model.parameters())} parameters, "
          f"val nll {spec['val_nll']:.5f}")
    print(f"  {spec['note']}")
    return model


class _Contiguous(torch.nn.Module):
    """Pass-through wrapper that makes the model's c2 contiguous. Needed for pairing.

    Both nets end in `h[..., 0]`, a strided VIEW of the (..., n_heads) tensor, so
    c2 comes back non-contiguous. Analysis.get_latents then builds alpha from it
    (also non-contiguous), and `alpha[:, :, k-1] * xi` inherits that layout, so
    the `torch.randn_like(xi)` of the NEXT step fills a non-contiguous tensor --
    which draws the same count of normals but lays them out in a different order,
    walking this arm off the innovation stream the other arms use.

    X0ConditionedOU is where it bites: its alpha has a full H, so the product
    keeps the odd layout. TimeVaryingOU's alpha has H=1, so broadcasting already
    produces a fresh contiguous result and the stream stays aligned either way.

    One .contiguous() on the model output costs 1.5 MB and nothing measurable,
    and keeps every arm bit-paired. Verified: with it, the whole 24-lead chain
    rebuilds exactly (max error 0.0) from the alpha=0 arm's innovations.
    """
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.fit_sigma = inner.fit_sigma        # get_latents reads this off the model

    def forward(self, lead, x0=None):
        out = self.inner(lead, x0)
        return (out[0].contiguous(), out[1].contiguous()) if self.fit_sigma \
            else out.contiguous()


def make_latent_fn(model, needs_x0):
    """A latent_fn for perform_direct_forecasting, one window at a time.

    x0 arrives from the forecast loop already divided by NORM_STD (2660), which
    is exactly the scaling Analysis/utils.get_window used for training, so it
    goes to the net as is.

    One window per call (N=1) matters for the pairing: Analysis.get_latents then
    draws randn(1, ens, C, H, W) for lead 1 and one randn_like per lead, the
    same stream in the same order as the constant-alpha get_latents of step 1.
    Same per-window seed therefore gives every arm the same innovations, and a
    bit-identical lead-1 latent.
    """
    model = _Contiguous(model).eval()

    def latent_fn(x0, ens, n_lead, device):
        x0_b = x0[None].to(device).float() if needs_x0 else None
        z = ou_latents((1, ens, n_lead, *x0.shape), model, x0=x0_b, device=device)
        return z[0]                                  # (ens, L, C, H, W)
    return latent_fn


def report_alpha(model, needs_x0, n_lead, device):
    """Print what the fitted chain does, so the log says which alpha ran."""
    lead = torch.arange(1, n_lead + 1, dtype=torch.float32, device=device)
    x0 = torch.zeros(1, 2, 64, 64, device=device) if needs_x0 else None
    with torch.no_grad():
        c2 = model(lead, x0)
    a = torch.exp(-0.5 * c2)[:, 1:]                  # transitions 2..L
    am = a.mean((2, 3, 4))[0]
    rho = torch.cumprod(am, 0)
    where = ' (at x0 = 0, so indicative only)' if needs_x0 else ''
    print(f'  alpha_2 {am[0]:.3f} -> alpha_{n_lead} {am[-1]:.3f}, '
          f'rho(1,{n_lead}) {rho[-1]:.4f}{where}')


def main():
    p = argparse.ArgumentParser(
        description='Direct forecast whose latents come from a fitted '
                    'time-varying OU model (step 2).')
    p.add_argument('--arm', required=True, choices=sorted(ARMS),
                   help='which fitted model drives the latent chain')
    p.add_argument('--split', default='test')
    p.add_argument('--n-traj', type=int, default=50)
    p.add_argument('--anchors', type=int, default=1)
    p.add_argument('--ens', type=int, default=10)
    p.add_argument('--n-lead', type=int, default=24)
    p.add_argument('--steps', type=int, default=100, help='Euler steps')
    p.add_argument('--micro-ens', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dirname', default=None,
                   help='output folder under data/forecasting (default: the arm name)')
    p.add_argument('--metrics', action='store_true',
                   help='also compute rmse / crps / ssr into metrics.npz '
                        '(off by default: the metric here is the temporal difference)')
    p.add_argument('--plot', action='store_true',
                   help='write metrics.png; needs --metrics. The temporal '
                        'difference is never plotted here')
    p.add_argument('--no-resume', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec = ARMS[args.arm]
    model = load_arm(args.arm, device)
    report_alpha(model, spec['needs_x0'], args.n_lead, device)

    out = perform_direct_forecasting(
        split=args.split, n_traj=args.n_traj, anchors_per_traj=args.anchors,
        ens=args.ens, n_lead=args.n_lead, steps=args.steps,
        micro_ens=args.micro_ens, seed=args.seed,
        dirname=args.dirname or args.arm, device=device,
        resume=not args.no_resume, plot=args.plot and args.metrics,
        metrics=args.metrics, tdiff=True,
        latent_fn=make_latent_fn(model, spec['needs_x0']),
        latent_meta={
            'kind':    args.arm,
            'model':   str(spec['ckpt'].relative_to(ANALYSIS.parent)),
            'sha256':  _sha256(spec['ckpt']),
            'class':   type(model).__name__,
            'fit_data': 'data/invert/from_val (Euler latents, val split)',
            'note':    spec['note'],
            'val_nll': spec['val_nll'],
        },
    )
    print('step 2 arm written to', out.relative_to(DATA_PATH.parent))


if __name__ == '__main__':
    main()
