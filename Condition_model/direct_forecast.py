"""Direct forecast of leads 1..24 from one anchor, for two noise arms.

Separate entry point so Condition_model.forecasting's own CLI (the ARCI
forecast over whole trajectories) stays exactly as it is.

No ARCI. Every lead is its own map Phi(x0, tau) from the SAME anchor, so lead
24 is not built from lead 23 and nothing is re-anchored. The only thing that
differs between the two arms is the latent chain:

    uncorrelated   alpha = 0.0   z_t = eps_t      an independent latent per lead
    fixed          alpha = 1.0   z_t = z_{t-1}    the SAME latent at every lead

the two extremes of the variance-preserving chain
z_t = a z_{t-1} + sqrt(1-a^2) eps_t.

Both arms run on the SAME windows with the SAME innovation stream (the torch
seed is derived per window), so the difference between their metrics is the
latent correlation and nothing else.

    python -m Condition_model.direct_forecast --alpha 0   # uncorrelated
    python -m Condition_model.direct_forecast --alpha 1   # fixed

Add the temporal difference to an arm that is already finished, reading the
saved windows and nothing else:

    python -m Condition_model.direct_forecast --alpha 0 --dirname uncorrelated_noise \
        --tdiff --no-metrics
"""
import argparse

from .forecasting import ALPHA_CONST, perform_direct_forecasting

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                   help='alpha was measured on val, so test keeps this held out')
    p.add_argument('--n-traj', type=int, default=50,
                   help='trajectories to forecast; windows = n_traj * anchors')
    p.add_argument('--anchors', type=int, default=1,
                   help='anchors per trajectory. 1 = every window from a '
                        'different trajectory, sharing no frames.')
    p.add_argument('--ens', type=int, default=10, help='ensemble members')
    p.add_argument('--alpha', type=float, required=True,
                   help=f'0.0 uncorrelated (fresh latent per lead), 1.0 fixed '
                        f'(one latent reused), {ALPHA_CONST} the measured lag-1')
    p.add_argument('--n-lead', type=int, default=24, help='leads 1..n-lead')
    p.add_argument('--steps', type=int, default=100, help='Euler steps')
    p.add_argument('--micro-ens', type=int, default=5,
                   help='members per forward call. 5 -> 120 rows -> 2.2 GiB; '
                        'all 10 at once is 240 rows -> 4.2 GiB on a 6 GB card.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dirname', default=None,
                   help='directory under data/forecasting/ (default by alpha: '
                        'uncorrelated_noise / fixed_noise / correlated_noise)')
    p.add_argument('--no-resume', action='store_true',
                   help='recompute windows even if their files exist')
    p.add_argument('--tdiff', action='store_true',
                   help='also write temporal_difference.npz, mean |x_k - x_k-1| '
                        'per lag (Analysis/utils.temporal_difference)')
    p.add_argument('--no-metrics', action='store_true',
                   help='skip rmse / crps / ssr. With --tdiff on an arm whose '
                        'windows are all on disk this is a cheap read-only pass '
                        'that adds the temporal difference and touches nothing else.')
    a = p.parse_args()

    perform_direct_forecasting(
        split            = a.split,
        n_traj           = a.n_traj,
        anchors_per_traj = a.anchors,
        ens              = a.ens,
        alpha            = a.alpha,
        n_lead           = a.n_lead,
        steps            = a.steps,
        micro_ens        = a.micro_ens,
        seed             = a.seed,
        dirname          = a.dirname,
        resume           = not a.no_resume,
        metrics          = not a.no_metrics,
        tdiff            = a.tdiff,
    )
