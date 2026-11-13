"""Generate SQG nature-run trajectories, in parallel across CPU cores.

Each trajectory is fully independent, so they are farmed out to worker
processes. Physics is unchanged from the original serial version.

Run (from anywhere):
    python SQG_generation/sqg_nature_run.py                 # all 800, 8 workers
    python SQG_generation/sqg_nature_run.py --workers 4
    python SQG_generation/sqg_nature_run.py --start 0 --count 10   # a small test batch
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Resolve paths from the file, not the cwd, so the script can be launched from
# anywhere (the old relative 'data/test' broke with a misleading EACCES).
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OUT_DIR      = PROJECT_ROOT / 'data' / 'test'
sys.path.insert(0, str(HERE))          # so `import sqgturb` works

ID_OFFSET  = 700        # first trajectory id
N_VERSIONS = 800        # ids 700 .. 1499

# ── model parameters ──────────────────────────────────────────────────────
# "dt" is tied to "N" by the CFL condition — dt must shrink if N grows, or the
# scheme goes unstable. "diff_efold" is the numerical dissipation that stops
# energy piling up at the largest wavenumbers (smallest scales).
# Known-good combinations:
#   N=1024 dt=40   diff_efold=900.         N=128 dt=600  diff_efold=86400./3.
#   N=512  dt=90   diff_efold=1800.        N=96  dt=900  diff_efold=86400./3.
#   N=256  dt=90   diff_efold=86400./16.   N=32  dt=2400 diff_efold=86400.*3
#   N=192  dt=300  diff_efold=86400./8.    N=16  dt=4800 diff_efold=86400.*9
N          = 64
dt         = 1200
diff_efold = 86400.

norder  = 8        # order of hyperdiffusion
dealias = True     # dealiased with 2/3 rule?

# Ekman damping coefficient r = dek*N**2/f, dek = ekman depth = sqrt(2.*Av/f)
# Av (turb viscosity) = 2.5 gives dek = sqrt(5/f) = 223.
# For ocean Av is 1-5, land 5-50 (Lin and Pierrehumbert, 1988), i.e. ekman
# depth 141-316 m over ocean. Spindown time of a barotropic vortex is
# tau = H/(f*dek): 10 days for H=10km, f=0.0001, dek=100m.
dek       = 0      # applied only at surface if symmetric=False
nsq       = 1.e-4
f         = 1.e-4
g         = 9.8
theta0    = 300
H         = 10.e3  # lid height
r         = dek * nsq / f
U         = 30     # jet speed
Lr        = np.sqrt(nsq) * H / f   # Rossby radius
L         = 20. * Lr
tdiab     = 10. * 86400            # thermal relaxation time scale (seconds)
symmetric = True   # if False, asymmetric equilibrium jet with zero wind at sfc
scalefact = f * theta0 / g         # scales PV to temperature units

nframes        = 100
hrs            = 1
outputinterval = hrs * 3600.
tmin           = 300. * 86400.     # spin-up before the first saved frame
tmax           = tmin + (nframes - 1) * outputinterval


def run_one(version):
    """Integrate one trajectory and write <id>.nc + <id>.npy. Returns a status str.

    Parallel-safety notes:
      * seeded per `version`, so every worker gets a DIFFERENT but reproducible
        initial condition. Without this, forked workers would share the parent's
        RNG state and could emit identical trajectories.
      * heavy libs are imported inside the worker (HDF5/pyfftw dislike fork).
      * pyfftw gets threads=1: the workers already saturate the cores, so
        per-worker threading would just oversubscribe and slow everything down.
    """
    from netCDF4 import Dataset
    from sqgturb import SQG, irfft2

    traj_id  = version + ID_OFFSET
    savedata = OUT_DIR / f'sqg_N{N}_{hrs}hrly_{traj_id}'
    npy_path = savedata.with_suffix('.npy')

    # .npy is written last, so its presence means this trajectory finished.
    # A half-done trajectory (nc but no npy) is simply redone.
    if npy_path.exists():
        return f'{traj_id}: skipped (exists)'

    rng = np.random.default_rng(version)

    pv = rng.normal(0, 100., size=(2, N, N)).astype(float)

    # add isolated blob on lid
    nexp = 20
    x = np.arange(0, 2. * np.pi, 2. * np.pi / N)
    y = np.arange(0., 2. * np.pi, 2. * np.pi / N)
    x, y = np.meshgrid(x, y)
    x = x.astype(float)
    y = y.astype(float)
    pv[1] = pv[1] + 2000. * (np.sin(x / 2) ** (2 * nexp) * np.sin(y) ** nexp)
    for k in range(2):                       # remove area mean from each level
        pv[k] = pv[k] - pv[k].mean()

    model = SQG(pv, nsq=nsq, f=f, U=U, H=H, r=r, tdiab=tdiab, dt=dt,
                diff_order=norder, diff_efold=diff_efold,
                dealias=dealias, symmetric=symmetric, threads=1,
                precision='single', tstart=0)
    model.timesteps = int(outputinterval / model.dt)

    nc = Dataset(f'{savedata}.nc', mode='w', format='NETCDF4_CLASSIC')
    try:
        nc.r = model.r
        nc.f = model.f
        nc.U = model.U
        nc.L = model.L
        nc.H = model.H
        nc.g = g
        nc.theta0 = theta0
        nc.nsq = model.nsq
        nc.tdiab = model.tdiab
        nc.dt = model.dt
        nc.diff_efold = model.diff_efold
        nc.diff_order = model.diff_order
        nc.symmetric = int(model.symmetric)
        nc.dealias = int(model.dealias)
        nc.createDimension('x', N)
        nc.createDimension('y', N)
        nc.createDimension('z', 2)
        nc.createDimension('t', None)
        pvvar = nc.createVariable('pv', float, ('t', 'z', 'y', 'x'), zlib=True)
        pvvar.units = 'K'   # pv scaled by g/(f*theta0) so du/dz = d(pv)/dy
        xvar = nc.createVariable('x', float, ('x',)); xvar.units = 'meters'
        yvar = nc.createVariable('y', float, ('y',)); yvar.units = 'meters'
        zvar = nc.createVariable('z', float, ('z',)); zvar.units = 'meters'
        tvar = nc.createVariable('t', float, ('t',)); tvar.units = 'seconds'
        xvar[:] = np.arange(0, model.L, model.L / N)
        yvar[:] = np.arange(0, model.L, model.L / N)
        zvar[0] = 0
        zvar[1] = model.H

        nout = 0
        t = 0.0
        while t < tmax:
            model.advance()
            t = model.t
            pv = irfft2(model.pvspec)
            if t >= tmin:
                pvvar[nout, :, :, :] = pv
                tvar[nout] = t
                nout += 1
    finally:
        nc.close()

    # TODO: would be nice to compute some data statistics and save those too.
    nc = Dataset(f'{savedata}.nc', 'r')
    X = np.array(nc['pv'][:])
    nc.close()
    np.save(npy_path, X)
    return f'{traj_id}: ok ({nout} frames)'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8,
                   help='parallel processes (default 8 = physical cores)')
    p.add_argument('--start', type=int, default=0,
                   help='first version index (id = index + %d)' % ID_OFFSET)
    p.add_argument('--count', type=int, default=N_VERSIONS,
                   help='how many trajectories to generate')
    args = p.parse_args()

    # One thread per worker: the processes already use every core, so letting
    # each one also spawn FFT/BLAS threads only causes oversubscription.
    # Must be set before the workers import numpy/pyfftw.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    versions = list(range(args.start, min(args.start + args.count, N_VERSIONS)))

    print(f'{len(versions)} trajectories -> ids '
          f'{versions[0] + ID_OFFSET}..{versions[-1] + ID_OFFSET}')
    print(f'output   : {OUT_DIR}')
    print(f'workers  : {args.workers}')
    print(f'per traj : {nframes} frames, {outputinterval / 3600.:g} h apart, '
          f'after {tmin / 86400.:g} d spin-up')

    t0 = time.perf_counter()
    if args.workers == 1:
        for v in tqdm(versions):
            run_one(v)
    else:
        # 'spawn' gives each worker a clean interpreter: HDF5 and pyfftw are not
        # reliably fork-safe, and fork would also clone the parent's RNG state.
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.workers) as pool:
            for _ in tqdm(pool.imap_unordered(run_one, versions),
                          total=len(versions)):
                pass
    dt_total = time.perf_counter() - t0
    print(f'done in {dt_total / 3600:.2f} h '
          f'({dt_total / max(len(versions), 1):.1f} s per trajectory)')


if __name__ == '__main__':
    main()
