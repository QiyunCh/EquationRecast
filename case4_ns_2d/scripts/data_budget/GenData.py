#!/usr/bin/env python3
"""
GenData.py — Generate all NS datasets for Test 4 data-budget study.

Generates:
  - canonical/data_canonical_N1500.h5      (1500 samples at Re*=250)
  - parametric/data_parametric_N1500.h5    (1500 samples, Re~U[50,400])
  - redistributed/data_redistributed.h5    (5 Re × 300 samples; Re∈{50,100,200,300,400})

Subsets for smaller budgets are created at training time (just slice first B samples).

Uses multiprocessing to parallelize over samples (one ETDRK4-style IMEX solve per sample).
"""
from __future__ import annotations
import os, time, h5py, sys, argparse
import numpy as np
from multiprocessing import Pool

# Local NS module
import VorticityNS_2D as ns

# Don't import VorticityNS_2D_parametric directly because we use our own Re list.
# We override ns.RE per worker.


N = 128
L_DOMAIN = 1.0
TOL = 1e-6
MAX_STEPS = 2500
DT = 0.2
SEED = 1234


def _worker(args):
    """Generate one (S, omega) sample at given Re and seed offset."""
    sample_idx, Re, seed = args
    rng = np.random.default_rng(seed)
    kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias = ns.make_spectral_operators(N)
    S = ns.generate_source(N, kx, ky, dealias, rng)
    # Override module-level RE per-worker
    ns.RE = float(Re)
    omega, ux, uy, nsteps, relres = ns.solve_steady_vorticity(
        S, ikx, iky, lap_symbol, inv_ksq, dealias)
    return sample_idx, S, omega, ux, uy, float(Re), nsteps, relres


def generate_set(out_h5: str, n_samples: int, Re_func, n_workers: int = 4,
                 seed_base: int = SEED):
    """Re_func(idx) -> Re value for sample idx."""
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    tasks = [(i, Re_func(i), seed_base + i) for i in range(n_samples)]
    print(f"  generating {n_samples} samples to {out_h5} with {n_workers} workers...")
    S_all = np.zeros((n_samples, N, N), dtype=np.float32)
    O_all = np.zeros((n_samples, N, N), dtype=np.float32)
    U_all = np.zeros((n_samples, N, N, 2), dtype=np.float32)
    Re_all = np.zeros((n_samples,), dtype=np.float32)
    steps_all = np.zeros((n_samples,), dtype=np.int32)
    relres_all = np.zeros((n_samples,), dtype=np.float32)

    t0 = time.time()
    if n_workers <= 1:
        for t in tasks:
            i, S, omega, ux, uy, Re, st, rr = _worker(t)
            S_all[i] = S; O_all[i] = omega
            U_all[i, ..., 0] = ux; U_all[i, ..., 1] = uy
            Re_all[i] = Re; steps_all[i] = st; relres_all[i] = rr
            if (i + 1) % 50 == 0 or i == 0:
                eta = (time.time() - t0) / (i + 1) * (n_samples - i - 1)
                print(f"    {i+1}/{n_samples} t={time.time()-t0:.0f}s ETA={eta:.0f}s")
    else:
        with Pool(n_workers) as pool:
            for cnt, (i, S, omega, ux, uy, Re, st, rr) in enumerate(pool.imap_unordered(_worker, tasks)):
                S_all[i] = S; O_all[i] = omega
                U_all[i, ..., 0] = ux; U_all[i, ..., 1] = uy
                Re_all[i] = Re; steps_all[i] = st; relres_all[i] = rr
                if (cnt + 1) % 50 == 0 or cnt == 0:
                    eta = (time.time() - t0) / (cnt + 1) * (n_samples - cnt - 1)
                    print(f"    {cnt+1}/{n_samples} t={time.time()-t0:.0f}s ETA={eta:.0f}s")

    # Compute global stats
    S_global_min = float(S_all.min()); S_global_max = float(S_all.max())
    omega_global_min = float(O_all.min()); omega_global_max = float(O_all.max())
    u_global_min = float(U_all.min()); u_global_max = float(U_all.max())
    Re_global_min = float(Re_all.min()); Re_global_max = float(Re_all.max())

    # Re_field broadcast (for parametric/PINO inputs)
    Re_field = np.broadcast_to(Re_all[:, None, None], (n_samples, N, N)).astype(np.float32).copy()

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("S", data=S_all)
        f.create_dataset("omega", data=O_all)
        f.create_dataset("u", data=U_all)
        f.create_dataset("Re", data=Re_field)
        f.create_dataset("Re_scalar", data=Re_all)
        f.create_dataset("solver_steps", data=steps_all)
        f.create_dataset("solver_relres", data=relres_all)
        f.attrs["domain"] = "(0,1)^2 periodic"
        f.attrs["grid_n"] = N
        f.attrs["DT"] = DT; f.attrs["MAX_STEPS"] = MAX_STEPS
        f.attrs["TOL_REL_RESID"] = TOL
        f.attrs["S_global_min"] = S_global_min; f.attrs["S_global_max"] = S_global_max
        f.attrs["omega_global_min"] = omega_global_min; f.attrs["omega_global_max"] = omega_global_max
        f.attrs["u_global_min"] = u_global_min; f.attrs["u_global_max"] = u_global_max
        f.attrs["Re_global_min"] = Re_global_min; f.attrs["Re_global_max"] = Re_global_max
    print(f"  Saved {out_h5}  elapsed={time.time()-t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["canonical", "parametric", "redistributed", "all"],
                   default="all")
    p.add_argument("--n_canonical", type=int, default=1500)
    p.add_argument("--n_parametric", type=int, default=1500)
    p.add_argument("--Re_per_redistributed", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    if args.target in ("canonical", "all"):
        # All at Re*=250
        generate_set("canonical/data_canonical_N1500.h5", args.n_canonical,
                     Re_func=lambda i: 250.0, n_workers=args.workers, seed_base=SEED)

    if args.target in ("parametric", "all"):
        rng = np.random.default_rng(SEED + 1)
        Re_samples = rng.uniform(50.0, 400.0, size=args.n_parametric)
        generate_set("parametric/data_parametric_N1500.h5", args.n_parametric,
                     Re_func=lambda i: float(Re_samples[i]), n_workers=args.workers,
                     seed_base=SEED + 10000)

    if args.target in ("redistributed", "all"):
        re_values = [50.0, 100.0, 200.0, 300.0, 400.0]
        per = args.Re_per_redistributed
        total = per * len(re_values)

        def re_func(i):
            return re_values[i // per]

        generate_set("redistributed/data_redistributed.h5", total,
                     Re_func=re_func, n_workers=args.workers, seed_base=SEED + 20000)


if __name__ == "__main__":
    main()
