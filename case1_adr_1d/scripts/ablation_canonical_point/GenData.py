#!/usr/bin/env python3
"""
GenData.py — Generate ADR 1D canonical training data for a given (Pe*, Da*).

Same source seed=12345 across all canonicals so S is identical; only u differs
according to the chosen canonical operator. 1000 GRF sources, length scale 0.08,
N=201 on periodic [0,1).

Usage:
    python GenData.py <Pe_star> <Da_star> <output_h5>
"""
from __future__ import annotations
import sys
import math
import h5py
import numpy as np

from ADR import (
    make_periodic_grid, wavenumbers,
    sample_grf_periodic_se_fourier, rescale_to_minus1_plus1,
    solve_adr_fourier,
)


def generate(pe_star: float, da_star: float, out_h5: str,
             n_samples: int = 1000, N: int = 201,
             grf_length_scale: float = 0.08, seed: int = 12345) -> None:
    L = 1.0
    x, dx = make_periodic_grid(L, N)
    k = wavenumbers(L, N)
    rng = np.random.default_rng(seed)

    S_all = np.empty((n_samples, N), dtype=np.float64)
    u_all = np.empty((n_samples, N), dtype=np.float64)

    u_min_global = math.inf
    u_max_global = -math.inf

    for i in range(n_samples):
        S_raw = sample_grf_periodic_se_fourier(rng, k=k, length_scale=grf_length_scale)
        S = rescale_to_minus1_plus1(S_raw)
        u = solve_adr_fourier(S, Pe=pe_star, Da=da_star, k=k)

        S_all[i, :] = S
        u_all[i, :] = u

        u_min_global = min(u_min_global, float(np.min(u)))
        u_max_global = max(u_max_global, float(np.max(u)))

        if (i + 1) % 200 == 0:
            print(f"  [Pe*={pe_star}, Da*={da_star}] generated {i+1}/{n_samples}")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("source", data=S_all)
        f.create_dataset("solution", data=u_all)
        f.attrs["L"] = float(L)
        f.attrs["N"] = int(N)
        f.attrs["dx"] = float(dx)
        f.attrs["Pe"] = float(pe_star)
        f.attrs["Da"] = float(da_star)
        f.attrs["nsamples"] = int(n_samples)
        f.attrs["grf_length_scale"] = float(grf_length_scale)
        f.attrs["seed"] = int(seed)
        f.attrs["solution_min"] = float(u_min_global)
        f.attrs["solution_max"] = float(u_max_global)

    print(f"  Saved: {out_h5}  u_range=[{u_min_global:.4e}, {u_max_global:.4e}]")


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        pe = float(sys.argv[1])
        da = float(sys.argv[2])
        out = sys.argv[3]
        generate(pe, da, out)
    else:
        # Default: generate all 4 canonicals
        canonicals = [(2.0, 4.0), (10.0, 10.0), (2.0, 25.0), (25.0, 2.0)]
        for pe, da in canonicals:
            generate(pe, da, f"data/data_Pe{pe:g}_Da{da:g}.h5")
