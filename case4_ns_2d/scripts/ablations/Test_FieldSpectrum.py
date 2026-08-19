#!/usr/bin/env python3
"""
Test_FieldSpectrum.py — Generate predicted vorticity fields + benchmarks at
selected Re values for a few representative sources, save to h5 for later
spectrum/field plotting (mirrors paper Figure 4 bottom).

Models compared: canonical_dataonly (recast), canonical_pinn (recast),
                 parametric FNO, PINO (custom), PINO (ext-repo), benchmark.
"""
from __future__ import annotations
import os, sys, time, h5py
import numpy as np
import torch

from FNO2D import FNO2d
from Train_PINN_Canonical import make_kgrids
from Test_Compare import (
    MODELS, load_model, load_norm_stats, solve_reference,
    recast_aitken_for_canonical, fno_predict_omega, fno_ext_predict_omega,
)
import VorticityNS_2D as ns


RE_DETAIL = [50, 150, 250, 350]   # matches paper Figure 4 bottom
SRC_INDICES = [0]                   # one representative source for plotting
RE_STAR = 250.0
N = 128
L_DOMAIN = 1.0
SEED = 13
N_SOURCES = 20  # match Test_Compare's source set; we pick SRC_INDICES from it
K_HARD = 21


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FieldSpectrum] device={device}")
    can_stats, par_stats = load_norm_stats()

    # Load all available models
    loaded = {}
    for name, info in MODELS.items():
        if not os.path.exists(info["ckpt"]):
            print(f"  SKIP {name}: {info['ckpt']} missing"); continue
        loaded[name] = (load_model(name, info, device), info)
        print(f"  loaded {name}")

    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    Kx64 = Kx.double(); Ky64 = Ky.double(); K2_inv64 = K2_inv.double()

    # Same sources as Test_Compare (seed=13, generated identically)
    rng = np.random.default_rng(SEED)
    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    sources_np = np.stack(
        [ns.generate_source(N, kx_np, ky_np, dealias_np, rng) for _ in range(N_SOURCES)],
        axis=0
    )

    out_h5 = "results/test3_fields.h5"
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Re_list", data=np.array(RE_DETAIL))
        f.create_dataset("src_indices", data=np.array(SRC_INDICES))
        f.attrs["seed"] = SEED
        f.attrs["k_hard"] = K_HARD
        f.attrs["Re_star"] = RE_STAR

        for ri, Re in enumerate(RE_DETAIL):
            for si in SRC_INDICES:
                S_np = sources_np[si]
                S_t = torch.tensor(S_np, dtype=torch.float64, device=device)
                key = f"Re{Re}_src{si:02d}"
                g = f.create_group(key)
                g.create_dataset("S", data=S_np.astype(np.float32))

                # Benchmark
                ref = solve_reference(S_np, Re, kx_np, ky_np, ikx_np, iky_np,
                                      lap_sym_np, inv_ksq_np, dealias_np)
                g.create_dataset("benchmark", data=ref.astype(np.float32))
                print(f"  Re={Re} src{si}: benchmark done")

                # All models
                for name, (model, info) in loaded.items():
                    if info["type"] == "canonical":
                        omega, _ = recast_aitken_for_canonical(
                            model, S_t, float(Re), RE_STAR, can_stats, K_HARD,
                            Kx64, Ky64, K2_inv64, device)
                    elif info["type"] in ("parametric", "pino"):
                        omega = fno_predict_omega(
                            model, S_t, par_stats["S_min"], par_stats["S_max"],
                            par_stats["omega_min"], par_stats["omega_max"],
                            Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                            Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"],
                            in_channels=2, device=device)
                    elif info["type"] == "pino_ext":
                        omega = fno_ext_predict_omega(
                            model, S_t, par_stats["S_min"], par_stats["S_max"],
                            par_stats["omega_min"], par_stats["omega_max"],
                            Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                            Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"], device=device)
                    g.create_dataset(name, data=omega.detach().cpu().numpy().astype(np.float32))

                # Print quick summary
                err_msg = []
                for name in loaded:
                    pred = g[name][:]
                    e = float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))
                    err_msg.append(f"{name[:8]}={e*100:.1f}%")
                print(f"    {key}: " + " ".join(err_msg))

    print(f"[Done] Saved {out_h5} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
