#!/usr/bin/env python3
"""
Test_MatchedAccuracyTime.py — Fair inference-time comparison at matched
accuracy. Same test methods as Fig S9d (Test_InferenceTime).

For each (Re, source):
  1. Run all surrogate methods → get (time, L²) for each.
  2. Pick the WORST surrogate L² as the accuracy threshold T (== ceiling).
  3. For the numerical solver (CPU + GPU torch port) and the canonical FNO
     recast with Aitken, sweep tolerances/iteration caps to construct a
     time-vs-L² Pareto curve.
  4. Report each method's time at which it first reaches L² ≤ T.

Methods compared (same as Fig S9d):
  - canonical_dataonly recast (Aitken)        — variable iters/tolerance
  - canonical_pinn recast (Aitken)            — variable iters/tolerance
  - parametric FNO (single forward)
  - PINO custom (single forward)
  - PINO ext-repo (single forward)
  - numerical solver, CPU NumPy (variable tol)
  - numerical solver, GPU torch port (variable tol)
"""
from __future__ import annotations
import os, sys, time, h5py
import numpy as np
import torch

from FNO2D import FNO2d
from Train_PINN_Canonical import make_kgrids
from Test_Compare import (
    MODELS, load_model, load_norm_stats, solve_reference,
    fno_predict_omega, fno_ext_predict_omega,
)
from Test_SolverTime import gpu_solve_steady
import VorticityNS_2D as ns


RE_LIST = [50.0, 150.0, 250.0, 350.0, 400.0]
N_SOURCES = 5             # smaller for tractable sweep
SEED = 13
RE_STAR = 250.0
N = 128
L_DOMAIN = 1.0
K_HARD = 21
WARMUP = 5
RUNS = 10
EPS = 1e-12


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------- Reference: tight ETDRK4 ----------
def ref_solve(S_np, Re, kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np):
    saved = ns.RE; ns.RE = float(Re)
    saved_tol = ns.TOL_REL_RESID
    ns.TOL_REL_RESID = 1e-7    # tight reference
    try:
        omega, _, _, _, _ = ns.solve_steady_vorticity(
            S_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np)
    finally:
        ns.RE = saved; ns.TOL_REL_RESID = saved_tol
    return omega


# ---------- Numerical solver Pareto on CPU (numpy) ----------
def cpu_solver_pareto(S_np, Re, ref, tols, kx_np, ky_np, ikx_np, iky_np,
                      lap_sym_np, inv_ksq_np, dealias_np):
    """Run solver at each tol; return list of (time, L²)."""
    saved = ns.RE; saved_tol = ns.TOL_REL_RESID
    out = []
    try:
        ns.RE = float(Re)
        for tol in tols:
            ns.TOL_REL_RESID = float(tol)
            # warmup
            _ = ns.solve_steady_vorticity(S_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np)
            times = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                omega, _, _, n_steps, _ = ns.solve_steady_vorticity(
                    S_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np)
                times.append(time.perf_counter() - t0)
            err = float(np.linalg.norm(omega - ref) / (np.linalg.norm(ref) + 1e-12))
            out.append({"tol": tol, "time": float(np.median(times)), "err": err, "steps": int(n_steps)})
    finally:
        ns.RE = saved; ns.TOL_REL_RESID = saved_tol
    return out


# ---------- Numerical solver Pareto on GPU (torch port) ----------
def gpu_solver_pareto(S_np, Re, ref, tols, device):
    S = torch.tensor(S_np, dtype=torch.float64, device=device)
    out = []
    for tol in tols:
        # warmup
        _ = gpu_solve_steady(S, Re, device, tol=tol)
        sync()
        times = []
        for _ in range(RUNS):
            sync(); t0 = time.perf_counter()
            omega_t, n_steps, rel = gpu_solve_steady(S, Re, device, tol=tol)
            sync(); times.append(time.perf_counter() - t0)
        err = float(np.linalg.norm(omega_t.cpu().numpy() - ref) / (np.linalg.norm(ref) + 1e-12))
        out.append({"tol": tol, "time": float(np.median(times)), "err": err, "steps": int(n_steps)})
    return out


# ---------- Recast Aitken Pareto ----------
def recast_aitken_pareto(model, S_t, Re_target, Re_star, stats, k_hard,
                         Kx, Ky, device, tols, max_iters_default=200):
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]
    two_pi = 2.0 * float(np.pi)
    k_rad = ((Kx / two_pi) ** 2 + (Ky / two_pi) ** 2).sqrt()
    mask = (k_rad <= float(k_hard)).to(torch.complex128)

    def apply_bl(field):
        Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(Fh * mask, s=field.shape[-2:], dim=(-2, -1), norm="backward")

    def lap(field):
        Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Fh, s=field.shape[-2:], dim=(-2, -1), norm="backward")

    def predict(S_field):
        s_n = 2.0 * (S_field - s_min) / (s_max - s_min) - 1.0
        x = s_n.unsqueeze(0).unsqueeze(0).to(torch.float32)
        with torch.no_grad():
            o = model(x).squeeze().to(torch.float64)
        return 0.5 * (o + 1.0) * (o_max - o_min) + o_min

    def solve(tol):
        omega = predict(S_t); omega = apply_bl(omega)
        omega_relax = 0.35
        r_prev = None
        for it in range(1, max_iters_default + 1):
            S_eff = S_t + (1.0 / Re_target - 1.0 / Re_star) * lap(omega)
            S_eff = apply_bl(S_eff)
            omega_hat = predict(S_eff); omega_hat = apply_bl(omega_hat)
            r = omega_hat - omega
            if r_prev is not None:
                dr = r - r_prev
                num = (r_prev * dr).sum()
                den = (dr * dr).sum().clamp(min=EPS)
                cand = -omega_relax * num / den
                if torch.isfinite(cand):
                    omega_relax = float(torch.clamp(cand, 0.02, 0.85).item())
            cand_omega = omega + omega_relax * r
            rel = float((cand_omega - omega).norm().item() / (omega.norm().item() + EPS))
            omega = cand_omega; r_prev = r
            if rel < tol:
                return omega, it
        return omega, max_iters_default

    out = []
    for tol in tols:
        # warmup
        _ = solve(tol)
        sync()
        times = []
        for _ in range(RUNS):
            sync(); t0 = time.perf_counter()
            omega_t, n_it = solve(tol)
            sync(); times.append(time.perf_counter() - t0)
        err = float(np.linalg.norm(omega_t.cpu().numpy() - 0) / 1.0)  # placeholder, replaced after
        out.append({"tol": tol, "time": float(np.median(times)), "iters": n_it, "omega": omega_t})
    return out


# ---------- Single-forward surrogates ----------
def time_single_forward(forward_fn, runs=RUNS, warmup=WARMUP):
    for _ in range(warmup):
        _ = forward_fn()
    sync()
    times = []
    for _ in range(runs):
        sync(); t0 = time.perf_counter()
        out = forward_fn()
        sync(); times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MatchedAccuracyTime] device={device}")
    can_stats, par_stats = load_norm_stats()

    # Load all available models
    loaded = {}
    for name, info in MODELS.items():
        if not os.path.exists(info["ckpt"]):
            print(f"  SKIP {name}: {info['ckpt']} missing"); continue
        loaded[name] = (load_model(name, info, device), info)
        print(f"  loaded {name}")

    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    Kx64 = Kx.double(); Ky64 = Ky.double()

    rng = np.random.default_rng(SEED)
    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    sources_np = np.stack(
        [ns.generate_source(N, kx_np, ky_np, dealias_np, rng) for _ in range(N_SOURCES)],
        axis=0
    )

    # Tolerances to sweep for solver + recast (Pareto)
    SOLVER_TOLS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    RECAST_TOLS = [1e-2, 1e-3, 1e-4, 1e-5]

    out_h5 = "results/test3_matched_accuracy_time.h5"
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    with h5py.File(out_h5, "w") as fout:
        fout.create_dataset("Re_list", data=np.array(RE_LIST))
        fout.create_dataset("solver_tols", data=np.array(SOLVER_TOLS))
        fout.create_dataset("recast_tols", data=np.array(RECAST_TOLS))
        fout.attrs["n_sources"] = N_SOURCES
        fout.attrs["seed"] = SEED

        for ri, Re in enumerate(RE_LIST):
            for si in range(N_SOURCES):
                S_np = sources_np[si]
                S_t = torch.tensor(S_np, dtype=torch.float64, device=device)
                # Tight reference
                ref = ref_solve(S_np, Re, kx_np, ky_np, ikx_np, iky_np,
                                lap_sym_np, inv_ksq_np, dealias_np)
                ref_norm = np.linalg.norm(ref) + 1e-12

                key = f"Re{Re:.0f}_src{si:02d}"
                g = fout.create_group(key)

                # --- Single-forward surrogates ---
                for name, (model, info) in loaded.items():
                    if info["type"] == "canonical":
                        continue  # canonical models go through recast (Pareto path)
                    if info["type"] in ("parametric", "pino"):
                        def fwd(model=model, info=info, S_t=S_t, Re=Re):
                            return fno_predict_omega(
                                model, S_t, par_stats["S_min"], par_stats["S_max"],
                                par_stats["omega_min"], par_stats["omega_max"],
                                Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                                Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"],
                                in_channels=2, device=device)
                    elif info["type"] == "pino_ext":
                        def fwd(model=model, info=info, S_t=S_t, Re=Re):
                            return fno_ext_predict_omega(
                                model, S_t, par_stats["S_min"], par_stats["S_max"],
                                par_stats["omega_min"], par_stats["omega_max"],
                                Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                                Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"],
                                device=device)
                    else:
                        continue
                    t_med, omega_pred = time_single_forward(fwd)
                    err = float(np.linalg.norm(omega_pred.cpu().numpy() - ref) / ref_norm)
                    g.create_dataset(f"{name}_time_s", data=np.array([t_med]))
                    g.create_dataset(f"{name}_err",   data=np.array([err]))

                # --- Recast Aitken Pareto (canonical_dataonly + canonical_pinn) ---
                for name in ("canonical_dataonly", "canonical_pinn"):
                    if name not in loaded:
                        continue
                    model, _ = loaded[name]
                    rs = recast_aitken_pareto(model, S_t, float(Re), RE_STAR,
                                              can_stats, K_HARD, Kx64, Ky64, device,
                                              tols=RECAST_TOLS)
                    times = np.array([r["time"] for r in rs])
                    errs = np.array([
                        float(np.linalg.norm(r["omega"].cpu().numpy() - ref) / ref_norm)
                        for r in rs])
                    iters = np.array([r["iters"] for r in rs])
                    g.create_dataset(f"{name}_time_s", data=times)
                    g.create_dataset(f"{name}_err",   data=errs)
                    g.create_dataset(f"{name}_iters", data=iters)

                # --- Numerical solver Pareto (CPU + GPU) ---
                rs_cpu = cpu_solver_pareto(S_np, Re, ref, SOLVER_TOLS,
                                           kx_np, ky_np, ikx_np, iky_np,
                                           lap_sym_np, inv_ksq_np, dealias_np)
                g.create_dataset("solver_cpu_time_s",
                                 data=np.array([r["time"] for r in rs_cpu]))
                g.create_dataset("solver_cpu_err",
                                 data=np.array([r["err"] for r in rs_cpu]))
                g.create_dataset("solver_cpu_steps",
                                 data=np.array([r["steps"] for r in rs_cpu]))

                rs_gpu = gpu_solver_pareto(S_np, Re, ref, SOLVER_TOLS, device)
                g.create_dataset("solver_gpu_time_s",
                                 data=np.array([r["time"] for r in rs_gpu]))
                g.create_dataset("solver_gpu_err",
                                 data=np.array([r["err"] for r in rs_gpu]))
                g.create_dataset("solver_gpu_steps",
                                 data=np.array([r["steps"] for r in rs_gpu]))

                print(f"  Re={Re:.0f} src{si}:  "
                      f"CPU={rs_cpu[-1]['time']*1000:.0f}ms@{rs_cpu[-1]['err']*100:.2f}% "
                      f"GPU={rs_gpu[-1]['time']*1000:.0f}ms@{rs_gpu[-1]['err']*100:.2f}%")

    print(f"[Done] Saved {out_h5} elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
