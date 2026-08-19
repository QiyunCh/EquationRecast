#!/usr/bin/env python3
"""
Test_AitkenAnderson.py — Compare iteration schemes for canonical NS recast.

Same canonical FNO (PINN-finetuned if available, else data-only). For each
Re ∈ [50, 400], for each source, run the recast fixed-point under three
relaxation schemes:
  - Aitken Δ² (adaptive)
  - Anderson(m=3) (history-based)
  - Under-relaxation, fixed ω = 0.5

Metrics: iterations to converge (or max_iters), wall-time, final relative
residual against ETDRK4 reference, final PDE residual.
"""
from __future__ import annotations
import os, time, h5py
from collections import deque
import numpy as np
import torch

from FNO2D import FNO2d
from Train_PINN_Canonical import make_kgrids
import VorticityNS_2D as ns


CKPT_PRIMARY = "models/best_fno2d_canonical_pinn.pt"
CKPT_FALLBACK = "models/best_fno2d_canonical_dataonly.pt"

RE_LIST = list(range(50, 401, 25))  # coarser sweep for ablation
RE_STAR = 250.0
N = 128
L_DOMAIN = 1.0
N_SOURCES = 20
SEED = 13
K_HARD = 21

TOL = 1e-5
MAX_ITERS = 200
EPS = 1e-12
ANDERSON_M = 3

MODES = 32
WIDTH = 64
N_LAYERS = 4


def load_canonical_model(device):
    ckpt = CKPT_PRIMARY if os.path.exists(CKPT_PRIMARY) else CKPT_FALLBACK
    print(f"  using {ckpt}")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = FNO2d(modes_x=MODES, modes_y=MODES, width=WIDTH, in_channels=1,
                  out_channels=1, n_layers=N_LAYERS).to(device)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(state); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def load_stats():
    with h5py.File("data_canonical.h5", "r") as f:
        return {
            "S_min": float(f.attrs["S_global_min"]),
            "S_max": float(f.attrs["S_global_max"]),
            "omega_min": float(f.attrs["omega_global_min"]),
            "omega_max": float(f.attrs["omega_global_max"]),
        }


def make_bandlimit_mask(Kx, Ky, k_hard, device):
    # Kx, Ky from make_kgrids are 2π * mode index; bandlimit uses mode-index units
    two_pi = 2.0 * float(np.pi)
    k_rad = ((Kx / two_pi) ** 2 + (Ky / two_pi) ** 2).sqrt()
    return (k_rad <= float(k_hard)).to(torch.complex128)


def apply_bl(field, mask):
    Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
    return torch.fft.irfft2(Fh * mask, s=field.shape[-2:], dim=(-2, -1), norm="backward")


def laplacian(field, Kx, Ky):
    Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
    return torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Fh, s=field.shape[-2:], dim=(-2, -1), norm="backward")


def fno_predict_omega(model, S_phys, stats):
    s_n = 2.0 * (S_phys - stats["S_min"]) / (stats["S_max"] - stats["S_min"]) - 1.0
    x = s_n.unsqueeze(0).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        out = model(x).squeeze().to(torch.float64)
    omega = 0.5 * (out + 1.0) * (stats["omega_max"] - stats["omega_min"]) + stats["omega_min"]
    return omega


def run_scheme(scheme: str, model, S_phys, Re_t, Re_s, stats, mask, Kx, Ky, device):
    """Returns (omega, iters, wall_time)."""
    t0 = time.perf_counter()
    omega = fno_predict_omega(model, S_phys, stats)
    omega = apply_bl(omega, mask)

    iters = MAX_ITERS
    if scheme == "aitken":
        omega_relax = 0.35
        r_prev = None
        for it in range(1, MAX_ITERS + 1):
            S_eff = S_phys + (1.0 / Re_t - 1.0 / Re_s) * laplacian(omega, Kx, Ky)
            S_eff = apply_bl(S_eff, mask)
            omega_hat = fno_predict_omega(model, S_eff, stats)
            omega_hat = apply_bl(omega_hat, mask)
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
            omega = cand_omega
            r_prev = r
            if rel < TOL:
                iters = it; break
    elif scheme == "underrelax":
        alpha = 0.5
        for it in range(1, MAX_ITERS + 1):
            S_eff = S_phys + (1.0 / Re_t - 1.0 / Re_s) * laplacian(omega, Kx, Ky)
            S_eff = apply_bl(S_eff, mask)
            omega_hat = fno_predict_omega(model, S_eff, stats)
            omega_hat = apply_bl(omega_hat, mask)
            cand_omega = omega + alpha * (omega_hat - omega)
            rel = float((cand_omega - omega).norm().item() / (omega.norm().item() + EPS))
            omega = cand_omega
            if rel < TOL:
                iters = it; break
    elif scheme == "anderson":
        # Anderson(m=ANDERSON_M) on residual r_k = G(x_k) - x_k
        m = ANDERSON_M
        X_hist = deque(maxlen=m + 1)
        F_hist = deque(maxlen=m + 1)
        x = omega.flatten()
        X_hist.append(x.clone())
        for it in range(1, MAX_ITERS + 1):
            x_field = x.reshape(N, N)
            S_eff = S_phys + (1.0 / Re_t - 1.0 / Re_s) * laplacian(x_field, Kx, Ky)
            S_eff = apply_bl(S_eff, mask)
            gx = fno_predict_omega(model, S_eff, stats)
            gx = apply_bl(gx, mask).flatten()
            F_hist.append(gx - x)

            mk = min(m, len(F_hist) - 1)
            if mk == 0:
                x_new = gx
            else:
                # Build matrix of residual differences ΔF_k = F[i+1] - F[i]
                Fs = list(F_hist)
                Xs = list(X_hist)
                DF = torch.stack([Fs[i + 1] - Fs[i] for i in range(mk)], dim=1)
                DX = torch.stack([Xs[i + 1] - Xs[i] for i in range(mk)], dim=1)
                # Solve least squares γ = argmin || F_k + DF γ ||
                F_k = Fs[-1]
                try:
                    gamma = torch.linalg.lstsq(DF, F_k.unsqueeze(1)).solution.squeeze(-1)
                    x_new = gx - (DX + DF) @ gamma
                except Exception:
                    x_new = gx

            rel = float((x_new - x).norm().item() / (x.norm().item() + EPS))
            X_hist.append(x_new.clone())
            x = x_new
            if rel < TOL:
                iters = it; break
        omega = x.reshape(N, N)
    else:
        raise ValueError(scheme)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return omega, iters, elapsed


def solve_reference(S_np, Re, kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias):
    saved = ns.RE
    ns.RE = float(Re)
    try:
        omega, _, _, _, _ = ns.solve_steady_vorticity(
            S_np, ikx, iky, lap_symbol, inv_ksq, dealias)
    finally:
        ns.RE = saved
    return omega


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Aitken/Anderson] device={device}")
    model, ckpt = load_canonical_model(device)
    stats = load_stats()

    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    Kx64 = Kx.double(); Ky64 = Ky.double()
    mask = make_bandlimit_mask(Kx64, Ky64, K_HARD, device)

    rng = np.random.default_rng(SEED)
    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    sources_np = np.stack(
        [ns.generate_source(N, kx_np, ky_np, dealias_np, rng) for _ in range(N_SOURCES)],
        axis=0
    )

    schemes = ("aitken", "anderson", "underrelax")
    iters_all = {s: np.zeros((len(RE_LIST), N_SOURCES), dtype=np.int32) for s in schemes}
    time_all = {s: np.zeros((len(RE_LIST), N_SOURCES)) for s in schemes}
    err_all = {s: np.zeros((len(RE_LIST), N_SOURCES)) for s in schemes}
    nonconv = {s: np.zeros((len(RE_LIST), N_SOURCES), dtype=np.int32) for s in schemes}

    t0 = time.time()
    for ri, Re in enumerate(RE_LIST):
        for si in range(N_SOURCES):
            S_np = sources_np[si]
            S_t = torch.tensor(S_np, dtype=torch.float64, device=device)
            ref = solve_reference(S_np, Re, kx_np, ky_np, ikx_np, iky_np,
                                  lap_sym_np, inv_ksq_np, dealias_np)
            for scheme in schemes:
                omega, iters, dt = run_scheme(scheme, model, S_t, float(Re), RE_STAR,
                                              stats, mask, Kx64, Ky64, device)
                iters_all[scheme][ri, si] = iters
                time_all[scheme][ri, si] = dt
                err_all[scheme][ri, si] = float(np.linalg.norm(omega.cpu().numpy() - ref) /
                                                (np.linalg.norm(ref) + 1e-12))
                if iters >= MAX_ITERS:
                    nonconv[scheme][ri, si] = 1
        msg = " ".join(f"{s[:3]}=({iters_all[s][ri].mean():.0f}it,{err_all[s][ri].mean():.2e})"
                       for s in schemes)
        print(f"  Re={Re:3d} | {msg}")

    os.makedirs("results", exist_ok=True)
    out_h5 = "results/test3_aitken_vs_anderson.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Re_list", data=np.array(RE_LIST))
        for s in schemes:
            g = f.create_group(s)
            g.create_dataset("iters", data=iters_all[s])
            g.create_dataset("time", data=time_all[s])
            g.create_dataset("err_l2", data=err_all[s])
            g.create_dataset("nonconv", data=nonconv[s])
        f.attrs["ckpt"] = ckpt
        f.attrs["Re_star"] = RE_STAR
        f.attrs["n_sources"] = N_SOURCES
        f.attrs["seed"] = SEED
        f.attrs["k_hard"] = K_HARD
        f.attrs["tol"] = TOL
        f.attrs["max_iters"] = MAX_ITERS
        f.attrs["anderson_m"] = ANDERSON_M
    print(f"[Done] Saved {out_h5}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
