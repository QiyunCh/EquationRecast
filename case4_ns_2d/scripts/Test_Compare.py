#!/usr/bin/env python3
"""
Test_Compare.py — Compare 4 models on NS Re sweep [50, 400].

Models:
  (1) canonical_dataonly:  Version1 canonical FNO at Re*=250 (data only)
  (2) canonical_pinn:      same + PINN finetune (this Version2)
  (3) parametric:          Version1 parametric FNO trained on Re~U[200,300]
  (4) pino:                PINO (parametric + λ=0.5 residual loss), this Version2

For each (Re, source), report:
  - relative L2 error vs Fourier-spectral IMEX reference (omega)
  - relative PDE residual ||R||_2 / ||S||_2

Outputs results/test3_compare.h5 + per-model error/residual curves.
"""
from __future__ import annotations
import os, time, h5py
import numpy as np
import torch

from FNO2D import FNO2d
from Train_PINN_Canonical import make_kgrids
import VorticityNS_2D as ns


RE_LIST = list(range(10, 401, 10))  # 10..400 step 10
RE_STAR = 250.0

N = 128
L_DOMAIN = 1.0
SEED_SOURCES = 13
N_SOURCES = 20
K_HARD = 21

# Aitken Δ²
AITKEN_INIT = 0.35
AITKEN_MIN = 0.02
AITKEN_MAX = 0.85
TOL = 1e-5
MAX_ITERS = 200
EPS = 1e-12

MODELS = {
    "canonical_dataonly": {
        "ckpt": "models/best_fno2d_canonical_dataonly.pt",
        "type": "canonical", "in_channels": 1,
    },
    "canonical_pinn": {
        "ckpt": "models/best_fno2d_canonical_pinn.pt",
        "type": "canonical", "in_channels": 1,
    },
    "parametric": {
        "ckpt": "models/best_fno2d_parametric.pt",
        "type": "parametric", "in_channels": 2,
        "Re_train_range": (200.0, 300.0),
    },
    "pino": {
        "ckpt": "models/best_fno2d_pino.pt",
        "type": "pino", "in_channels": 2,
        "Re_train_range": (200.0, 300.0),
    },
    # pino_ext (external neuraloperator/physics_informed FNO2d) archived to
    # legacy/ — not part of the article's results. Leaving it out of MODELS
    # makes Test_Compare / Test_MatchedAccuracyTime / Test_FieldSpectrum
    # pino_ext-free (the ext_pino import is gated by this entry).
}

MODES = 32
WIDTH = 64
N_LAYERS = 4


def load_norm_stats():
    """Load normalization stats from canonical and parametric datasets."""
    with h5py.File("data_canonical.h5", "r") as f:
        c = {"S_min": float(f.attrs["S_global_min"]), "S_max": float(f.attrs["S_global_max"]),
             "omega_min": float(f.attrs["omega_global_min"]), "omega_max": float(f.attrs["omega_global_max"])}
    with h5py.File("data_parametric.h5", "r") as f:
        p = {"S_min": float(f.attrs["S_global_min"]), "S_max": float(f.attrs["S_global_max"]),
             "omega_min": float(f.attrs["omega_global_min"]), "omega_max": float(f.attrs["omega_global_max"]),
             "Re_min": float(f.attrs["Re_global_min"]), "Re_max": float(f.attrs["Re_global_max"])}
    return c, p


def gen_test_sources(seed=SEED_SOURCES, n=N_SOURCES):
    rng = np.random.default_rng(seed)
    kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias = ns.make_spectral_operators(N)
    sources = [ns.generate_source(N, kx, ky, dealias, rng) for _ in range(n)]
    return np.stack(sources, axis=0)  # (n, N, N)


def solve_reference(S_np, Re, kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias):
    """Fourier-spectral IMEX pseudo-time solver (same as data gen)."""
    saved_re = ns.RE
    ns.RE = float(Re)
    try:
        omega, ux, uy, nsteps, relres = ns.solve_steady_vorticity(
            S_np, ikx, iky, lap_symbol, inv_ksq, dealias
        )
    finally:
        ns.RE = saved_re
    return omega


def load_model(name, info, device):
    ck = torch.load(info["ckpt"], map_location="cpu", weights_only=False)
    in_ch = info["in_channels"]
    if info["type"] == "pino_ext":
        # Use ext_pino's FNO2d
        import sys, os as _os
        sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "ext_pino")))
        from models import FNO2d as ExtFNO2d
        model = ExtFNO2d(modes1=ck["modes1"], modes2=ck["modes2"],
                         width=ck["width"], fc_dim=ck["fc_dim"], layers=ck["layers"],
                         in_dim=ck["in_dim"], out_dim=ck["out_dim"], act=ck["act"]).to(device)
    else:
        model = FNO2d(modes_x=MODES, modes_y=MODES, width=WIDTH, in_channels=in_ch,
                      out_channels=1, n_layers=N_LAYERS).to(device)
    if isinstance(ck, dict) and "state_dict" in ck:
        state = ck["state_dict"]
    else:
        state = ck
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def fno_ext_predict_omega(model, S_phys, S_min, S_max, o_min, o_max,
                          Re, Re_min, Re_max, device):
    """ext-PINO forward: input (B, X, Y, 4) = (S, Re_field, x_coord, y_coord)."""
    N_ = S_phys.shape[-1]
    s_n = 2.0 * (S_phys - S_min) / (S_max - S_min) - 1.0
    re_n = 2.0 * (Re - Re_min) / (Re_max - Re_min) - 1.0
    xs = torch.linspace(-1, 1, N_, dtype=torch.float32, device=device)
    ys = torch.linspace(-1, 1, N_, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    re_field = torch.full_like(s_n, float(re_n))
    x = torch.stack([s_n.to(torch.float32), re_field.to(torch.float32),
                     X, Y], dim=-1).unsqueeze(0)  # (1, N, N, 4)
    with torch.no_grad():
        out = model(x).squeeze(0).squeeze(-1).to(torch.float64)
    omega = 0.5 * (out + 1.0) * (o_max - o_min) + o_min
    return omega


def fno_predict_omega(model, S_phys, S_min, S_max, o_min, o_max, Re=None,
                      Re_min=None, Re_max=None, in_channels=1, device=None):
    """Single FNO forward + denormalize."""
    s_n = 2.0 * (S_phys - S_min) / (S_max - S_min) - 1.0
    if in_channels == 1:
        x = s_n.unsqueeze(0).unsqueeze(0).to(torch.float32)
    else:
        re_n = 2.0 * (Re - Re_min) / (Re_max - Re_min) - 1.0
        re_field = torch.full_like(s_n, float(re_n))
        x = torch.stack([s_n, re_field], dim=0).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        out = model(x).squeeze().to(torch.float64)
    omega = 0.5 * (out + 1.0) * (o_max - o_min) + o_min
    return omega


def recast_aitken_for_canonical(model, S_phys, Re_target, Re_star, stats, k_hard,
                                Kx, Ky, K2_inv, device):
    """Canonical FNO + Aitken Δ² recast: S_eff = S + (1/Re_target - 1/Re*) Δω."""
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]

    # Bandlimit mask matching K_HARD in mode-index units (Kx is 2π * mode index, so divide by 2π)
    two_pi = 2.0 * float(np.pi)
    k_rad = ((Kx / two_pi) ** 2 + (Ky / two_pi) ** 2).sqrt()
    mask = (k_rad <= float(k_hard)).to(torch.complex128)

    def apply_bl(field):
        Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(Fh * mask, s=field.shape[-2:], dim=(-2, -1), norm="backward")

    def lap(field):
        Fh = torch.fft.rfft2(field, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Fh, s=field.shape[-2:], dim=(-2, -1), norm="backward")

    # Initial guess from canonical FNO on raw S
    omega = fno_predict_omega(model, S_phys, s_min, s_max, o_min, o_max,
                              in_channels=1, device=device)
    omega = apply_bl(omega)

    omega_relax = AITKEN_INIT
    r_prev = None
    iters = MAX_ITERS
    for it in range(1, MAX_ITERS + 1):
        S_eff = S_phys + (1.0 / Re_target - 1.0 / Re_star) * lap(omega)
        S_eff_bl = apply_bl(S_eff)
        omega_hat = fno_predict_omega(model, S_eff_bl, s_min, s_max, o_min, o_max,
                                      in_channels=1, device=device)
        omega_hat = apply_bl(omega_hat)
        r = omega_hat - omega

        if r_prev is not None:
            dr = r - r_prev
            num = (r_prev * dr).sum()
            den = (dr * dr).sum().clamp(min=EPS)
            cand = -omega_relax * num / den
            if torch.isfinite(cand):
                omega_relax = float(torch.clamp(cand, AITKEN_MIN, AITKEN_MAX).item())
        cand_omega = omega + omega_relax * r
        rel = float((cand_omega - omega).norm().item() / (omega.norm().item() + EPS))
        omega = cand_omega
        r_prev = r
        if rel < TOL:
            iters = it
            break
    return omega, iters


def compute_residual(omega, S_phys, Re, Kx, Ky, K2_inv):
    """Relative PDE residual: ||u·∇ω - (1/Re)Δω - S|| / ||S||."""
    o = omega.unsqueeze(0)
    s = S_phys.unsqueeze(0)
    Oh = torch.fft.rfft2(o, dim=(-2, -1), norm="backward")
    psi_h = Oh * K2_inv
    u_h = (1j * Ky) * psi_h
    v_h = -(1j * Kx) * psi_h
    u = torch.fft.irfft2(u_h, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    v = torch.fft.irfft2(v_h, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    om_x = torch.fft.irfft2((1j * Kx) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    om_y = torch.fft.irfft2((1j * Ky) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    lap_om = torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    R = u * om_x + v * om_y - (1.0 / float(Re)) * lap_om - s
    return float(R.flatten().norm().item() / (s.flatten().norm().item() + EPS))


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED_SOURCES); np.random.seed(SEED_SOURCES)
    print(f"[Compare] device={device}")

    can_stats, par_stats = load_norm_stats()

    # Load models
    models = {}
    for name, info in MODELS.items():
        if not os.path.exists(info["ckpt"]):
            print(f"  SKIP {name}: {info['ckpt']} not found")
            continue
        models[name] = load_model(name, info, device)
        print(f"  loaded {name}: {info['ckpt']}")

    # Pre-compute torch buffers
    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    # convert to float64 for residual/recast
    Kx64 = Kx.double(); Ky64 = Ky.double(); K2_inv64 = K2_inv.double()

    # Test sources
    rng_master = np.random.default_rng(SEED_SOURCES)
    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    sources_np = np.stack(
        [ns.generate_source(N, kx_np, ky_np, dealias_np, rng_master) for _ in range(N_SOURCES)],
        axis=0
    )

    # Buffers for outputs
    err = {m: np.zeros((len(RE_LIST), N_SOURCES)) for m in models}
    res = {m: np.zeros((len(RE_LIST), N_SOURCES)) for m in models}
    iters_recast = np.zeros((len(RE_LIST), N_SOURCES), dtype=np.int32)

    t0 = time.time()
    for ri, Re in enumerate(RE_LIST):
        for si in range(N_SOURCES):
            S_np = sources_np[si]
            S_t = torch.tensor(S_np, dtype=torch.float64, device=device)
            # Reference
            ref = solve_reference(S_np, Re, kx_np, ky_np, ikx_np, iky_np,
                                  lap_sym_np, inv_ksq_np, dealias_np)

            for name in models:
                info = MODELS[name]
                if info["type"] == "canonical":
                    omega, it = recast_aitken_for_canonical(
                        models[name], S_t, float(Re), RE_STAR, can_stats, K_HARD,
                        Kx64, Ky64, K2_inv64, device)
                    if name == "canonical_pinn":
                        iters_recast[ri, si] = it
                    err[name][ri, si] = rel_l2(omega.detach().cpu().numpy(), ref)
                    res[name][ri, si] = compute_residual(omega, S_t, Re, Kx64, Ky64, K2_inv64)
                elif info["type"] in ("parametric", "pino"):
                    omega = fno_predict_omega(
                        models[name], S_t, par_stats["S_min"], par_stats["S_max"],
                        par_stats["omega_min"], par_stats["omega_max"],
                        Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                        Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"],
                        in_channels=2, device=device)
                    err[name][ri, si] = rel_l2(omega.detach().cpu().numpy(), ref)
                    res[name][ri, si] = compute_residual(omega, S_t, Re, Kx64, Ky64, K2_inv64)
                elif info["type"] == "pino_ext":
                    omega = fno_ext_predict_omega(
                        models[name], S_t, par_stats["S_min"], par_stats["S_max"],
                        par_stats["omega_min"], par_stats["omega_max"],
                        Re=torch.tensor(float(Re), dtype=torch.float64, device=device),
                        Re_min=par_stats["Re_min"], Re_max=par_stats["Re_max"], device=device)
                    err[name][ri, si] = rel_l2(omega.detach().cpu().numpy(), ref)
                    res[name][ri, si] = compute_residual(omega, S_t, Re, Kx64, Ky64, K2_inv64)

        msg = " ".join(f"{m[:6]}={err[m][ri].mean():.3e}" for m in models)
        print(f"  Re={Re:3d} | {msg}")

    out_h5 = "results/test3_compare.h5"
    os.makedirs("results", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Re_list", data=np.array(RE_LIST))
        f.create_dataset("sources", data=sources_np)
        f.create_dataset("recast_iters", data=iters_recast)
        for m in models:
            g = f.create_group(m)
            g.create_dataset("err_l2", data=err[m])
            g.create_dataset("res_rel", data=res[m])
        f.attrs["Re_star"] = RE_STAR
        f.attrs["n_sources"] = N_SOURCES
        f.attrs["seed"] = SEED_SOURCES
        f.attrs["k_hard"] = K_HARD
    print(f"[Done] Saved {out_h5}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
