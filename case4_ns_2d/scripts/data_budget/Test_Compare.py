#!/usr/bin/env python3
"""
Test_Compare.py — Compare models across data budgets on Re∈[50,400].

Loads all trained checkpoints in models/, tests on the same 20 sources
(seed=13) over the same Re list, computes rel L2 error vs ETDRK4 reference
and PDE residual. Saves results/test4_compare.h5.
"""
from __future__ import annotations
import os, time, h5py, glob
import numpy as np
import torch

from FNO2D import FNO2d
from Train_Canonical_PINN import make_kgrids
import VorticityNS_2D as ns


RE_LIST = list(range(50, 401, 25))
RE_STAR = 250.0
N = 128
L_DOMAIN = 1.0
N_SOURCES = 20
SEED = 13
K_HARD = 21
TOL = 1e-5
MAX_ITERS = 200
EPS = 1e-12
AITKEN_INIT = 0.35
AITKEN_MIN = 0.02
AITKEN_MAX = 0.85
BUDGETS = [200, 500, 1000, 1500]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if isinstance(ck, dict) and ck.get("stage") == "pino_ext":
        import sys, os as _os
        sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "ext_pino")))
        from models import FNO2d as ExtFNO2d
        model = ExtFNO2d(modes1=ck["modes1"], modes2=ck["modes2"],
                         width=ck["width"], fc_dim=ck["fc_dim"], layers=ck["layers"],
                         in_dim=ck["in_dim"], out_dim=ck["out_dim"], act=ck["act"]).to(device)
    else:
        in_ch = ck.get("in_channels", 1)
        model = FNO2d(modes_x=ck.get("modes", 32), modes_y=ck.get("modes", 32),
                      width=ck.get("width", 64), in_channels=in_ch, out_channels=1,
                      n_layers=ck.get("n_layers", 4)).to(device)
    model.load_state_dict(state); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def fno_ext_pino_predict(model, S_t, Re, stats, device):
    """ext-PINO inference: input (B, X, Y, 4) = (S, Re_field, x, y)."""
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]
    re_min, re_max = stats["Re_min"], stats["Re_max"]
    N_ = S_t.shape[-1]
    s_n = (2.0 * (S_t - s_min) / (s_max - s_min) - 1.0).to(torch.float32)
    re_n = (2.0 * (Re - re_min) / (re_max - re_min) - 1.0).item()
    xs = torch.linspace(-1, 1, N_, dtype=torch.float32, device=device)
    ys = torch.linspace(-1, 1, N_, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    re_field = torch.full_like(s_n, float(re_n))
    x = torch.stack([s_n, re_field, X, Y], dim=-1).unsqueeze(0)
    with torch.no_grad():
        o = model(x).squeeze(0).squeeze(-1).to(torch.float64)
    return 0.5 * (o + 1.0) * (o_max - o_min) + o_min


def fno_canonical_recast(model, S_t, Re_target, Re_star, stats, Kx, Ky, mask, device):
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]

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

    omega = predict(S_t); omega = apply_bl(omega)
    omega_relax = AITKEN_INIT
    r_prev = None
    iters = MAX_ITERS
    for it in range(1, MAX_ITERS + 1):
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
                omega_relax = float(torch.clamp(cand, AITKEN_MIN, AITKEN_MAX).item())
        cand_omega = omega + omega_relax * r
        rel = float((cand_omega - omega).norm().item() / (omega.norm().item() + EPS))
        omega = cand_omega; r_prev = r
        if rel < TOL:
            iters = it; break
    return omega, iters


def fno_parametric_predict(model, S_t, Re, stats, device):
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]
    re_min, re_max = stats["Re_min"], stats["Re_max"]
    s_n = 2.0 * (S_t - s_min) / (s_max - s_min) - 1.0
    re_n = 2.0 * (Re - re_min) / (re_max - re_min) - 1.0
    re_field = torch.full_like(s_n, float(re_n))
    x = torch.stack([s_n, re_field], dim=0).unsqueeze(0).to(torch.float32)
    with torch.no_grad():
        o = model(x).squeeze().to(torch.float64)
    return 0.5 * (o + 1.0) * (o_max - o_min) + o_min


def compute_residual(omega, S, Re, Kx, Ky, K2_inv):
    o = omega.unsqueeze(0); s = S.unsqueeze(0)
    Oh = torch.fft.rfft2(o, dim=(-2, -1), norm="backward")
    psi_h = Oh * K2_inv
    u = torch.fft.irfft2((1j * Ky) * psi_h, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    v = torch.fft.irfft2(-(1j * Kx) * psi_h, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    om_x = torch.fft.irfft2((1j * Kx) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    om_y = torch.fft.irfft2((1j * Ky) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    lap_om = torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Oh, s=o.shape[-2:], dim=(-2, -1), norm="backward")
    R = u * om_x + v * om_y - (1.0 / float(Re)) * lap_om - s
    return float(R.flatten().norm().item() / (s.flatten().norm().item() + EPS))


def solve_reference(S_np, Re, kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias):
    saved = ns.RE; ns.RE = float(Re)
    try:
        omega, _, _, _, _ = ns.solve_steady_vorticity(S_np, ikx, iky, lap_symbol, inv_ksq, dealias)
    finally:
        ns.RE = saved
    return omega


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test4 Compare] device={device}")

    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    Kx64 = Kx.double(); Ky64 = Ky.double(); K2_inv64 = K2_inv.double()
    two_pi = 2.0 * float(np.pi)
    k_rad = ((Kx64 / two_pi) ** 2 + (Ky64 / two_pi) ** 2).sqrt()
    mask = (k_rad <= float(K_HARD)).to(torch.complex128)

    rng = np.random.default_rng(SEED)
    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    sources_np = np.stack(
        [ns.generate_source(N, kx_np, ky_np, dealias_np, rng) for _ in range(N_SOURCES)],
        axis=0
    )

    # Pre-compute references (once per (Re, source))
    print("  computing ETDRK4 references...")
    refs = {}
    t_ref = time.time()
    for ri, Re in enumerate(RE_LIST):
        for si in range(N_SOURCES):
            refs[(Re, si)] = solve_reference(sources_np[si], Re, kx_np, ky_np, ikx_np, iky_np,
                                              lap_sym_np, inv_ksq_np, dealias_np)
        if (ri + 1) % 5 == 0:
            print(f"    {ri+1}/{len(RE_LIST)} Re's, t={time.time()-t_ref:.0f}s")
    print(f"  References done, {time.time()-t_ref:.0f}s")

    # Discover models from models/ directory; group by budget and type
    model_files = sorted(glob.glob("models/**/*.pt", recursive=True))
    print(f"  found {len(model_files)} model files")

    err = {}; res = {}; iters_all = {}
    for ckpt in model_files:
        name = os.path.basename(ckpt).replace(".pt", "")
        model, ck = load_model(ckpt, device)
        stage = ck.get("stage", "data")
        budget = ck.get("budget", 0)
        stats = {"S_min": ck["S_min"], "S_max": ck["S_max"],
                 "omega_min": ck["omega_min"], "omega_max": ck["omega_max"]}
        if "Re_min" in ck:
            stats["Re_min"] = ck["Re_min"]; stats["Re_max"] = ck["Re_max"]
        is_canonical_like = (stage in ("data", "pinn_finetune", "redistributed_recast"))
        is_pino_ext = (stage == "pino_ext")

        print(f"  testing {name} (stage={stage} budget={budget})")
        e = np.zeros((len(RE_LIST), N_SOURCES)); r = np.zeros((len(RE_LIST), N_SOURCES))
        its = np.zeros((len(RE_LIST), N_SOURCES), dtype=np.int32)

        t0 = time.time()
        for ri, Re in enumerate(RE_LIST):
            for si in range(N_SOURCES):
                S_np = sources_np[si]
                S_t = torch.tensor(S_np, dtype=torch.float64, device=device)
                ref = refs[(Re, si)]
                if is_canonical_like:
                    omega, it = fno_canonical_recast(model, S_t, float(Re), RE_STAR, stats,
                                                     Kx64, Ky64, mask, device)
                    its[ri, si] = it
                elif is_pino_ext:
                    omega = fno_ext_pino_predict(
                        model, S_t,
                        torch.tensor(float(Re), dtype=torch.float64, device=device),
                        stats, device)
                else:
                    omega = fno_parametric_predict(
                        model, S_t,
                        torch.tensor(float(Re), dtype=torch.float64, device=device),
                        stats, device)
                e[ri, si] = float(np.linalg.norm(omega.cpu().numpy() - ref) / (np.linalg.norm(ref) + 1e-12))
                r[ri, si] = compute_residual(omega, S_t, Re, Kx64, Ky64, K2_inv64)
        print(f"    elapsed {time.time()-t0:.0f}s, mean err {e.mean():.3e}")
        err[name] = e; res[name] = r; iters_all[name] = its

    os.makedirs("results", exist_ok=True)
    out_h5 = "results/test4_compare.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Re_list", data=np.array(RE_LIST))
        f.create_dataset("sources", data=sources_np)
        for name in err:
            g = f.create_group(name)
            g.create_dataset("err_l2", data=err[name])
            g.create_dataset("res_rel", data=res[name])
            g.create_dataset("iters", data=iters_all[name])
        f.attrs["budgets"] = BUDGETS
        f.attrs["Re_star"] = RE_STAR
        f.attrs["n_sources"] = N_SOURCES
        f.attrs["seed"] = SEED
        f.attrs["k_hard"] = K_HARD
    print(f"[Done] Saved {out_h5}")


if __name__ == "__main__":
    main()
