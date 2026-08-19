#!/usr/bin/env python3
"""
Test_Scan.py — Recast scan over [1,50]^2 for a trained canonical FNO.

Loads a checkpoint (Stage 1 or Stage 2), reads its canonical (Pe*, Da*), and
runs the GPU-batched recast + bandlimit + Aitken Delta^2 fixed-point on a
target (Pe, Da) grid. Identical solver logic to Version1 ADR Test_Recast.py,
but Pe*/Da* come from the checkpoint and the scan range is configurable.

Usage:
    python Test_Scan.py <ckpt.pt> <out_h5> [PE_MIN PE_MAX PE_STEP DA_MIN DA_MAX DA_STEP]
"""
from __future__ import annotations
import os, sys, time, h5py
import numpy as np
import torch

from FNO1D import FNO1d


# Aitken Δ² — Version1 baseline but with lower OMEGA_MIN to allow heavier
# damping for hard cells in the wider [1,50]² scan range (vs Version1 [1,20]).
DAMP_INIT = 0.1       # Version1 default
OMEGA_MIN = 0.01      # lower than Version1's 0.05 — allows tighter damping
OMEGA_MAX = 1.5       # Version1 default — keep Aitken acceleration headroom
TOL = 1e-5
MAX_ITERS = 300       # extra headroom for wider scan range
EPS = 1e-12

L = 1.0
N = 201
SEED = 13
N_SOURCES = 20
AMP_FRAC = 1e-3


def sample_periodic_grf_se(rng, N, L, ell, sigma):
    n = np.arange(0, N // 2 + 1, dtype=np.float64)
    c = (sigma ** 2) * np.sqrt(2.0 * np.pi) * ell * np.exp(
        -(2.0 * (np.pi ** 2) * (ell ** 2) * (n ** 2)) / (L ** 2)
    )
    coeff = np.zeros(n.shape[0], dtype=np.complex128)
    coeff[0] = rng.normal(0.0, 1.0) * np.sqrt(c[0])
    if coeff.shape[0] > 2:
        re = rng.normal(0.0, 1.0, size=coeff.shape[0] - 2)
        im = rng.normal(0.0, 1.0, size=coeff.shape[0] - 2)
        coeff[1:-1] = (re + 1j * im) * np.sqrt(0.5 * c[1:-1])
    if N % 2 == 0:
        coeff[-1] = rng.normal(0.0, 1.0) * np.sqrt(c[-1])
    field = np.fft.irfft(coeff, n=N).astype(np.float64)
    field -= np.mean(field)
    return field


def rescale_pm1(S):
    smin = float(S.min()); smax = float(S.max())
    if np.isclose(smin, smax):
        return np.zeros_like(S)
    return 2.0 * (S - smin) / (smax - smin) - 1.0


def bandlimit_mask(S, amp_frac):
    Sh = np.fft.rfft(S)
    amp = np.abs(Sh)
    amax = float(np.max(amp))
    if not np.isfinite(amax) or amax <= 0.0:
        m = np.zeros_like(amp, dtype=bool); m[0] = True; return m
    m = amp >= amp_frac * amax; m[0] = True; return m


def rfft_k(N, L, device, dtype=torch.float64):
    freq = torch.fft.rfftfreq(N, d=L / N, dtype=dtype, device=device)
    return 2.0 * float(np.pi) * freq


def spec_deriv(u, k):
    uh = torch.fft.rfft(u, dim=-1)
    return torch.fft.irfft(1j * k * uh, n=u.shape[-1], dim=-1)


def spec_solve(S, Pe, Da, k):
    Sh = torch.fft.rfft(S, dim=-1)
    denom = (k ** 2) + 1j * float(Pe) * k + float(Da)
    return torch.fft.irfft(Sh / denom, n=S.shape[-1], dim=-1)


def apply_bandlimit(u, mask_c):
    uh = torch.fft.rfft(u, dim=-1)
    return torch.fft.irfft(uh * mask_c, n=u.shape[-1], dim=-1)


def fno_apply(model, rhs, umin, umax):
    xb = rhs.unsqueeze(1).to(torch.float32)
    with torch.no_grad():
        yb = model(xb)
    u_n = yb.squeeze(1).to(torch.float64)
    return 0.5 * (u_n + 1.0) * (float(umax) - float(umin)) + float(umin)


def fixed_point_aitken(model, S_batch, Pe, Da, Pe_star, Da_star, k, mask_c, umin, umax,
                       tol=TOL, max_iters=MAX_ITERS):
    B, _ = S_batch.shape
    device = S_batch.device
    dPe = float(Pe - Pe_star)
    dDa = float(Da - Da_star)

    u0 = fno_apply(model, S_batch, umin, umax)
    finite0 = torch.isfinite(u0).all(dim=-1)
    u = torch.where(finite0[:, None], u0, torch.zeros_like(u0))
    u = apply_bandlimit(u, mask_c)

    omega = torch.full((B,), DAMP_INIT, dtype=torch.float64, device=device)
    iters_first = torch.full((B,), max_iters, dtype=torch.long, device=device)
    relchg_last = torch.full((B,), float("nan"), dtype=torch.float64, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    failed = ~finite0

    r_prev = None
    for it in range(1, max_iters + 1):
        active = (~converged) & (~failed)
        if not bool(active.any().item()):
            break

        u_bl = apply_bandlimit(u, mask_c)
        ux = spec_deriv(u_bl, k)
        rhs = S_batch - dPe * ux - dDa * u_bl

        c = rhs.abs().amax(dim=-1).clamp(min=1e-12)
        rhs_in = rhs / c[:, None]
        u_hat_s = fno_apply(model, rhs_in, umin, umax)
        u_hat_s = apply_bandlimit(u_hat_s, mask_c)
        u_hat = c[:, None] * u_hat_s

        finite_hat = torch.isfinite(u_hat).all(dim=-1)
        failed = failed | (active & (~finite_hat))
        active = active & finite_hat
        if not bool(active.any().item()):
            break

        r = u_hat - u_bl
        if r_prev is not None:
            dr = r - r_prev
            num = (r_prev * dr).sum(dim=-1)
            den = (dr * dr).sum(dim=-1).clamp(min=EPS)
            cand = -omega * num / den
            valid = torch.isfinite(cand)
            omega = torch.where(valid & active, cand, omega)
            omega = torch.clamp(omega, OMEGA_MIN, OMEGA_MAX)

        candidate = u_bl + omega[:, None] * r
        candidate = apply_bandlimit(candidate, mask_c)

        diff = (candidate - u_bl).norm(dim=-1)
        denom_u = u_bl.norm(dim=-1) + EPS
        relchg = diff / denom_u
        relchg_last = torch.where(active, relchg, relchg_last)

        u = torch.where(active[:, None], candidate, u)
        newly = active & (relchg < tol)
        iters_first = torch.where(newly, torch.full_like(iters_first, it), iters_first)
        converged = converged | newly
        r_prev = r

    if bool(failed.any().item()):
        u = torch.where(failed[:, None], torch.full_like(u, float("nan")), u)
    return u, iters_first


def run_scan(ckpt_path: str, out_h5: str,
             pe_range=(1.0, 50.0, 1.0), da_range=(1.0, 50.0, 1.0),
             ell=0.08, sigma=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Scan] device={device}, ckpt={ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pe_star = float(ck["Pe_star"]); da_star = float(ck["Da_star"])
    umin = float(ck["u_min"]); umax = float(ck["u_max"])
    modes = ck.get("modes", 64); width = ck.get("width", 64)

    model = FNO1d(modes=modes, width=width, in_channels=1, out_channels=1).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(SEED)
    S_np = np.stack([rescale_pm1(sample_periodic_grf_se(rng, N, L, ell, sigma))
                     for _ in range(N_SOURCES)])  # (B, N)
    masks_np = np.stack([bandlimit_mask(s, AMP_FRAC) for s in S_np])  # (B, N//2+1) bool

    S_b = torch.tensor(S_np, dtype=torch.float64, device=device)
    mask_c = torch.tensor(masks_np.astype(np.float64), dtype=torch.complex128, device=device)
    k_t = rfft_k(N, L, device)

    pe_list = np.arange(pe_range[0], pe_range[1] + 0.5 * pe_range[2], pe_range[2], dtype=np.float64)
    da_list = np.arange(da_range[0], da_range[1] + 0.5 * da_range[2], da_range[2], dtype=np.float64)
    nPe, nDa = len(pe_list), len(da_list)

    rel_l2 = np.full((nDa, nPe), np.nan, dtype=np.float64)
    iters_mean = np.full((nDa, nPe), np.nan, dtype=np.float64)
    nonconv = np.zeros((nDa, nPe), dtype=np.int32)

    total = nPe * nDa
    cnt = 0
    t0 = time.time()
    for j, Da in enumerate(da_list):
        for i, Pe in enumerate(pe_list):
            cnt += 1
            u_ref = spec_solve(S_b, Pe, Da, k_t)
            u_pred, iters = fixed_point_aitken(model, S_b, float(Pe), float(Da),
                                               pe_star, da_star, k_t, mask_c, umin, umax)
            ok = torch.isfinite(u_pred).all(dim=-1)
            if ok.any():
                e = (u_pred - u_ref).norm(dim=-1) / (u_ref.norm(dim=-1) + EPS)
                e_ok = e[ok]
                rel_l2[j, i] = float(e_ok.mean().item())
                iters_mean[j, i] = float(iters[ok].float().mean().item())
            nonconv[j, i] = int((iters >= MAX_ITERS).sum().item())

            if cnt % 100 == 0 or cnt == total:
                eta = (time.time() - t0) / cnt * (total - cnt)
                print(f"  [{cnt}/{total}] Pe={Pe:.0f} Da={Da:.0f} "
                      f"relL2={rel_l2[j,i]:.3e} iters={iters_mean[j,i]:.1f} ETA={eta:.0f}s")

    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Pe_list", data=pe_list)
        f.create_dataset("Da_list", data=da_list)
        f.create_dataset("rel_l2", data=rel_l2)
        f.create_dataset("iters_mean", data=iters_mean)
        f.create_dataset("nonconv", data=nonconv)
        f.attrs["Pe_star"] = pe_star
        f.attrs["Da_star"] = da_star
        f.attrs["ckpt"] = ckpt_path
        f.attrs["n_sources"] = N_SOURCES
        f.attrs["seed"] = SEED
        f.attrs["tol"] = TOL
        f.attrs["max_iters"] = MAX_ITERS
    print(f"[Done] Saved {out_h5}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        ck = sys.argv[1]; out = sys.argv[2]
        if len(sys.argv) >= 9:
            pe_r = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
            da_r = (float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]))
        else:
            pe_r = (1.0, 50.0, 1.0); da_r = (1.0, 50.0, 1.0)
        run_scan(ck, out, pe_r, da_r)
    else:
        # Stage 1 only (data-trained); stage 2 PINN-finetune was found to hurt recast.
        canonicals = [(2.0, 4.0), (10.0, 10.0), (2.0, 25.0), (25.0, 2.0)]
        for pe, da in canonicals:
            ck = f"models/fno_Pe{pe:g}_Da{da:g}_stage1.pt"
            if os.path.exists(ck):
                out = f"results/scan_Pe{pe:g}_Da{da:g}_stage1.h5"
                run_scan(ck, out)
