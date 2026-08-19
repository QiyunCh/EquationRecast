#!/usr/bin/env python3
"""
Train_PINO.py — Physics-Informed Neural Operator on parametric NS data.

Same data and architecture as the parametric FNO baseline (Re~U[200,300],
200 samples, 2-channel input [S, Re_field]), but training loss is:

    L = L_data(omega) + lambda * L_residual_relative

where L_residual = mean over batch of:
    || u·∇ω - (1/Re)Δω - S ||_2 / || S ||_2
evaluated at each sample's own Re. lambda = 0.5.

The (u, v) velocity is computed from omega via streamfunction in Fourier space.
"""
from __future__ import annotations
import os, time, h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from FNO2D import FNO2d
from Train_PINN_Canonical import make_kgrids


DATA_H5 = "data_parametric.h5"
OUT_CKPT = "models/best_fno2d_pino.pt"

L_DOMAIN = 1.0
N = 128

MODES = 32
WIDTH = 64
N_LAYERS = 4

EPOCHS = 2000
BATCH_SIZE = 16
LR = 1e-3
LAMBDA_RES = 0.5
PATIENCE_STOP = 500
SEED = 1029  # matches Version1 parametric


def ns_residual_per_sample(omega, S, Kx, Ky, K2_inv, Re_vec):
    """Compute residual L2 norm per sample, plus ||S||_2 per sample.

    omega, S: (B, N, N) physical units
    Re_vec: (B,) scalar Re per sample
    Returns: (B,) relative residual ||R||_2 / ||S||_2
    """
    Oh = torch.fft.rfft2(omega, dim=(-2, -1), norm="backward")
    psi_h = Oh * K2_inv
    u_h = (1j * Ky) * psi_h
    v_h = -(1j * Kx) * psi_h
    u = torch.fft.irfft2(u_h, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    v = torch.fft.irfft2(v_h, s=omega.shape[-2:], dim=(-2, -1), norm="backward")

    om_x = torch.fft.irfft2((1j * Kx) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    om_y = torch.fft.irfft2((1j * Ky) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    lap_om = torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")

    invRe = (1.0 / Re_vec).view(-1, 1, 1)
    R = u * om_x + v * om_y - invRe * lap_om - S
    Rn = R.flatten(1).norm(dim=-1)
    Sn = S.flatten(1).norm(dim=-1) + 1e-12
    return Rn / Sn


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"[PINO] device={device}")

    with h5py.File(DATA_H5, "r") as f:
        S_raw = f["S"][:]; O_raw = f["omega"][:]
        Re_field = f["Re"][:]    # (N_sam, N, N) constant per sample
        Re_scalar = f["Re_scalar"][:]   # (N_sam,)
        s_min = float(f.attrs["S_global_min"]); s_max = float(f.attrs["S_global_max"])
        o_min = float(f.attrs["omega_global_min"]); o_max = float(f.attrs["omega_global_max"])
        re_min = float(f.attrs["Re_global_min"]); re_max = float(f.attrs["Re_global_max"])
        print(f"  S range=[{s_min}, {s_max}]  omega range=[{o_min}, {o_max}]  Re range=[{re_min}, {re_max}]")

    def norm(x, lo, hi):
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    S_n = norm(S_raw, s_min, s_max).astype(np.float32)
    O_n = norm(O_raw, o_min, o_max).astype(np.float32)
    Re_n = norm(Re_field, re_min, re_max).astype(np.float32)
    X = torch.tensor(np.stack([S_n, Re_n], axis=1), dtype=torch.float32)  # (Nsam, 2, N, N)
    Y = torch.tensor(O_n, dtype=torch.float32).unsqueeze(1)               # (Nsam, 1, N, N)
    S_phys = torch.tensor(S_raw, dtype=torch.float32)
    Re_sc = torch.tensor(Re_scalar, dtype=torch.float32)

    n_total = X.shape[0]
    ntr = int(0.9 * n_total)
    tr_ds = TensorDataset(X[:ntr], Y[:ntr], S_phys[:ntr], Re_sc[:ntr])
    va_ds = TensorDataset(X[ntr:], Y[ntr:], S_phys[ntr:], Re_sc[ntr:])
    tl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    vl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = FNO2d(modes_x=MODES, modes_y=MODES, width=WIDTH, in_channels=2, out_channels=1,
                  n_layers=N_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=50, min_lr=1e-6)
    mse = nn.MSELoss()

    Kx, Ky, K2_inv = make_kgrids(N, L_DOMAIN, device)
    o_half = 0.5 * (o_max - o_min); o_center = 0.5 * (o_max + o_min)

    def step(xb, yb, S_b, Re_b):
        pred_n = model(xb).squeeze(1)
        L_d = mse(pred_n.unsqueeze(1), yb)
        omega_phys = pred_n * o_half + o_center
        L_r = ns_residual_per_sample(omega_phys, S_b, Kx, Ky, K2_inv, Re_b).mean()
        return L_d, L_r

    best_val = float("inf"); best_state = None; bad = 0
    t0 = time.time()
    train_hist, val_hist, lr_hist = [], [], []
    final_ep = 0
    for ep in range(EPOCHS):
        model.train()
        s_d = 0.0; s_r = 0.0; n = 0
        for xb, yb, S_b, Re_b in tl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            S_b = S_b.to(device, non_blocking=True); Re_b = Re_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            L_d, L_r = step(xb, yb, S_b, Re_b)
            loss = L_d + LAMBDA_RES * L_r
            loss.backward(); opt.step()
            s_d += L_d.item() * xb.size(0); s_r += L_r.item() * xb.size(0); n += xb.size(0)
        td = s_d / n; tr = s_r / n
        train_hist.append((td, tr))

        model.eval()
        s_d = 0.0; s_r = 0.0; n = 0
        with torch.no_grad():
            for xb, yb, S_b, Re_b in vl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                S_b = S_b.to(device, non_blocking=True); Re_b = Re_b.to(device, non_blocking=True)
                L_d, L_r = step(xb, yb, S_b, Re_b)
                s_d += L_d.item() * xb.size(0); s_r += L_r.item() * xb.size(0); n += xb.size(0)
        vd = s_d / n; vr = s_r / n
        val = vd + LAMBDA_RES * vr
        val_hist.append((vd, vr))
        lr_hist.append(opt.param_groups[0]["lr"])

        sch.step(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        final_ep = ep + 1
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep {ep+1}/{EPOCHS} train_d {td:.4e} train_r {tr:.4e} | val_d {vd:.4e} val_r {vr:.4e} | total {val:.4e} bad {bad} lr {opt.param_groups[0]['lr']:.2e}")
        if bad >= PATIENCE_STOP:
            print(f"  early stop at ep {ep+1}")
            break

    torch.save({"state_dict": best_state, "modes": MODES, "width": WIDTH, "n_layers": N_LAYERS,
                "in_channels": 2, "out_channels": 1,
                "S_min": s_min, "S_max": s_max, "omega_min": o_min, "omega_max": o_max,
                "Re_min": re_min, "Re_max": re_max,
                "lambda_residual": LAMBDA_RES,
                "best_val": best_val, "train_hist": train_hist, "val_hist": val_hist, "lr_hist": lr_hist,
                "epochs": final_ep, "stage": "pino"}, OUT_CKPT)
    print(f"  Saved {OUT_CKPT}  best_val={best_val:.4e}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
