#!/usr/bin/env python3
"""
Train_PINO.py — PINO on parametric NS data (data + λ·residual loss).
"""
from __future__ import annotations
import os, time, h5py, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from FNO2D import FNO2d
from Train_Canonical_PINN import make_kgrids


def ns_residual_per_sample(omega, S, Kx, Ky, K2_inv, Re_vec):
    Oh = torch.fft.rfft2(omega, dim=(-2, -1), norm="backward")
    psi_h = Oh * K2_inv
    u = torch.fft.irfft2((1j * Ky) * psi_h, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    v = torch.fft.irfft2(-(1j * Kx) * psi_h, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    om_x = torch.fft.irfft2((1j * Kx) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    om_y = torch.fft.irfft2((1j * Ky) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    lap_om = torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Oh, s=omega.shape[-2:], dim=(-2, -1), norm="backward")
    invRe = (1.0 / Re_vec).view(-1, 1, 1)
    R = u * om_x + v * om_y - invRe * lap_om - S
    Rn = R.flatten(1).norm(dim=-1)
    Sn = S.flatten(1).norm(dim=-1) + 1e-12
    return Rn / Sn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_res", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--seed", type=int, default=1029)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[PINO] data={args.data} budget={args.budget} out={args.out} lambda={args.lambda_res}")

    with h5py.File(args.data, "r") as f:
        S_raw = f["S"][:args.budget]; O_raw = f["omega"][:args.budget]
        Re_field = f["Re"][:args.budget]; Re_sc = f["Re_scalar"][:args.budget]
        s_min = float(f.attrs["S_global_min"]); s_max = float(f.attrs["S_global_max"])
        o_min = float(f.attrs["omega_global_min"]); o_max = float(f.attrs["omega_global_max"])
        re_min = float(f.attrs["Re_global_min"]); re_max = float(f.attrs["Re_global_max"])

    def norm(x, lo, hi):
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    S_n = norm(S_raw, s_min, s_max).astype(np.float32)
    O_n = norm(O_raw, o_min, o_max).astype(np.float32)
    Re_n = norm(Re_field, re_min, re_max).astype(np.float32)
    X = torch.tensor(np.stack([S_n, Re_n], axis=1), dtype=torch.float32)
    Y = torch.tensor(O_n, dtype=torch.float32).unsqueeze(1)
    S_phys = torch.tensor(S_raw, dtype=torch.float32)
    Re_t = torch.tensor(Re_sc, dtype=torch.float32)

    ntr = int(0.9 * X.shape[0])
    tr = DataLoader(TensorDataset(X[:ntr], Y[:ntr], S_phys[:ntr], Re_t[:ntr]),
                    batch_size=args.batch_size, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(X[ntr:], Y[ntr:], S_phys[ntr:], Re_t[ntr:]),
                    batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = FNO2d(modes_x=32, modes_y=32, width=64, in_channels=2,
                  out_channels=1, n_layers=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=50, min_lr=1e-6)
    mse = nn.MSELoss()

    Kx, Ky, K2_inv = make_kgrids(128, 1.0, device)
    o_half = 0.5 * (o_max - o_min); o_center = 0.5 * (o_max + o_min)

    def step_loss(xb, yb, S_b, Re_b):
        pred = model(xb).squeeze(1)
        L_d = mse(pred.unsqueeze(1), yb)
        omega_phys = pred * o_half + o_center
        L_r = ns_residual_per_sample(omega_phys, S_b, Kx, Ky, K2_inv, Re_b).mean()
        return L_d, L_r

    best_val = float("inf"); best_state = None; bad = 0
    t0 = time.time(); final_ep = 0
    train_hist, val_hist = [], []
    for ep in range(args.epochs):
        model.train()
        sd = 0.0; sr = 0.0; n = 0
        for xb, yb, S_b, Re_b in tr:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            S_b = S_b.to(device, non_blocking=True); Re_b = Re_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            L_d, L_r = step_loss(xb, yb, S_b, Re_b)
            loss = L_d + args.lambda_res * L_r
            loss.backward(); opt.step()
            sd += L_d.item() * xb.size(0); sr += L_r.item() * xb.size(0); n += xb.size(0)
        td = sd / n; tr_r = sr / n; train_hist.append((td, tr_r))

        model.eval()
        sd = 0.0; sr = 0.0; n = 0
        with torch.no_grad():
            for xb, yb, S_b, Re_b in vl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                S_b = S_b.to(device, non_blocking=True); Re_b = Re_b.to(device, non_blocking=True)
                L_d, L_r = step_loss(xb, yb, S_b, Re_b)
                sd += L_d.item() * xb.size(0); sr += L_r.item() * xb.size(0); n += xb.size(0)
        vd = sd / n; vr = sr / n; val = vd + args.lambda_res * vr
        val_hist.append((vd, vr))
        sch.step(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        final_ep = ep + 1
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  ep {ep+1}/{args.epochs} d {td:.4e} r {tr_r:.4e} | val_d {vd:.4e} val_r {vr:.4e} total {val:.4e} bad {bad}")
        if bad >= args.patience:
            print(f"  early stop {ep+1}")
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "modes": 32, "width": 64, "n_layers": 4,
                "in_channels": 2, "out_channels": 1,
                "S_min": s_min, "S_max": s_max, "omega_min": o_min, "omega_max": o_max,
                "Re_min": re_min, "Re_max": re_max, "lambda_residual": args.lambda_res,
                "best_val": best_val, "train_hist": train_hist, "val_hist": val_hist,
                "epochs": final_ep, "stage": "pino", "budget": args.budget,
                "data": args.data}, args.out)
    print(f"  Saved {args.out} best_val={best_val:.4e} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
