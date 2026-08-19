#!/usr/bin/env python3
"""
Train_Stage1.py — Pure data training of canonical FNO at (Pe*, Da*).

Stage-1 of the canonical-point ablation. Trains an FNO1d on the (S, u) pairs
generated at the given canonical, with MSE loss in normalized u-space.

Usage:
    python Train_Stage1.py <data_h5> <out_ckpt>
"""
from __future__ import annotations
import os, sys, h5py, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from FNO1D import FNO1d


def u_norm(u, umin, umax):
    return 2.0 * (u - umin) / (umax - umin) - 1.0


def train_stage1(data_h5: str, out_ckpt: str,
                 epochs: int = 1000, batch_size: int = 200,
                 lr: float = 1e-3, modes: int = 64, width: int = 64,
                 seed: int = 1234) -> dict:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    with h5py.File(data_h5, "r") as f:
        S_raw = f["source"][:]
        U_raw = f["solution"][:]
        umin = float(f.attrs["solution_min"])
        umax = float(f.attrs["solution_max"])
        pe_star = float(f.attrs["Pe"])
        da_star = float(f.attrs["Da"])

    X = torch.tensor(S_raw, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(u_norm(U_raw, umin, umax), dtype=torch.float32).unsqueeze(1)

    n_total = X.shape[0]
    ntrain = int(0.90 * n_total)
    X_tr, X_va = X[:ntrain], X[ntrain:]
    Y_tr, Y_va = Y[:ntrain], Y[ntrain:]

    tl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(X_va, Y_va), batch_size=batch_size, shuffle=False, pin_memory=True)

    model = FNO1d(modes=modes, width=width, in_channels=1, out_channels=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None
    t0 = time.time()
    train_hist, val_hist = [], []
    for ep in range(epochs):
        model.train()
        s = 0.0; n = 0
        for xb, yb in tl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            s += loss.item() * xb.size(0); n += xb.size(0)
        train_hist.append(s / n)

        model.eval()
        s = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                s += loss_fn(model(xb), yb).item() * xb.size(0); n += xb.size(0)
        val = s / n; val_hist.append(val)

        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        sch.step()
        if (ep + 1) % 100 == 0 or ep == 0 or ep == epochs - 1:
            print(f"[Pe*={pe_star},Da*={da_star}] ep {ep+1}/{epochs} train {train_hist[-1]:.4e} val {val:.4e}")

    torch.save({"state_dict": best_state, "modes": modes, "width": width,
                "u_min": umin, "u_max": umax, "Pe_star": pe_star, "Da_star": da_star,
                "best_val": best_val, "train_hist": train_hist, "val_hist": val_hist,
                "stage": "data", "epochs": epochs}, out_ckpt)
    print(f"  Saved {out_ckpt}  best_val={best_val:.4e}  elapsed={time.time()-t0:.1f}s")
    return {"best_val": best_val, "ckpt": out_ckpt}


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        train_stage1(sys.argv[1], sys.argv[2])
    else:
        canonicals = [(2.0, 4.0), (10.0, 10.0), (2.0, 25.0), (25.0, 2.0)]
        for pe, da in canonicals:
            dh5 = f"data/data_Pe{pe:g}_Da{da:g}.h5"
            ckpt = f"models/fno_Pe{pe:g}_Da{da:g}_stage1.pt"
            train_stage1(dh5, ckpt)
