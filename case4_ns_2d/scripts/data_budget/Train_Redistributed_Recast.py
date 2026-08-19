#!/usr/bin/env python3
"""
Train_Redistributed_Recast.py — Train canonical-style FNO on redistributed
(multi-Re) data normalized to Re*=250 via equation recast.

For each training pair (S_i, u_i, Re_i) generated at Re_i:
  S_eff_i = S_i + (1/Re_i - 1/Re_star) * Δω_i
The model learns the canonical inverse: S_eff_i → ω_i.
"""
from __future__ import annotations
import os, time, h5py, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from FNO2D import FNO2d


RE_STAR = 250.0
N = 128


def laplacian_np(field, n=128):
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    K2 = -(2 * np.pi) ** 2 * (kx ** 2 + ky ** 2)
    Fh = np.fft.fft2(field)
    return np.real(np.fft.ifft2(K2 * Fh))


def make_effective_sources(S, O, Re_vec):
    """For each sample, compute S_eff = S + (1/Re - 1/Re*) * Δω."""
    S_eff = np.zeros_like(S)
    for i in range(S.shape[0]):
        lap_o = laplacian_np(O[i], n=S.shape[1])
        S_eff[i] = S[i] + (1.0 / Re_vec[i] - 1.0 / RE_STAR) * lap_o
    return S_eff


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=None,
                   help="total samples; drawn evenly as budget/5 per Re block. "
                        "None = use the full file (1500).")
    p.add_argument("--n_re", type=int, default=5,
                   help="number of fixed-Re blocks in the redistributed file")
    p.add_argument("--canonical_data", default="canonical/data_canonical_N1500.h5",
                   help="for normalization stats consistent with canonical training")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[Redistributed] data={args.data} out={args.out}")

    with h5py.File(args.data, "r") as f:
        S_raw = f["S"][:]; O_raw = f["omega"][:]; Re_sc = f["Re_scalar"][:]

    # Subsample budget/n_re samples from each contiguous fixed-Re block so the
    # redistributed budget stays balanced across all Reynolds numbers.
    if args.budget is not None:
        n_total_file = S_raw.shape[0]
        block = n_total_file // args.n_re          # e.g. 1500/5 = 300
        per = args.budget // args.n_re             # e.g. 200/5 = 40
        if per < 1:
            raise ValueError(f"budget {args.budget} too small for {args.n_re} Re blocks")
        idx = np.concatenate([np.arange(b * block, b * block + per)
                              for b in range(args.n_re)])
        S_raw = S_raw[idx]; O_raw = O_raw[idx]; Re_sc = Re_sc[idx]
        print(f"  budget={args.budget}: {per} samples x {args.n_re} Re blocks "
              f"= {S_raw.shape[0]} total")

    # Use canonical normalization stats for consistency
    if os.path.exists(args.canonical_data):
        with h5py.File(args.canonical_data, "r") as f:
            s_min = float(f.attrs["S_global_min"]); s_max = float(f.attrs["S_global_max"])
            o_min = float(f.attrs["omega_global_min"]); o_max = float(f.attrs["omega_global_max"])
    else:
        s_min = float(S_raw.min()); s_max = float(S_raw.max())
        o_min = float(O_raw.min()); o_max = float(O_raw.max())

    # Compute effective sources (S + (1/Re - 1/Re*) * Δω)
    print(f"  Computing effective sources for {S_raw.shape[0]} samples...")
    S_eff = make_effective_sources(S_raw, O_raw, Re_sc).astype(np.float32)

    # Normalize using canonical S stats (effective sources span a similar range)
    def norm(x, lo, hi):
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    # Re-fit S range to effective sources global range for stable training
    se_min = float(S_eff.min()); se_max = float(S_eff.max())
    print(f"  S_eff range: [{se_min:.3e}, {se_max:.3e}]")
    X = torch.tensor(norm(S_eff, se_min, se_max), dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(norm(O_raw, o_min, o_max), dtype=torch.float32).unsqueeze(1)

    # Shuffle (samples are grouped by Re)
    n_total = X.shape[0]
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    X = X[perm]; Y = Y[perm]

    ntr = int(0.9 * n_total)
    tr = DataLoader(TensorDataset(X[:ntr], Y[:ntr]), batch_size=args.batch_size,
                    shuffle=True, pin_memory=True)
    vl = DataLoader(TensorDataset(X[ntr:], Y[ntr:]), batch_size=args.batch_size,
                    shuffle=False, pin_memory=True)

    model = FNO2d(modes_x=32, modes_y=32, width=64, in_channels=1,
                  out_channels=1, n_layers=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                     patience=50, min_lr=1e-6)
    mse = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad = 0
    t0 = time.time(); final_ep = 0
    train_hist, val_hist = [], []
    for ep in range(args.epochs):
        model.train()
        s = 0.0; n = 0
        for xb, yb in tr:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = mse(model(xb), yb)
            loss.backward(); opt.step()
            s += loss.item() * xb.size(0); n += xb.size(0)
        train_hist.append(s / n)
        model.eval()
        s = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                s += mse(model(xb), yb).item() * xb.size(0); n += xb.size(0)
        val = s / n; val_hist.append(val)
        sch.step(val)
        if val < best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        final_ep = ep + 1
        if (ep + 1) % 50 == 0 or ep == 0:
            print(f"  ep {ep+1}/{args.epochs} train {train_hist[-1]:.4e} val {val:.4e} bad {bad}")
        if bad >= args.patience:
            print(f"  early stop {ep+1}")
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": best_state, "modes": 32, "width": 64, "n_layers": 4,
                "in_channels": 1, "out_channels": 1, "Re_star": RE_STAR,
                "S_min": se_min, "S_max": se_max,   # NOTE: trained on effective-source range
                "omega_min": o_min, "omega_max": o_max,
                "best_val": best_val, "train_hist": train_hist, "val_hist": val_hist,
                "epochs": final_ep, "stage": "redistributed_recast",
                "budget": args.budget if args.budget is not None else int(S_raw.shape[0]),
                "data": args.data}, args.out)
    print(f"  Saved {args.out} best_val={best_val:.4e} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
