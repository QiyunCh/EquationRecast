#!/usr/bin/env python3
"""
Train_parametric.py

Train a parametric 2D FNO: input = (S, Re_field) -> output = omega.
Re is sampled from U[30, 90] per sample and fed as a 2D constant field
(second input channel).

Designed for fair comparison with the fixed-Re FNO:
- Same architecture (modes, width, layers), only in_channels = 2 instead of 1
- Same optimizer, scheduler, epochs, batch size, loss, save logic
- Same normalization strategy (global min/max -> [-1, 1])

Expected HDF5 format (from VorticityNS_2D_parametric.py):
  Datasets: "S", "omega", "Re"  (all shape N x Ny x Nx)
  Attributes: S_global_min/max, omega_global_min/max, Re_global_min/max
"""

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from time import time

from FNO2D import FNO2d


# -----------------------------
# User settings
# -----------------------------
DATA_PATH = "data_parametric.h5"  # Re~U[200,300] dataset
RESULTS_DIR = "Results_Train_Parametric"

EPOCHS = 10000
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0

TRAIN_FRACTION = 0.90
SEED = 1029

MODES_X = 32
MODES_Y = 32
WIDTH = 64

SAVE_EVERY = 200

USE_AMP = False

EARLY_STOP_PATIENCE = 500


# -----------------------------
# Helpers
# -----------------------------
def normalize_to_minus1_1(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    denom = x_max - x_min
    if denom <= 0:
        raise ValueError(f"Invalid min/max: min={x_min}, max={x_max}")
    return 2.0 * (x - x_min) / denom - 1.0


def denormalize_from_minus1_1(xn: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return 0.5 * (xn + 1.0) * (x_max - x_min) + x_min


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.linalg.norm(a.ravel() - b.ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return float(num / den)


def _get_attr_float(attrs, keys):
    for k in keys:
        if k in attrs:
            return float(attrs[k])
    return None


def save_snapshot_figures(epoch, model, device, X_val, Y_val_true_raw, smin, smax, umin, umax, out_dir):
    model.eval()
    n = X_val.shape[0]
    idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    idxs = sorted(set([i for i in idxs if 0 <= i < n]))

    with torch.no_grad():
        xb = X_val[idxs].to(device)
        pred_norm = model(xb).cpu().numpy()

    pred_raw = denormalize_from_minus1_1(pred_norm[:, 0], umin, umax)

    for j, idx in enumerate(idxs):
        true_raw = Y_val_true_raw[idx]
        err = pred_raw[j] - true_raw
        r2 = rel_l2(pred_raw[j], true_raw)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        im0 = axes[0].imshow(true_raw, origin="lower")
        axes[0].set_title(f"True (raw) | idx={idx}")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(pred_raw[j], origin="lower")
        axes[1].set_title(f"Pred (raw) | rel L2={r2:.3e}")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(np.abs(err), origin="lower")
        axes[2].set_title("Abs Error (raw)")
        axes[2].set_xticks([]); axes[2].set_yticks([])
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig_path = os.path.join(out_dir, f"epoch_{epoch:04d}_valsample_{idx:04d}.png")
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)


def save_loss_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="Train MSE")
    plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label="Val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (normalized [-1,1])")
    plt.title("Parametric FNO2d Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "snapshots"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Device:", device)
    print("Data file:", DATA_PATH)

    # -----------------------------
    # Load data
    # -----------------------------
    with h5py.File(DATA_PATH, "r") as f:
        S_raw = f["S"][:]
        U_raw = f["omega"][:]
        Re_raw = f["Re"][:]

        smin = float(f.attrs["S_global_min"])
        smax = float(f.attrs["S_global_max"])
        umin = float(f.attrs["omega_global_min"])
        umax = float(f.attrs["omega_global_max"])
        re_min = float(f.attrs["Re_global_min"])
        re_max = float(f.attrs["Re_global_max"])

    if S_raw.ndim != 3 or U_raw.ndim != 3 or Re_raw.ndim != 3:
        raise ValueError(f"Expected 3D arrays. Got S:{S_raw.shape}, U:{U_raw.shape}, Re:{Re_raw.shape}")

    n_total, Ny, Nx = S_raw.shape
    print(f"Loaded: N={n_total}, Ny={Ny}, Nx={Nx}")
    print(f"Source  global min/max: {smin:.6e} / {smax:.6e}")
    print(f"Solution global min/max: {umin:.6e} / {umax:.6e}")
    print(f"Re global min/max: {re_min:.6e} / {re_max:.6e}")

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    S_norm = normalize_to_minus1_1(S_raw.astype(np.float32), smin, smax)
    U_norm = normalize_to_minus1_1(U_raw.astype(np.float32), umin, umax)
    Re_norm = normalize_to_minus1_1(Re_raw.astype(np.float32), re_min, re_max)

    X = np.stack([S_norm, Re_norm], axis=1)
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(U_norm, dtype=torch.float32).unsqueeze(1)

    U_raw_np = U_raw.astype(np.float32)

    # -----------------------------
    # Train/val split
    # -----------------------------
    ntrain = int(TRAIN_FRACTION * n_total)
    X_train, X_val = X[:ntrain], X[ntrain:]
    Y_train, Y_val = Y[:ntrain], Y[ntrain:]
    U_val_true_raw = U_raw_np[ntrain:]

    print(f"Split: ntrain={X_train.shape[0]}, nval={X_val.shape[0]}")

    # -----------------------------
    # Dataloaders
    # -----------------------------
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=0
    )

    # -----------------------------
    # Model: in_channels=2 (S + Re)
    # -----------------------------
    model = FNO2d(
        modes_x=MODES_X, modes_y=MODES_Y, width=WIDTH,
        in_channels=2, out_channels=1
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and device.type == "cuda")
    )

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    early_stop_counter = 0

    # -----------------------------
    # Training loop
    # -----------------------------
    t0 = time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        train_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            train_sum += loss.item() * xb.size(0)

        train_mse = train_sum / len(X_train)
        train_losses.append(train_mse)

        # Validate
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_sum += loss_fn(pred, yb).item() * xb.size(0)

        val_mse = val_sum / len(X_val)
        val_losses.append(val_mse)

        # Track best & early stopping
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(RESULTS_DIR, "best_fno2d_parametric.pt"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        scheduler.step(val_mse)

        if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {ep:4d}/{EPOCHS} | Train {train_mse:.6e} | Val {val_mse:.6e} | lr {lr_now:.3e} | ES {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        if (ep % SAVE_EVERY == 0) or (ep == EPOCHS):
            snap_dir = os.path.join(RESULTS_DIR, "snapshots", f"epoch_{ep:04d}")
            os.makedirs(snap_dir, exist_ok=True)

            save_loss_curve(train_losses, val_losses, os.path.join(snap_dir, "loss_curve.png"))

            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(snap_dir, "model_state.pt"))

            save_snapshot_figures(ep, model, device, X_val, U_val_true_raw,
                                 smin, smax, umin, umax, snap_dir)

            np.savez(os.path.join(snap_dir, "loss_history.npz"),
                     train=np.array(train_losses, dtype=np.float64),
                     val=np.array(val_losses, dtype=np.float64))

        # Early stopping
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {ep} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            snap_dir = os.path.join(RESULTS_DIR, "snapshots", f"epoch_{ep:04d}_early_stop")
            os.makedirs(snap_dir, exist_ok=True)
            save_loss_curve(train_losses, val_losses, os.path.join(snap_dir, "loss_curve.png"))
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(snap_dir, "model_state.pt"))
            save_snapshot_figures(ep, model, device, X_val, U_val_true_raw,
                                 smin, smax, umin, umax, snap_dir)
            np.savez(os.path.join(snap_dir, "loss_history.npz"),
                     train=np.array(train_losses, dtype=np.float64),
                     val=np.array(val_losses, dtype=np.float64))
            break

    dt = time() - t0
    print(f"Training done in {dt:.2f}s")
    print(f"Best val MSE (normalized): {best_val:.8e}")

    if best_state is not None:
        best_name = os.path.join(RESULTS_DIR, f"best_fno2d_parametric_valmse_{best_val:.8e}.pt")
        torch.save(best_state, best_name)
        print("Best model saved to:", best_name)

    save_loss_curve(train_losses, val_losses, os.path.join(RESULTS_DIR, "loss_curve.png"))

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            pred_norm = []
            for i in range(0, X_val.shape[0], BATCH_SIZE):
                xb = X_val[i:i+BATCH_SIZE].to(device)
                pred_norm.append(model(xb).cpu().numpy())
            pred_norm = np.concatenate(pred_norm, axis=0)

        pred_raw = denormalize_from_minus1_1(pred_norm[:, 0], umin, umax)
        errs = [rel_l2(pred_raw[i], U_val_true_raw[i]) for i in range(pred_raw.shape[0])]
        print(f"Val rel L2 (raw scale): mean={np.mean(errs):.6e}, std={np.std(errs):.6e}")


if __name__ == "__main__":
    main()