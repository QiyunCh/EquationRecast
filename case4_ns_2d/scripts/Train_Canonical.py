#!/usr/bin/env python3
"""
Train2D.py

Train a 2D Fourier Neural Operator (FNO2d) for operator learning:
  input  : source field S(x,y)
  output : solution field omega(x,y) (or whatever "solution" dataset is)

Key features (per request):
- Read HDF5 data
- Normalize source and solution using SAVED global min/max in file attributes
- 1000 epochs, batch size = 64
- No checkpointing each epoch
- Save results every 200 epochs (loss curves + a few prediction figures + model snapshot)
- Save the best model (lowest val MSE)

Expected HDF5 format (robustly handled):
- Prefer datasets: "S" and "omega"
- Fallback: "source" and "solution"
- Attributes prefer: "<name>_global_min/max" then "<name>_min/max" then "solution_min/max"

Author: (you)
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
DATA_PATH = "data_canonical.h5"  # canonical Re*=250 dataset
RESULTS_DIR = "Results_Train_Canonical"

EPOCHS = 10000
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.0

TRAIN_FRACTION = 0.90
SEED = 1029

# Model hyperparameters (adjust as you like)
MODES_X = 32
MODES_Y = 32
WIDTH = 64

# Save every K epochs
SAVE_EVERY = 200

# Mixed precision (safe default: False). Turn on if you want speed on modern GPUs.
USE_AMP = False

# Early stopping
EARLY_STOP_PATIENCE = 500  # stop if no val improvement for this many epochs

# -----------------------------
# Helpers: normalization
# -----------------------------
def normalize_to_minus1_1(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Map x from [x_min, x_max] to [-1, 1]."""
    denom = (x_max - x_min)
    if denom <= 0:
        raise ValueError(f"Invalid min/max for normalization: min={x_min}, max={x_max}")
    return 2.0 * (x - x_min) / denom - 1.0


def denormalize_from_minus1_1(xn: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Map xn from [-1, 1] back to [x_min, x_max]."""
    return 0.5 * (xn + 1.0) * (x_max - x_min) + x_min


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 error ||a-b||2 / ||b||2."""
    num = np.linalg.norm(a.ravel() - b.ravel())
    den = np.linalg.norm(b.ravel()) + eps
    return float(num / den)


def _get_attr_float(attrs, keys):
    for k in keys:
        if k in attrs:
            return float(attrs[k])
    return None


def infer_dataset_keys(h5: h5py.File):
    """
    Try common dataset naming conventions.
    Returns (source_key, solution_key).
    """
    if "S" in h5 and "omega" in h5:
        return "S", "omega"
    if "source" in h5 and "solution" in h5:
        return "source", "solution"
    # fallback: pick first two datasets if obvious
    keys = [k for k in h5.keys() if isinstance(h5[k], h5py.Dataset)]
    if len(keys) >= 2:
        # heuristic: prefer keys containing 'source' and 'sol'
        src = None
        sol = None
        for k in keys:
            kl = k.lower()
            if src is None and ("source" in kl or k.lower() == "s"):
                src = k
            if sol is None and ("solution" in kl or "omega" in kl):
                sol = k
        if src is None:
            src = keys[0]
        if sol is None:
            sol = keys[1] if keys[1] != src else keys[0]
        return src, sol

    raise KeyError("Could not infer dataset keys in the HDF5 file.")


def infer_minmax(h5: h5py.File, source_key: str, sol_key: str):
    attrs = h5.attrs

    smin = _get_attr_float(attrs, [f"{source_key}_global_min", f"{source_key}_min",
                                  "S_global_min", "S_min", "source_global_min", "source_min"])
    smax = _get_attr_float(attrs, [f"{source_key}_global_max", f"{source_key}_max",
                                  "S_global_max", "S_max", "source_global_max", "source_max"])

    umin = _get_attr_float(attrs, [f"{sol_key}_global_min", f"{sol_key}_min",
                                  "omega_global_min", "omega_min", "solution_global_min", "solution_min"])
    umax = _get_attr_float(attrs, [f"{sol_key}_global_max", f"{sol_key}_max",
                                  "omega_global_max", "omega_max", "solution_global_max", "solution_max"])

    if smin is None or smax is None:
        raise KeyError("Could not find source global min/max in HDF5 attributes.")
    if umin is None or umax is None:
        raise KeyError("Could not find solution global min/max in HDF5 attributes.")

    return smin, smax, umin, umax


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
    plt.title("FNO2d Training / Validation Loss")
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
        src_key, sol_key = infer_dataset_keys(f)
        print(f"Using datasets: source='{src_key}', solution='{sol_key}'")

        S_raw = f[src_key][:]
        U_raw = f[sol_key][:]

        smin, smax, umin, umax = infer_minmax(f, src_key, sol_key)

    if S_raw.ndim != 3 or U_raw.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N, Ny, Nx). Got S:{S_raw.shape}, U:{U_raw.shape}")

    n_total, Ny, Nx = S_raw.shape
    print(f"Loaded: N={n_total}, Ny={Ny}, Nx={Nx}")
    print(f"Source global min/max: {smin:.6e} / {smax:.6e}")
    print(f"Solution global min/max: {umin:.6e} / {umax:.6e}")

    # -----------------------------
    # Normalize to [-1, 1]
    # -----------------------------
    S_norm = normalize_to_minus1_1(S_raw.astype(np.float32), smin, smax)
    U_norm = normalize_to_minus1_1(U_raw.astype(np.float32), umin, umax)

    X = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(1)
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
    # Model
    # -----------------------------
    model = FNO2d(modes_x=MODES_X, modes_y=MODES_Y, width=WIDTH, in_channels=1, out_channels=1).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ReduceLROnPlateau: halve LR when val loss plateaus for 50 epochs, floor at 1e-7
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
        # Train
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
            torch.save(best_state, os.path.join(RESULTS_DIR, "best_fno2d.pt"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Step scheduler with val loss
        scheduler.step(val_mse)

        # Logging
        if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {ep:4d}/{EPOCHS} | Train {train_mse:.6e} | Val {val_mse:.6e} | lr {lr_now:.3e} | ES {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        # Save periodic results
        if (ep % SAVE_EVERY == 0) or (ep == EPOCHS):
            snap_dir = os.path.join(RESULTS_DIR, "snapshots", f"epoch_{ep:04d}")
            os.makedirs(snap_dir, exist_ok=True)

            save_loss_curve(train_losses, val_losses, os.path.join(snap_dir, "loss_curve.png"))

            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(snap_dir, "model_state.pt"))

            save_snapshot_figures(ep, model, device, X_val, U_val_true_raw, smin, smax, umin, umax, snap_dir)

            np.savez(os.path.join(snap_dir, "loss_history.npz"),
                     train=np.array(train_losses, dtype=np.float64),
                     val=np.array(val_losses, dtype=np.float64))

        # Early stopping check
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {ep} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            # Save final snapshot before breaking
            snap_dir = os.path.join(RESULTS_DIR, "snapshots", f"epoch_{ep:04d}_early_stop")
            os.makedirs(snap_dir, exist_ok=True)
            save_loss_curve(train_losses, val_losses, os.path.join(snap_dir, "loss_curve.png"))
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                       os.path.join(snap_dir, "model_state.pt"))
            save_snapshot_figures(ep, model, device, X_val, U_val_true_raw, smin, smax, umin, umax, snap_dir)
            np.savez(os.path.join(snap_dir, "loss_history.npz"),
                     train=np.array(train_losses, dtype=np.float64),
                     val=np.array(val_losses, dtype=np.float64))
            break

    dt = time() - t0
    print(f"Training done in {dt:.2f}s")
    print(f"Best val MSE (normalized): {best_val:.8e}")

    if best_state is not None:
        best_name = os.path.join(RESULTS_DIR, f"best_fno2d_valmse_{best_val:.8e}.pt")
        torch.save(best_state, best_name)
        print("Best model saved to:", best_name)

    save_loss_curve(train_losses, val_losses, os.path.join(RESULTS_DIR, "loss_curve.png"))

    # Report mean/std rel L2 on val set using BEST model
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