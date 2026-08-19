import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from time import time
from FNO1D import FNO1d


# -----------------------------
# Setup
# -----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Reproducibility
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

# Folders
os.makedirs("Results", exist_ok=True)
os.makedirs("Results/checkpoints", exist_ok=True)


# -----------------------------
# Helpers: normalization
# -----------------------------
def u_normalize_to_minus1_1(u: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
    """Map u from [u_min, u_max] to [-1, 1]."""
    return 2.0 * (u - u_min) / (u_max - u_min) - 1.0


def u_denormalize_from_minus1_1(u_norm: np.ndarray, u_min: float, u_max: float) -> np.ndarray:
    """Map u from [-1, 1] back to [u_min, u_max]."""
    return 0.5 * (u_norm + 1.0) * (u_max - u_min) + u_min


def rel_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Relative L2 error ||a-b||2 / ||b||2 for flattened arrays."""
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(b) + eps
    return float(num / den)


# -----------------------------
# Load data (NEW ADR dataset)
# -----------------------------
path = "data_canonical.h5"

with h5py.File(path, "r") as f:
    S_raw = f["source"][:]       # (num_samples, 201) ; already normalized to [-1, 1]
    U_raw = f["solution"][:]     # (num_samples, 201) ; true u (not normalized)
    u_min = float(f.attrs["solution_min"])
    u_max = float(f.attrs["solution_max"])

print("Data file:", path)
print("u_min (global) =", u_min)
print("u_max (global) =", u_max)
print("S range:", float(S_raw.min()), float(S_raw.max()))
print("U range:", float(U_raw.min()), float(U_raw.max()))


# -----------------------------
# Tensors / shapes
#   Input X: 1 channel = [S]; shape (N, 1, 201)
#   Target Y: normalized U in [-1, 1]; shape (N, 1, 201)
# -----------------------------
S = torch.tensor(S_raw, dtype=torch.float32)                 # (N, 201)
X = S.unsqueeze(1)                                           # (N, 1, 201)

U_norm = u_normalize_to_minus1_1(U_raw, u_min, u_max)        # (N, 201) in [-1,1]
Y = torch.tensor(U_norm, dtype=torch.float32).unsqueeze(1)   # (N, 1, 201)


# -----------------------------
# Train-test split
# -----------------------------
n_total = X.shape[0]
ntrain = int(0.90 * n_total)

X_train, X_test = X[:ntrain], X[ntrain:]
Y_train, Y_test = Y[:ntrain], Y[ntrain:]

# Keep raw U for post-training evaluation/plots
U_test_true = U_raw[ntrain:]  # (Ntest, 201), numpy


# -----------------------------
# Dataloaders
# -----------------------------
batch_size = 200  # adjust as needed for your GPU/CPU memory

train_loader = DataLoader(
    TensorDataset(X_train, Y_train),
    batch_size=batch_size, shuffle=True, pin_memory=True
)
test_loader = DataLoader(
    TensorDataset(X_test, Y_test),
    batch_size=batch_size, shuffle=False, pin_memory=True
)


# -----------------------------
# Model, loss, optimizer, schedule
# -----------------------------
model = FNO1d(modes=64, width=64, in_channels=1, out_channels=1).to(device)

learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 1000
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

train_loss_list, test_loss_list = [], []
best_test_loss = float("inf")
best_model_state = None


# -----------------------------
# Training loop
# -----------------------------
t0 = time()
for ep in range(epochs):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= len(X_train)
    train_loss_list.append(train_loss)

    # Validation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            test_loss += loss_fn(pred, yb).item() * xb.size(0)

    test_loss /= len(X_test)
    test_loss_list.append(test_loss)

    # Track best
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_model_state, os.path.join("Results", "best_fno1d.pt"))

    scheduler.step()

    # Logging
    if (ep + 1) % 100 == 0 or ep == 0 or ep == epochs - 1:
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep + 1}/{epochs} | Train {train_loss:.6e} | Val {test_loss:.6e} | lr {lr_now:.3e}")

print(f"Time: {time() - t0:.2f}s")

best_model_filename = os.path.join("Results", f"best_fno1d_testloss_{best_test_loss:.8f}.pt")
torch.save(best_model_state, best_model_filename)
print(f"Best model saved as: {best_model_filename}")


# -----------------------------
# Plot loss curves
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_loss_list, label="Train Loss")
plt.plot(range(1, epochs + 1), test_loss_list, label="Val Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("FNO1d Training / Validation Loss (u normalized to [-1,1])")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join("Results", "fno1d_loss_curve.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"Loss curve saved to: {fig_path}")


# -----------------------------
# Post-training evaluation: denormalize and compare in true scale
# -----------------------------
# Load best model weights into model
model.load_state_dict(best_model_state)
model = model.to(device)
model.eval()

# Predict on the entire test set
with torch.no_grad():
    X_test_device = X_test.to(device)
    pred_norm = model(X_test_device).squeeze(1).cpu().numpy()  # (Ntest, 201), in [-1,1]

pred_true = u_denormalize_from_minus1_1(pred_norm, u_min, u_max)  # (Ntest, 201), true scale

# Report relative L2 error averaged over test samples (true scale)
errs = [rel_l2(pred_true[i], U_test_true[i]) for i in range(pred_true.shape[0])]
print(f"Test relative L2 (true scale): mean={np.mean(errs):.6e}, std={np.std(errs):.6e}")

# Plot a few sample comparisons (true scale)
idx_plot = [0, min(9, pred_true.shape[0] - 1), min(99, pred_true.shape[0] - 1)]
x_plot = np.linspace(0.0, 1.0, pred_true.shape[1], endpoint=False)

fig, axes = plt.subplots(len(idx_plot), 1, figsize=(10, 9), sharex=True)
if len(idx_plot) == 1:
    axes = [axes]

for ax, idx in zip(axes, idx_plot):
    ax.plot(x_plot, U_test_true[idx], lw=1.8, label="True u")
    ax.plot(x_plot, pred_true[idx], lw=1.2, ls="--", label="Pred u (denorm)")
    ax.set_title(f"Test sample #{idx+1} | rel L2 = {errs[idx]:.3e}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

axes[-1].set_xlabel("x")
axes[-1].set_ylabel("u")
plt.tight_layout()
cmp_path = os.path.join("Results", "fno1d_predictions_true_scale.png")
plt.savefig(cmp_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Prediction comparison saved to: {cmp_path}")
