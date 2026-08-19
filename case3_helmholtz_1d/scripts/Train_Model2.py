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
# Load data
# -----------------------------
path = "data_merged_0.785_1.099.h5"
with h5py.File(path, "r") as f:
    S_raw = f["source"][:]     # (num_samples, 201)
    U_raw = f["solution"][:]   # (num_samples, 201)
    u_min = float(f.attrs["solution_min"])
    u_max = float(f.attrs["solution_max"])

print("u_min = ", u_min)
print("u_max = ", u_max)
# -----------------------------
# Normalization / tensor shapes
#   - Input X: 2 channels = [S, x]; shape (N, 2, 201)
#   - Target Y: normalized U in [-1, 1]; shape (N, 1, 201)
# -----------------------------
S = torch.tensor(S_raw, dtype=torch.float32)            # (N, 201)

# coordinate channel (match grid length 201); use [-1, 1]
x_coord = torch.linspace(-1.0, 1.0, S.shape[1], dtype=torch.float32)
Xcoord = x_coord.unsqueeze(0).repeat(S.shape[0], 1)     # (N, 201)

# stack as channels: (N, 2, 201)
X = torch.stack([S, Xcoord], dim=1)

# normalize U to [-1, 1]
U_norm = 2.0 * (U_raw - u_min) / (u_max - u_min) - 1.0
Y = torch.tensor(U_norm, dtype=torch.float32).unsqueeze(1)   # (N, 1, 201)

# -----------------------------
# Train-test split
# -----------------------------
n_total = X.shape[0]
ntrain = int(0.90 * n_total)

X_train, X_test = X[:ntrain], X[ntrain:]
Y_train, Y_test = Y[:ntrain], Y[ntrain:]

# -----------------------------
# Dataloaders
# -----------------------------
batch_size = 2000
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
model = FNO1d(modes=64, width=64, in_channels=2, out_channels=1).to(device)
learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 2000
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
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
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
            xb = xb.to(device)
            yb = yb.to(device)
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

    # # Optional periodic checkpoint
    # if (ep + 1) % 50 == 0 or ep == epochs - 1:
    #     ckpt_path = os.path.join("Results", "checkpoints", f"epoch_{ep+1:04d}.pt")
    #     torch.save(model.state_dict(), ckpt_path)

print(f"Time: {time() - t0:.2f}s")

# Save best with test loss in filename
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
plt.title("FNO1d Training / Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig_path = os.path.join("Results", "fno1d_loss_curve.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"Loss curve saved to: {fig_path}")
