"""
merge_datasets_recast_RD.py

Merge Reaction–Diffusion datasets at:
- canonical beta* = 0.785
- new beta = 0.05

Goal:
Train a canonical operator at beta* = 0.785 using merged data:
  [ (S, u) from beta* ]  U [ (S_recast, u) from beta_new recast to beta* ]

Equation:
  -u'' + beta u = S   (Dirichlet)

Operator identity:
  A_beta u = S,  A_beta = (-d2/dx2 + beta I)
  A_beta = A_beta_star + (beta - beta_star) I

Recast (beta_new -> beta*):
  A_beta_star u = S - (beta_new - beta_star) u
               = S + (beta_star - beta_new) u

So:
  S_recast = S + (beta_star - beta_new) * u

Key rules (as requested):
- Use u_min and u_max STRICTLY from beta*=0.785 dataset
  (you provided: u_max=0.9274784849708392, u_min=-0.9180988552608058)
- Do NOT change solution u for beta=0.05 data
- Output format compatible with Train.py / FNO1D.py:
    datasets: "source", "solution"
    attrs: L, dx, N, beta, solution_min, solution_max

Output:
  data_merged_beta_0.785_0.050.h5
"""

import h5py
import numpy as np

# -----------------------------
# User-specified constants
# -----------------------------
beta_star = 0.785
beta_new  = 0.05

# Files (rename if your filenames differ)
FILE_0785 = "data_beta_0.785.h5"
FILE_0050 = "data_beta_0.050.h5"
OUT_FILE  = "data_merged_beta_0.785_0.050.h5"

# Canonical normalization (MUST use these)
U_MIN_CANON = -0.9180988552608058
U_MAX_CANON =  0.9274784849708392

# -----------------------------
# Load beta=0.785 dataset (canonical)
# -----------------------------
with h5py.File(FILE_0785, "r") as f:
    S_0785 = f["source"][:]        # (N0, N)
    U_0785 = f["solution"][:]      # (N0, N)

    # Grid metadata (copied verbatim)
    L  = float(f.attrs["L"])
    dx = float(f.attrs["dx"])
    N  = int(f.attrs["N"])

    beta_ref = float(f.attrs.get("beta", beta_star))

# sanity
if abs(beta_ref - beta_star) > 1e-12:
    raise ValueError(f"{FILE_0785} is not beta={beta_star}. Found beta={beta_ref}.")

# -----------------------------
# Load beta=0.05 dataset
# -----------------------------
with h5py.File(FILE_0050, "r") as f:
    S_0050 = f["source"][:]        # (N1, N)
    U_0050 = f["solution"][:]      # (N1, N)

    beta_loaded = float(f.attrs.get("beta", beta_new))

if abs(beta_loaded - beta_new) > 1e-12:
    raise ValueError(f"{FILE_0050} is not beta={beta_new}. Found beta={beta_loaded}.")

# -----------------------------
# Recast sources for beta=0.05 → beta*
# -----------------------------
delta = (beta_star - beta_new)
S_0050_recast = S_0050 + delta * U_0050   # S_recast = S + (beta_star - beta_new) * u

# -----------------------------
# Merge datasets
# -----------------------------
S_merged = np.concatenate([S_0785, S_0050_recast], axis=0)
U_merged = np.concatenate([U_0785, U_0050], axis=0)

n_total = S_merged.shape[0]

print(f"Merged dataset size: {n_total} samples")
print(f"Using canonical u_min = {U_MIN_CANON}")
print(f"Using canonical u_max = {U_MAX_CANON}")
print(f"Recast delta = beta_star - beta_new = {delta}")

# -----------------------------
# Save merged dataset
# -----------------------------
with h5py.File(OUT_FILE, "w") as f:
    f.create_dataset("source", data=S_merged)
    f.create_dataset("solution", data=U_merged)

    # Canonical attributes (from beta*=0.785 dataset)
    f.attrs["L"] = L
    f.attrs["dx"] = dx
    f.attrs["N"] = N
    f.attrs["beta"] = beta_star

    # MUST use canonical u_min/u_max
    f.attrs["solution_min"] = float(U_MIN_CANON)
    f.attrs["solution_max"] = float(U_MAX_CANON)

    f.attrs["note"] = (
        "Merged dataset for canonical Reaction–Diffusion operator at beta=0.785. "
        "Includes original beta=0.785 data and recast beta=0.05 data using "
        "S_recast = S + (beta_star - beta_new) * u, derived from "
        "A_beta = A_beta_star + (beta - beta_star)I and A_beta u = S. "
        "solution_min and solution_max are taken strictly from beta=0.785 dataset."
    )

print(f"Saved merged dataset to: {OUT_FILE}")
print("Ready for direct use with Train.py.")
