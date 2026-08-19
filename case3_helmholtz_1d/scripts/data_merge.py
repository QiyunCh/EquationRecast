"""
merge_datasets_recast.py

Merge data_0.785.h5 and data_1.099.h5 into a single dataset for training
a canonical Helmholtz operator at k* = 0.785.

Key rules (as requested):
- Use u_min and u_max STRICTLY from data_0.785.h5
- For k = 1.099 data, recast the source:
      S_recast = S + (k^2 - k_star^2) * u
- The solution u remains unchanged
- Final merged dataset has 2000 samples
- Output format is compatible with Train.py (same dataset names)

Output:
  data_merged_0.785_1.099.h5
"""

import h5py
import numpy as np

# -----------------------------
# User-specified constants
# -----------------------------
k_star = 0.785
k_new = 1.099

FILE_0785 = "data_0.785.h5"
FILE_1099 = "data_1.099.h5"
OUT_FILE = "data_merged_0.785_1.099.h5"

# -----------------------------
# Load k=0.785 dataset (canonical)
# -----------------------------
with h5py.File(FILE_0785, "r") as f:
    S_0785 = f["source"][:]        # (N0, 201)
    U_0785 = f["solution"][:]      # (N0, 201)

    # Canonical normalization (MUST use these)
    u_min = float(f.attrs["solution_min"])
    u_max = float(f.attrs["solution_max"])

    # Grid metadata (copied verbatim)
    L  = float(f.attrs["L"])
    dx = float(f.attrs["dx"])
    N  = int(f.attrs["N"])
    k_ref = float(f.attrs["k"])

assert abs(k_ref - k_star) < 1e-12, "data_0.785.h5 is not k = 0.785"

# -----------------------------
# Load k=1.099 dataset
# -----------------------------
with h5py.File(FILE_1099, "r") as f:
    S_1099 = f["source"][:]        # (N1, 201)
    U_1099 = f["solution"][:]      # (N1, 201)

    k_loaded = float(f.attrs["k"])

assert abs(k_loaded - k_new) < 1e-12, "data_1.099.h5 is not k = 1.099"

# -----------------------------
# Recast sources for k=1.099 → k*
# -----------------------------
delta = k_new**2 - k_star**2
S_1099_recast = S_1099 + delta * U_1099   # IMPORTANT: no extra minus sign

# -----------------------------
# Merge datasets
# -----------------------------
S_merged = np.concatenate([S_0785, S_1099_recast], axis=0)
U_merged = np.concatenate([U_0785, U_1099], axis=0)

n_total = S_merged.shape[0]

print(f"Merged dataset size: {n_total} samples")
print(f"Using u_min = {u_min}")
print(f"Using u_max = {u_max}")

# -----------------------------
# Save merged dataset
# -----------------------------
with h5py.File(OUT_FILE, "w") as f:
    f.create_dataset("source", data=S_merged)
    f.create_dataset("solution", data=U_merged)

    # Canonical attributes (from k=0.785 dataset)
    f.attrs["L"] = L
    f.attrs["dx"] = dx
    f.attrs["N"] = N
    f.attrs["k"] = k_star
    f.attrs["solution_min"] = u_min
    f.attrs["solution_max"] = u_max

    f.attrs["note"] = (
        "Merged dataset for canonical Helmholtz operator at k=0.785. "
        "Includes original k=0.785 data and recast k=1.099 data using "
        "S_recast = S + (k^2 - k_star^2) * u. "
        "solution_min and solution_max are taken strictly from k=0.785 dataset."
    )

print(f"Saved merged dataset to: {OUT_FILE}")
print("Ready for direct use with Train.py (2000 samples).")
