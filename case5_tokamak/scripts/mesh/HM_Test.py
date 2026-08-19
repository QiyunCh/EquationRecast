#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HM_Test.py – Plot the physical-domain mesh only.

Output: a single PNG of the FE mesh with no markers, no title, no axis titles,
bold axis labels, and configurable tick intervals.
"""

from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import BD

# =============================== USER CONFIG ===============================

H5_PACKED = Path("CMOD_Ran.h5")
H5_MAP    = Path("CMOD_SHARED_HM.h5")

CASE = "T1"
TIME = "time_002"

DPI = 300

# Axis-label (tick) intervals – set to None for automatic ticks
X_TICK_INTERVAL = 0.2   # e.g. 0.1 → ticks every 0.1 in R
Y_TICK_INTERVAL = 0.3   # e.g. 0.1 → ticks every 0.1 in Z

# ===========================================================================


def main():
    # ---- load mapping mesh ----
    with h5py.File(H5_MAP, "r") as f:
        pts_map = np.asarray(f["mesh/pts"][...], dtype=float)
        tri_map = np.asarray(f["mesh/tri"][...], dtype=np.int64)

    # ---- load boundary for bold edge ----
    bdry_phys, _ = BD.extract_boundary_and_fe(H5_PACKED, CASE, TIME)

    # ---- build edge segments from triangles ----
    segs = pts_map[tri_map][:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 2)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 7), dpi=DPI)
    ax.add_collection(LineCollection(segs, linewidths=0.25, alpha=0.65))
    ax.plot(bdry_phys.Rb, bdry_phys.Zb, "-", lw=1.5, alpha=0.9)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")

    # Bold axis labels (tick labels), no axis titles
    ax.tick_params(axis="both", labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Apply custom tick intervals if specified
    if X_TICK_INTERVAL is not None:
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.arange(
            np.ceil(xmin / X_TICK_INTERVAL) * X_TICK_INTERVAL,
            xmax + X_TICK_INTERVAL * 0.01,
            X_TICK_INTERVAL,
        ))
    if Y_TICK_INTERVAL is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.arange(
            np.ceil(ymin / Y_TICK_INTERVAL) * Y_TICK_INTERVAL,
            ymax + Y_TICK_INTERVAL * 0.01,
            Y_TICK_INTERVAL,
        ))

    fig.tight_layout()

    # Save to the same directory as the H5 file
    out_png = H5_MAP.parent / f"Mesh_{CASE}_{TIME}.png"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_png.resolve()}")


if __name__ == "__main__":
    main()