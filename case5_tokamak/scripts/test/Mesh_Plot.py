#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure1_Mesh_4x2.py

4 rows (CMOD, CMOD_NewGeo, SPARC, ARC) × 2 cols:
  Col 0: Physical domain mesh  (H:W = 2:1)
  Col 1: Unit disk mesh        (1:1)

Boundary drawn from HM boundary_nodes (no BD dependency).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# ======================== GEOMETRY CONFIGS ========================

GEOMETRIES = [
    dict(
        label="CMOD",
        h5_map=Path("HM/CMOD_HM.h5"),
    ),
    dict(
        label="CMOD NewGeo",
        h5_map=Path("HM/CMOD_NewGeo_HM.h5"),
    ),
    dict(
        label="SPARC",
        h5_map=Path("HM/SPARC_HM.h5"),
    ),
    dict(
        label="ARC",
        h5_map=Path("HM/ARC_V02_HM.h5"),
    ),
]

OUT_PNG = Path("MeshPlot_combined_4x2.png")
DPI = 400


# ======================== HELPERS ========================

def unit_circle_six_points_from_anchor(W_nodes, anchor_node, npts=6):
    ua = float(np.real(W_nodes[anchor_node]))
    va = float(np.imag(W_nodes[anchor_node]))
    theta0 = np.arctan2(va, ua)
    thetas = theta0 + (2.0 * np.pi / npts) * np.arange(npts)
    return np.cos(thetas), np.sin(thetas)


def match_uv_points_to_physical_boundary(U, V, bnodes, pts_map, W_nodes):
    uv_b = np.c_[np.real(W_nodes[bnodes]), np.imag(W_nodes[bnodes])]
    targets = np.c_[U, V]
    phys = np.zeros((targets.shape[0], 2), dtype=float)
    for k in range(targets.shape[0]):
        d2 = np.sum((uv_b - targets[k]) ** 2, axis=1)
        phys[k, :] = pts_map[int(bnodes[int(np.argmin(d2))]), :]
    return phys


def load_mapping_data(h5_map):
    with h5py.File(h5_map, "r") as f:
        pts_map = np.asarray(f["mesh/pts"][...], dtype=float)
        tri_map = np.asarray(f["mesh/tri"][...], dtype=np.int64)
        bnodes_map = np.asarray(f["mesh/boundary_nodes"][...], dtype=np.int64)
        anchor_node = int(f["mesh"].attrs.get("anchor_node", 0))
        W_nodes = (np.asarray(f["map/W_nodes_real"][...], dtype=float)
                   + 1j * np.asarray(f["map/W_nodes_imag"][...], dtype=float))
    return pts_map, tri_map, bnodes_map, anchor_node, W_nodes


# ======================== MAIN ========================

def main():
    # ------ load all four geometries ------
    data = []
    for geo in GEOMETRIES:
        pts_map, tri_map, bnodes_map, anchor_node, W_nodes = load_mapping_data(geo["h5_map"])

        # physical boundary from HM boundary_nodes (ordered)
        bdry_R = pts_map[bnodes_map, 0]
        bdry_Z = pts_map[bnodes_map, 1]

        U, V = unit_circle_six_points_from_anchor(W_nodes, anchor_node, npts=6)
        phys6 = match_uv_points_to_physical_boundary(U, V, bnodes_map, pts_map, W_nodes)

        data.append(dict(
            label=geo["label"],
            pts_map=pts_map, tri_map=tri_map,
            bnodes_map=bnodes_map, anchor_node=anchor_node,
            W_nodes=W_nodes,
            bdry_R=bdry_R, bdry_Z=bdry_Z,
            U=U, V=V, phys6=phys6,
        ))

    # ------ figure layout: 4 rows x 2 cols ------
    n_rows = 4
    panel_h = 3.2
    phys_w  = panel_h * 0.5      # H:W = 2:1
    disk_w  = panel_h * 1.0      # 1:1

    wspace_in = 0.7
    hspace_in = 0.4
    left_m, right_m = 0.05, 0.05
    top_m, bot_m    = 0.50, 0.10

    total_w = left_m + phys_w + wspace_in + disk_w + right_m
    total_h = top_m + n_rows * panel_h + (n_rows - 1) * hspace_in + bot_m

    fig = plt.figure(figsize=(total_w, total_h), dpi=DPI)

    gs = fig.add_gridspec(
        n_rows, 2,
        left   = left_m / total_w,
        right  = 1 - right_m / total_w,
        top    = 1 - top_m / total_h,
        bottom = bot_m / total_h,
        wspace = wspace_in / ((phys_w + disk_w) / 2),
        hspace = hspace_in / panel_h,
        width_ratios = [phys_w, disk_w],
        height_ratios = [1] * n_rows,
    )

    # style
    fs_title = 15
    fs_tick  = 11
    fs_num   = 10
    marker_s = 30
    anchor_ms = 12
    mesh_lw  = 0.18
    mesh_alpha = 0.50
    bdry_lw  = 1.6
    mesh_color = "steelblue"
    bdry_color = "steelblue"
    pt_color   = "red"

    for row, d in enumerate(data):
        pts_map     = d["pts_map"]
        tri_map     = d["tri_map"]
        W_nodes     = d["W_nodes"]
        anchor_node = d["anchor_node"]
        phys6       = d["phys6"]
        U, V        = d["U"], d["V"]
        uv = np.c_[np.real(W_nodes), np.imag(W_nodes)]

        # ========== Col 0: Physical domain mesh ==========
        ax = fig.add_subplot(gs[row, 0])

        segs = pts_map[tri_map][:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segs, linewidths=mesh_lw, alpha=mesh_alpha, colors=mesh_color))

        # boundary from HM boundary_nodes
        ax.plot(d["bdry_R"], d["bdry_Z"], "-", lw=bdry_lw, alpha=0.9, color=bdry_color)

        # anchor (point 1): star, red
        Ra, Za = pts_map[anchor_node]
        ax.plot([Ra], [Za], marker="*", ms=anchor_ms, color=pt_color, zorder=6,
                markeredgecolor="darkred", markeredgewidth=0.5)

        # points 2–6: red circles
        ax.scatter(phys6[1:, 0], phys6[1:, 1], s=marker_s,
                   marker="o", facecolors=pt_color, edgecolors="darkred",
                   linewidths=0.7, zorder=5)

        # number labels
        for k in range(6):
            ax.text(phys6[k, 0], phys6[k, 1], f" {k+1}", fontsize=fs_num,
                    ha="left", va="bottom", fontweight="bold", color="k")

        ax.set_aspect("equal", adjustable="datalim")
        ax.autoscale_view()
        ax.tick_params(labelsize=fs_tick)

        # ========== Col 1: Unit disk mesh ==========
        ax = fig.add_subplot(gs[row, 1])

        segs_uv = uv[tri_map][:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 2)
        ax.add_collection(LineCollection(segs_uv, linewidths=mesh_lw, alpha=mesh_alpha, colors=mesh_color))

        # unit circle boundary
        th = np.linspace(0, 2 * np.pi, 600)
        ax.plot(np.cos(th), np.sin(th), lw=bdry_lw, color="steelblue")

        # anchor (point 1): star, red
        ua, va = uv[anchor_node]
        ax.plot([ua], [va], marker="*", ms=anchor_ms, color=pt_color, zorder=6,
                markeredgecolor="darkred", markeredgewidth=0.5)

        # points 2–6: red circles
        ax.scatter(U[1:], V[1:], s=marker_s,
                   marker="o", facecolors=pt_color, edgecolors="darkred",
                   linewidths=0.7, zorder=5)

        # number labels
        for k in range(6):
            ax.text(U[k], V[k], f" {k+1}", fontsize=fs_num,
                    ha="left", va="bottom", fontweight="bold", color="k")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.tick_params(labelsize=fs_tick)

    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print(f"[OK] Saved: {OUT_PNG.resolve()}")


if __name__ == "__main__":
    main()