from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import h5py
import numpy as np

# =============================== DATA CONTAINERS ===============================

@dataclass
class BoundaryData:
    """Boundary info for harmonic mapping."""
    # Mesh (for reference / downstream usage)
    pts: np.ndarray            # (N, 2) float64, columns: [R, Z]
    tri: np.ndarray            # (T, 3) int64, triangle connectivity (node ids)

    # Boundary
    b_nodes: np.ndarray        # (Nb,) int64, ordered CCW node ids (outer boundary)
    Rb: np.ndarray             # (Nb,) float64, boundary R at vertices
    Zb: np.ndarray             # (Nb,) float64, boundary Z at vertices

    # Anchor
    anchor_node: int           # int, node id of alignment vertex (closest to Z=0, then min R)


@dataclass
class FEInterpData:
    """
    Minimal precomputed info for fast P1 FE interpolation:
      For each triangle t:
        p0 = pts[i0]
        A_inv = inverse of [[x1-x0, x2-x0],
                            [y1-y0, y2-y0]]
      With (lambda1, lambda2) = A_inv @ (p - p0),
      lambda0 = 1 - lambda1 - lambda2.
      Field value: f = sum(lambda_k * f_k).
    """
    tri: np.ndarray            # (T, 3) int64
    p0: np.ndarray             # (T, 2) float64
    A_inv: np.ndarray          # (T, 2, 2) float64
    area2: np.ndarray          # (T,) float64, signed 2*area (useful for checks)


# =============================== CORE UTILITIES ================================

def _load_zone1_geometry(h5_in: Path, case: str, time_group: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild unstructured triangle mesh nodes & connectivity for Zone-1."""
    with h5py.File(h5_in, "r") as f:
        g = f[f"{case}/{time_group}/geometry"]
        a     = g["a"][...]
        b     = g["b"][...]
        c     = g["c"][...]
        theta = g["theta"][...]
        x     = g["x"][...]
        z     = g["z"][...]

    co = np.cos(theta); sn = np.sin(theta)

    R0, Z0 = x, z
    R1 = x + (a + b) * co
    Z1 = z + (a + b) * sn
    R2 = x +  b * co - c * sn
    Z2 = z +  c * co + b * sn

    RV = np.stack([R0, R1, R2], axis=1)  # (nel, 3)
    ZV = np.stack([Z0, Z1, Z2], axis=1)

    P = np.c_[RV.ravel(), ZV.ravel()]    # (3*nel, 2)

    # Deduplicate nodes robustly
    span = max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 1.0)
    tol  = max(1e-12, 1e-10 * span)
    key  = np.round(P / tol).astype(np.int64)
    _, idx_unique, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pts = P[idx_unique].astype(np.float64)
    tri = inv.reshape(RV.shape[0], 3).astype(np.int64)

    # Drop degenerate triangles
    x0 = pts[tri[:, 0], 0]; y0 = pts[tri[:, 0], 1]
    x1 = pts[tri[:, 1], 0]; y1 = pts[tri[:, 1], 1]
    x2 = pts[tri[:, 2], 0]; y2 = pts[tri[:, 2], 1]
    area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    scale = np.maximum(1.0, np.maximum.reduce([
        np.abs(x0) + np.abs(y0),
        np.abs(x1) + np.abs(y1),
        np.abs(x2) + np.abs(y2)
    ]))
    keep = np.abs(area2) > 1e-14 * scale
    tri = tri[keep]
    if tri.size == 0:
        raise RuntimeError("No valid triangles after dedup.")
    return pts, tri


def _boundary_edges(tri: np.ndarray) -> np.ndarray:
    """Edges used by exactly one triangle (outer boundary candidate)."""
    T = np.asarray(tri, dtype=np.int64)
    e01 = np.sort(T[:, [0, 1]], axis=1)
    e12 = np.sort(T[:, [1, 2]], axis=1)
    e20 = np.sort(T[:, [2, 0]], axis=1)
    edges = np.vstack([e01, e12, e20])
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _polygon_signed_area(pts: np.ndarray, loop: list[int]) -> float:
    R = pts[loop, 0]; Z = pts[loop, 1]
    return 0.5 * np.sum(R * np.roll(Z, -1) - Z * np.roll(R, -1))


def _order_boundary_outer_loop(pts: np.ndarray, b_edges: np.ndarray) -> np.ndarray:
    """Assemble loops from boundary edges, return the outermost loop as CCW node ids."""
    # Build adjacency
    adj: dict[int, list[int]] = {}
    for i, j in b_edges:
        i = int(i); j = int(j)
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # Walk loops greedily in CCW order at each vertex
    def neighbors_ccw(i: int) -> list[int]:
        nbrs = adj[i]
        p = pts[i]
        ang = [math.atan2(pts[j, 1] - p[1], pts[j, 0] - p[0]) for j in nbrs]
        return [nbrs[k] for k in np.argsort(ang)]

    visited: set[int] = set()
    loops: list[list[int]] = []

    for start in list(adj.keys()):
        if start in visited:
            continue
        curr, prev = start, None
        loop = [curr]
        visited.add(curr)

        for _ in range(len(adj) + 5):
            nbrs = neighbors_ccw(curr)
            nxt = None
            for cand in nbrs:
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None:
                break
            if nxt == start and len(loop) > 2:
                loops.append(loop[:])
                break
            if nxt in visited:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

    if not loops:
        raise RuntimeError("Could not assemble boundary loops.")

    # Pick outermost by |area| and enforce CCW
    areas = [abs(_polygon_signed_area(pts, L)) for L in loops]
    L = loops[int(np.argmax(areas))]
    if _polygon_signed_area(pts, L) < 0:
        L = L[::-1]
    return np.asarray(L, dtype=int)


def _find_anchor_node(pts: np.ndarray,
                      b_loop: np.ndarray,
                      z_tol_factor: float = 5.0) -> int:
    """
    Choose anchor as the leftmost boundary vertex near Z = 0.

    Steps:
      1) On boundary loop, compute |Z| for each vertex and find min|Z|;
      2) Build a band of 'mid-plane' candidates: |Z| <= z_tol_factor * min|Z|;
      3) Among these candidates, pick the one with minimal R (leftmost).

    Parameters
    ----------
    pts : (N, 2) array
        All mesh node coordinates (R, Z).
    b_loop : (M,) array of int
        Boundary node indices in order along the loop.
    z_tol_factor : float, optional
        How wide the Z-band is relative to the closest |Z|.
        Default 5.0 works well for typical tokamak cross-sections.

    Returns
    -------
    anchor : int
        Global node id of the anchor vertex.
    """
    # boundary coordinates
    Rb = pts[b_loop, 0]
    Zb = pts[b_loop, 1]

    absZ = np.abs(Zb)
    z_min = absZ.min()
    z_range = Zb.max() - Zb.min()
    tol = max(z_tol_factor * z_min, 0.01 * z_range) + 1e-14  # 0.01 可调
    cand_mask = absZ <= tol


    if not np.any(cand_mask):
        cand_mask[:] = True

    R_cand = Rb[cand_mask]
    idx_local = np.argmin(R_cand)

    cand_indices = np.nonzero(cand_mask)[0]
    anchor_idx_on_loop = cand_indices[idx_local]

    return int(b_loop[anchor_idx_on_loop])




def _precompute_fe_affine(pts: np.ndarray, tri: np.ndarray) -> FEInterpData:
    """
    Precompute per-triangle affine inverses for barycentric weights.
    """
    p0 = pts[tri[:, 0], :]             # (T, 2)
    p1 = pts[tri[:, 1], :]
    p2 = pts[tri[:, 2], :]

    v1 = p1 - p0                       # (T, 2)
    v2 = p2 - p0                       # (T, 2)

    # A = [v1 v2]; we need A_inv
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (T,) = 2*area (signed)
    if np.any(np.abs(det) < 1e-30):
        raise RuntimeError("Degenerate triangle detected while building FE affine maps.")

    inv_det = 1.0 / det
    A_inv = np.empty((tri.shape[0], 2, 2), dtype=np.float64)
    A_inv[:, 0, 0] =  v2[:, 1] * inv_det
    A_inv[:, 0, 1] = -v2[:, 0] * inv_det
    A_inv[:, 1, 0] = -v1[:, 1] * inv_det
    A_inv[:, 1, 1] =  v1[:, 0] * inv_det

    return FEInterpData(tri=tri, p0=p0, A_inv=A_inv, area2=det)


# =============================== PUBLIC ENTRYPOINT =============================

def extract_boundary_and_fe(h5_in: Path, case: str, time_group: str) -> tuple[BoundaryData, FEInterpData]:
    """
    Minimal helper: load mesh, extract outer boundary + anchor, and precompute FE affine maps.
    Returns (BoundaryData, FEInterpData).
    """
    pts, tri = _load_zone1_geometry(h5_in, case, time_group)
    b_edges = _boundary_edges(tri)
    b_loop  = _order_boundary_outer_loop(pts, b_edges)
    anchor  = _find_anchor_node(pts, b_loop)

    bdry = BoundaryData(
        pts=pts,
        tri=tri,
        b_nodes=b_loop,
        Rb=pts[b_loop, 0].copy(),
        Zb=pts[b_loop, 1].copy(),
        anchor_node=anchor,
    )
    fe = _precompute_fe_affine(pts, tri)
    return bdry, fe



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # --- edit these paths as needed ---
    H5   = Path("all_cases_unstructured_zone1.h5")
    CASE = "TG2"
    TIME = "time_006"
    # ----------------------------------

    bdry, _ = extract_boundary_and_fe(H5, CASE, TIME)

    fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=220)

    # mesh edges
    segs = bdry.pts[bdry.tri][:, [[0, 1], [1, 2], [2, 0]], :].reshape(-1, 2, 2)
    ax.add_collection(LineCollection(segs, colors="#999999", linewidths=0.25, alpha=0.8))

    # axis limits with padding
    xmin, xmax = float(bdry.pts[:, 0].min()), float(bdry.pts[:, 0].max())
    ymin, ymax = float(bdry.pts[:, 1].min()), float(bdry.pts[:, 1].max())
    xr, yr = xmax - xmin, ymax - ymin
    pad_x = 0.05 * xr if xr > 0 else 1e-3
    pad_y = 0.05 * yr if yr > 0 else 1e-3
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(x=0.02, y=0.02)

    # boundary + anchor
    Rb, Zb = bdry.Rb, bdry.Zb
    ax.plot(Rb, Zb, "o", ms=2.2, color="#1f77b4", alpha=0.9, label="boundary nodes (CCW)")
    ax.plot(np.r_[Rb, Rb[0]], np.r_[Zb, Zb[0]], "-", lw=1.6, color="#ff7f0e", label="boundary polyline")
    Ra, Za = bdry.pts[bdry.anchor_node]
    ax.plot(Ra, Za, marker="*", ms=12, color="#d62728", label="anchor (Z≈0, min R)")

    ax.set_xlabel("R"); ax.set_ylabel("Z")
    ax.set_title(f"Zone-1 Mesh & Boundary (case={CASE}, {TIME})")
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.92)
    plt.show()
