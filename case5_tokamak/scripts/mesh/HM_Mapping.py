#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HM_Mapping.py (REVISED)

Implements a shared harmonic mapping for ONE reference mesh/time-group and saves:
  1) Correct disk validity mask: physical domain mapped into UNIT DISK; outside disk is invalid (NaN downstream).
     valid_mask := inside_disk & (inv_tri_id >= 0)

  2) Mapping assets saved at BOTH 1024 and 256:
     - Forward (rectangular grid -> physical mesh bary lookup): fwd_tri_id, fwd_bary
     - Inverse (unit disk grid -> physical mesh bary lookup):   inv_tri_id, inv_bary, inside, valid_mask

  3) Geometry tensors following your equations:
     - K(y) = dPhi/dy (piecewise-constant per triangle) saved on 256 grid
     - |K| = det(K)   saved on 256 grid
     - G  = K^{-1} K^{-T} saved on 256 grid

Notes:
  - Mapping is computed ONCE from a chosen reference (case, time_group) and then shared across all cases.
  - K, |K|, G are computed consistently using uv-triangle geometry (harmonic map) and physical triangle geometry.

Dependencies:
  - numpy, scipy, h5py, matplotlib (only if you re-enable plotting)
  - BD.py must be available (extract_boundary_and_fe)
"""

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve
from matplotlib.tri import Triangulation

import BD  # extract_boundary_and_fe(...) -> (BoundaryData, FEInterpData)


# ============================== CONFIG ==============================

# Input packed FE/geometry HDF5 (from pack_unstructured_zone1.py)
H5_IN   = Path("CMOD_Ran.h5")

# Reference group to build ONE shared mapping
REF_CASE   = "T1"
REF_TGROUP = "time_000"

# Rectangular sampling resolutions (forward tables)
NRECT_1024 = 1024
NRECT_256  = 256

# Disk resolutions (inverse tables)
NDISK_1024 = 1024
NDISK_256  = 256

# Output mapping asset file
OUT_TAG = f"CMOD_SHARED_HM"
H5_OUT  = Path(f"{OUT_TAG}.h5")


# ============================== MEAN-VALUE LAPLACIAN ==============================

def _adjacency_from_tri(tri_indices: np.ndarray, n_nodes: int) -> list[list[int]]:
    neigh = [[] for _ in range(n_nodes)]
    for a, b, c in tri_indices:
        neigh[a].extend([b, c])
        neigh[b].extend([a, c])
        neigh[c].extend([a, b])
    for i in range(n_nodes):
        neigh[i] = sorted(set(j for j in neigh[i] if j != i))
    return neigh


def _mean_value_laplacian(pts: np.ndarray, tri: np.ndarray) -> sparse.csr_matrix:
    x, y = pts[:, 0], pts[:, 1]
    n = pts.shape[0]
    neigh = _adjacency_from_tri(tri, n)

    rows, cols, vals = [], [], []
    for i in range(n):
        Ni = neigh[i]
        if not Ni:
            rows.append(i); cols.append(i); vals.append(1.0)
            continue

        vi = np.array([x[i], y[i]])
        vecs = np.array([[x[j]-vi[0], y[j]-vi[1]] for j in Ni])
        angs = np.arctan2(vecs[:, 1], vecs[:, 0])
        order = np.argsort(angs)
        Ni = [Ni[k] for k in order]
        vecs = vecs[order]
        m = len(Ni)

        wi_sum = 0.0
        for t, j in enumerate(Ni):
            vj   = vecs[t]
            prev = vecs[(t-1) % m]
            nxt  = vecs[(t+1) % m]

            def angle(a, b):
                na = np.linalg.norm(a); nb = np.linalg.norm(b)
                if na*nb < 1e-30:
                    return 0.0
                c = np.clip(np.dot(a, b) / (na*nb), -1.0, 1.0)
                return np.arccos(c)

            a_prev = angle(prev, vj)
            a_next = angle(vj,  nxt)
            rj = np.linalg.norm(vj)
            w = (np.tan(0.5*a_prev) + np.tan(0.5*a_next)) / max(rj, 1e-30)

            if np.isfinite(w) and w > 0:
                rows.append(i); cols.append(j); vals.append(-w)
                wi_sum += w

        rows.append(i); cols.append(i); vals.append(wi_sum)

    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================== BOUNDARY ANGLES (ANCHOR) ==============================

def _boundary_angles_with_anchor(bdry: BD.BoundaryData) -> tuple[np.ndarray, np.ndarray]:
    bnodes = bdry.b_nodes.copy()
    anchor_pos = int(np.where(bnodes == bdry.anchor_node)[0][0])
    bnodes = np.roll(bnodes, -anchor_pos)

    Rb = bdry.pts[bnodes, 0]
    Zb = bdry.pts[bnodes, 1]
    d = np.hypot(np.diff(Rb, append=Rb[0]), np.diff(Zb, append=Zb[0]))
    s = np.r_[0.0, np.cumsum(d[:-1])]
    L = float(s[-1] + d[-1])
    s_norm = s / max(L, 1e-30)

    theta = (2.0*np.pi*s_norm + np.pi) % (2.0*np.pi)  # anchor -> (-1,0)
    return bnodes, theta


# ============================== HARMONIC MAP SOLVER ==============================

def solve_harmonic_map(bdry: BD.BoundaryData) -> np.ndarray:
    n = bdry.pts.shape[0]
    L = _mean_value_laplacian(bdry.pts, bdry.tri)

    bnodes, theta = _boundary_angles_with_anchor(bdry)
    zbc = np.exp(1j*theta)

    bc = np.zeros(n, dtype=complex)
    bc[bnodes] = zbc

    fixed = np.zeros(n, dtype=bool); fixed[bnodes] = True
    free = np.where(~fixed)[0]; bnd = np.where(fixed)[0]
    if free.size == 0:
        return bc

    Lff = L[free][:, free]
    Lfb = L[free][:, bnd]
    rhs_u = -Lfb.dot(np.real(bc[bnd]))
    rhs_v = -Lfb.dot(np.imag(bc[bnd]))

    u = np.zeros(n); v = np.zeros(n)
    u[bnd] = np.real(bc[bnd]); v[bnd] = np.imag(bc[bnd])
    u[free] = spsolve(Lff, rhs_u)
    v[free] = spsolve(Lff, rhs_v)

    W = u + 1j*v
    r = np.abs(W)
    over = r > 1.0
    if np.any(over):
        W[over] *= (1.0 - 1e-8) / r[over]
    return W


# ============================== BARYCENTRIC TABLES ==============================

def precompute_forward_rect_bary(pts: np.ndarray, tri: np.ndarray, fe: BD.FEInterpData, N: int):
    """
    Forward lookup: rectangular physical grid -> triangle index + bary weights in (R,Z).
    Useful if you ever want rasterize physical -> rect. Not required for disk mapping.
    """
    Rmin, Rmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    Zmin, Zmax = float(pts[:, 1].min()), float(pts[:, 1].max())
    R_rect = np.linspace(Rmin, Rmax, N)
    Z_rect = np.linspace(Zmin, Zmax, N)
    ZZ, RR = np.meshgrid(Z_rect, R_rect, indexing="ij")

    tri_dom = Triangulation(pts[:, 0], pts[:, 1], tri)
    trifinder = tri_dom.get_trifinder()
    t_id = trifinder(RR, ZZ).astype(np.int32)  # -1 outside physical mesh

    nz, nr = ZZ.shape
    bary = np.zeros((nz, nr, 3), dtype=np.float32)

    inside = t_id >= 0
    if np.any(inside):
        idx = t_id[inside]
        p0   = fe.p0[idx]                    # (M,2)
        Ainv = fe.A_inv[idx]                 # (M,2,2)
        p    = np.c_[RR[inside], ZZ[inside]] # (M,2)
        lam12 = (Ainv @ (p - p0)[:, :, None])[:, :, 0]  # (M,2)
        lam1, lam2 = lam12[:, 0], lam12[:, 1]
        lam0 = 1.0 - lam1 - lam2
        bary.reshape(-1, 3)[inside.ravel()] = np.c_[lam0, lam1, lam2].astype(np.float32)

    return R_rect, Z_rect, t_id, bary


def precompute_inverse_disk_bary(pts: np.ndarray, tri: np.ndarray, W_nodes: np.ndarray, N: int, eps: float = 0.0):
    """
    Inverse lookup: unit disk grid -> triangle index in uv-mesh + bary weights.
    Defines 'inside' as r<=1. Only those pixels are intended valid.
    """
    xd = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    yd = np.linspace(-1.0, 1.0, N, dtype=np.float64)
    Xd, Yd = np.meshgrid(xd, yd, indexing="xy")
    r = np.hypot(Xd, Yd)
    inside = (r <= 1.0).astype(np.uint8)

    if eps > 0.0:
        scale = np.minimum(1.0, (1.0 - eps) / np.maximum(r, 1e-30))
        Xq, Yq = Xd * scale, Yd * scale
    else:
        Xq, Yq = Xd, Yd

    uv = np.c_[W_nodes.real, W_nodes.imag]  # (Nnodes, 2)
    tri_uv = Triangulation(uv[:, 0], uv[:, 1], tri)
    trif_uv = tri_uv.get_trifinder()
    t_id = trif_uv(Xq, Yq).astype(np.int32)  # -1 outside uv-hull

    inv_bary = np.zeros((Xd.shape[0], Xd.shape[1], 3), dtype=np.float32)

    # Precompute per-triangle affine inverse in uv (for barycentric weights)
    p0 = uv[tri[:, 0], :]
    p1 = uv[tri[:, 1], :]
    p2 = uv[tri[:, 2], :]
    v1 = p1 - p0
    v2 = p2 - p0
    det = v1[:, 0]*v2[:, 1] - v1[:, 1]*v2[:, 0]
    inv_det = 1.0 / np.where(np.abs(det) < 1e-30, np.inf, det)

    Ainv = np.empty((tri.shape[0], 2, 2), dtype=np.float64)
    Ainv[:, 0, 0] =  v2[:, 1]*inv_det
    Ainv[:, 0, 1] = -v2[:, 0]*inv_det
    Ainv[:, 1, 0] = -v1[:, 1]*inv_det
    Ainv[:, 1, 1] =  v1[:, 0]*inv_det

    ok = (t_id >= 0) & (inside.astype(bool))
    if np.any(ok):
        idx = t_id[ok]
        q   = np.c_[Xq[ok], Yq[ok]]
        lam12 = (Ainv[idx] @ (q - p0[idx])[:, :, None])[:, :, 0]
        lam1, lam2 = lam12[:, 0], lam12[:, 1]
        lam0 = 1.0 - lam1 - lam2
        inv_bary.reshape(-1, 3)[ok.ravel()] = np.c_[lam0, lam1, lam2].astype(np.float32)

    # Define VALID MASK: inside disk AND triangle is found
    valid_mask = ok.astype(np.uint8)
    return xd, yd, inside, t_id, inv_bary, valid_mask


# ============================== OPERATORS (Φ and M) ==============================

def build_forward_operator(inv_tri_id: np.ndarray,
                           inv_bary: np.ndarray,
                           tri: np.ndarray,
                           n_nodes: int,
                           valid_mask: np.ndarray) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build Φ such that y = Φ x maps nodal field x (size n_nodes) to valid disk pixels (flattened).
    IMPORTANT: validity must enforce unit-disk semantics; valid_mask should already be (inside & t_id>=0).
    """
    ok = (valid_mask.astype(bool))
    mask = ok.copy()

    linear = np.arange(inv_tri_id.size, dtype=np.int64).reshape(inv_tri_id.shape)
    rows_flat = linear[ok]
    order = np.argsort(rows_flat)
    rows_flat = rows_flat[order]

    t_idx = inv_tri_id[ok][order]
    lam   = inv_bary[ok][order]        # (M,3)
    a = tri[t_idx, 0]; b = tri[t_idx, 1]; c = tri[t_idx, 2]

    M = lam.shape[0]
    data = np.empty(3*M, dtype=np.float64)
    rvec = np.empty(3*M, dtype=np.int64)
    cvec = np.empty(3*M, dtype=np.int64)

    data[0::3] = lam[:, 0]; data[1::3] = lam[:, 1]; data[2::3] = lam[:, 2]
    r = np.repeat(np.arange(M, dtype=np.int64), 3)
    rvec[:] = r
    cvec[0::3] = a; cvec[1::3] = b; cvec[2::3] = c

    Phi = sparse.coo_matrix((data, (rvec, cvec)), shape=(M, n_nodes)).tocsr()
    return Phi, mask


def build_normal_matrix(Phi: sparse.csr_matrix) -> sparse.csr_matrix:
    return (Phi.T @ Phi).tocsr()


# ============================== GEOMETRY TENSORS: K, |K|, G ==============================

def compute_element_K_absK_G(pts: np.ndarray, tri: np.ndarray, W_nodes: np.ndarray):
    """
    Compute per-element:
      K_e   = dx/dy (2x2)
      absK  = det(K_e)
      G_e   = K_e^{-1} K_e^{-T} (2x2)
    Here y=(u,v) are disk coordinates (harmonic map values), x=(R,Z) physical.

    For linear triangles:
      x(y) = x0 + Ax * Auv^{-1} (y - y0)
      so K_e = Ax * Auv^{-1}.
    """
    uv = np.c_[W_nodes.real, W_nodes.imag].astype(np.float64)

    x0 = pts[tri[:, 0], :].astype(np.float64)
    x1 = pts[tri[:, 1], :].astype(np.float64)
    x2 = pts[tri[:, 2], :].astype(np.float64)

    y0 = uv[tri[:, 0], :]
    y1 = uv[tri[:, 1], :]
    y2 = uv[tri[:, 2], :]

    w1x = x1 - x0
    w2x = x2 - x0
    w1y = y1 - y0
    w2y = y2 - y0

    # Ax and Auv (2x2 per element): columns are edge vectors
    Ax = np.empty((tri.shape[0], 2, 2), dtype=np.float64)
    Auv = np.empty((tri.shape[0], 2, 2), dtype=np.float64)

    Ax[:, :, 0]  = w1x
    Ax[:, :, 1]  = w2x
    Auv[:, :, 0] = w1y
    Auv[:, :, 1] = w2y

    detAuv = Auv[:, 0, 0]*Auv[:, 1, 1] - Auv[:, 0, 1]*Auv[:, 1, 0]
    bad = np.abs(detAuv) < 1e-30
    if np.any(bad):
        raise RuntimeError("Degenerate uv-triangle detected: det(Auv) ~ 0. Harmonic map may have folds.")

    invdet = 1.0 / detAuv
    Auv_inv = np.empty_like(Auv)
    Auv_inv[:, 0, 0] =  Auv[:, 1, 1]*invdet
    Auv_inv[:, 0, 1] = -Auv[:, 0, 1]*invdet
    Auv_inv[:, 1, 0] = -Auv[:, 1, 0]*invdet
    Auv_inv[:, 1, 1] =  Auv[:, 0, 0]*invdet

    # K = Ax @ Auv^{-1}
    K = Ax @ Auv_inv

    absK = K[:, 0, 0]*K[:, 1, 1] - K[:, 0, 1]*K[:, 1, 0]

    # invK
    detK = absK
    if np.any(np.abs(detK) < 1e-30):
        raise RuntimeError("Degenerate K detected: det(K) ~ 0.")

    invdetK = 1.0 / detK
    invK = np.empty_like(K)
    invK[:, 0, 0] =  K[:, 1, 1]*invdetK
    invK[:, 0, 1] = -K[:, 0, 1]*invdetK
    invK[:, 1, 0] = -K[:, 1, 0]*invdetK
    invK[:, 1, 1] =  K[:, 0, 0]*invdetK

    # G = invK @ invK^T = K^{-1} K^{-T}
    G = invK @ np.transpose(invK, (0, 2, 1))

    return K, absK, G


def rasterize_element_tensors_to_disk(inv_tri_id: np.ndarray,
                                      valid_mask: np.ndarray,
                                      K_e: np.ndarray,
                                      absK_e: np.ndarray,
                                      G_e: np.ndarray):
    """
    Create disk-grid arrays (Ny,Nx,...) by indexing per-element tensors using inv_tri_id.
    Outside valid_mask: NaN.
    """
    Ny, Nx = inv_tri_id.shape

    K_disk = np.full((Ny, Nx, 2, 2), np.nan, dtype=np.float64)
    absK_disk = np.full((Ny, Nx), np.nan, dtype=np.float64)
    G_disk = np.full((Ny, Nx, 2, 2), np.nan, dtype=np.float64)

    ok = valid_mask.astype(bool)
    if np.any(ok):
        idx = inv_tri_id[ok]
        K_disk[ok, :, :] = K_e[idx]
        absK_disk[ok] = absK_e[idx]
        G_disk[ok, :, :] = G_e[idx]

    return K_disk, absK_disk, G_disk


# ============================== HDF5 SAVE HELPERS ==============================

def _save_csr(group: h5py.Group, name: str, mat: sparse.csr_matrix):
    g = group.create_group(name)
    g.create_dataset("data",    data=mat.data)
    g.create_dataset("indices", data=mat.indices)
    g.create_dataset("indptr",  data=mat.indptr)
    g.create_dataset("shape",   data=np.array(mat.shape, dtype=np.int64))


def _require_clean_group(parent: h5py.File | h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def save_assets_h5(out_h5: Path,
                   bdry: BD.BoundaryData,
                   W_nodes: np.ndarray,
                   forward_1024,
                   forward_256,
                   inverse_1024,
                   inverse_256,
                   Phi_1024: sparse.csr_matrix,
                   M_1024: sparse.csr_matrix,
                   K_256_disk: np.ndarray,
                   absK_256_disk: np.ndarray,
                   G_256_disk: np.ndarray) -> None:
    (R1024, Z1024, fwd_tid_1024, fwd_bary_1024) = forward_1024
    (R256,  Z256,  fwd_tid_256,  fwd_bary_256)  = forward_256

    (xd1024, yd1024, inside1024, inv_tid_1024, inv_bary_1024, valid1024) = inverse_1024
    (xd256,  yd256,  inside256,  inv_tid_256,  inv_bary_256,  valid256)  = inverse_256

    with h5py.File(out_h5, "w") as f:
        # ---------------- meta ----------------
        f.attrs["algorithm"] = "harmonic_mean_value"
        f.attrs["shared_mapping"] = 1
        f.attrs["source_h5"] = str(H5_IN)
        f.attrs["reference_case"] = str(REF_CASE)
        f.attrs["reference_time_group"] = str(REF_TGROUP)
        f.attrs["notes_mask"] = "valid_mask = inside_disk & (inv_tri_id>=0); outside unit disk should be treated as NaN."
        f.attrs["notes_geometry"] = "K = dPhi/dy (piecewise-constant per triangle), |K| = det(K), G = K^{-1}K^{-T}."

        # ---------------- mesh ----------------
        gmesh = f.create_group("mesh")
        gmesh.create_dataset("pts", data=bdry.pts.astype(np.float64))
        gmesh.create_dataset("tri", data=bdry.tri.astype(np.int64))
        gmesh.attrs["anchor_node"] = int(bdry.anchor_node)
        gmesh.create_dataset("boundary_nodes", data=bdry.b_nodes.astype(np.int64))

        # ---------------- harmonic map ----------------
        gmap = f.create_group("map")
        gmap.create_dataset("W_nodes_real", data=W_nodes.real.astype(np.float64))
        gmap.create_dataset("W_nodes_imag", data=W_nodes.imag.astype(np.float64))

        # ---------------- forward tables (rect grids) ----------------
        gdom = f.create_group("domain")

        g1024 = gdom.create_group("rect_1024")
        g1024.create_dataset("R_rect", data=R1024.astype(np.float64))
        g1024.create_dataset("Z_rect", data=Z1024.astype(np.float64))
        g1024.create_dataset("fwd_tri_id", data=fwd_tid_1024.astype(np.int32))
        g1024.create_dataset("fwd_bary", data=fwd_bary_1024.astype(np.float32))

        g256 = gdom.create_group("rect_256")
        g256.create_dataset("R_rect", data=R256.astype(np.float64))
        g256.create_dataset("Z_rect", data=Z256.astype(np.float64))
        g256.create_dataset("fwd_tri_id", data=fwd_tid_256.astype(np.int32))
        g256.create_dataset("fwd_bary", data=fwd_bary_256.astype(np.float32))

        # ---------------- inverse tables (disk grids) ----------------
        gdisk = f.create_group("disk")

        gd1024 = gdisk.create_group("disk_1024")
        gd1024.create_dataset("xd", data=xd1024.astype(np.float64))
        gd1024.create_dataset("yd", data=yd1024.astype(np.float64))
        gd1024.create_dataset("inside", data=inside1024.astype(np.uint8))
        gd1024.create_dataset("inv_tri_id", data=inv_tid_1024.astype(np.int32))
        gd1024.create_dataset("inv_bary", data=inv_bary_1024.astype(np.float32))
        gd1024.create_dataset("valid_mask", data=valid1024.astype(np.uint8))

        gd256 = gdisk.create_group("disk_256")
        gd256.create_dataset("xd", data=xd256.astype(np.float64))
        gd256.create_dataset("yd", data=yd256.astype(np.float64))
        gd256.create_dataset("inside", data=inside256.astype(np.uint8))
        gd256.create_dataset("inv_tri_id", data=inv_tid_256.astype(np.int32))
        gd256.create_dataset("inv_bary", data=inv_bary_256.astype(np.float32))
        gd256.create_dataset("valid_mask", data=valid256.astype(np.uint8))

        # Geometry tensors on 256 grid
        gd256.create_dataset("K",    data=K_256_disk.astype(np.float64))      # (256,256,2,2)
        gd256.create_dataset("absK", data=absK_256_disk.astype(np.float64))   # (256,256)
        gd256.create_dataset("G",    data=G_256_disk.astype(np.float64))      # (256,256,2,2)

        # ---------------- operators (1024) ----------------
        gops = f.create_group("ops_1024")
        _save_csr(gops, "Phi", Phi_1024)
        _save_csr(gops, "M",   M_1024)

    print(f"[OK] Saved shared harmonic mapping assets to: {out_h5.resolve()}")


# ============================== MAIN ==============================

def main():
    # 1) Geometry + boundary + P1 helper
    bdry, fe = BD.extract_boundary_and_fe(H5_IN, REF_CASE, REF_TGROUP)

    # 2) Harmonic map: physical nodes -> disk nodes (uv)
    W_nodes = solve_harmonic_map(bdry)

    # 3) Forward bary tables on rectangular physical grids (1024 and 256)
    forward_1024 = precompute_forward_rect_bary(bdry.pts, bdry.tri, fe, N=NRECT_1024)
    forward_256  = precompute_forward_rect_bary(bdry.pts, bdry.tri, fe, N=NRECT_256)

    # 4) Inverse bary tables on disk grids (1024 and 256), with corrected valid_mask
    inverse_1024 = precompute_inverse_disk_bary(bdry.pts, bdry.tri, W_nodes, N=NDISK_1024, eps=0.0)
    inverse_256  = precompute_inverse_disk_bary(bdry.pts, bdry.tri, W_nodes, N=NDISK_256,  eps=0.0)

    (xd1024, yd1024, inside1024, inv_tid_1024, inv_bary_1024, valid1024) = inverse_1024
    (xd256,  yd256,  inside256,  inv_tid_256,  inv_bary_256,  valid256)  = inverse_256

    # 5) Operators on 1024: Φ and M = ΦᵀΦ using corrected valid_mask (unit disk semantics)
    Phi_1024, _mask1024 = build_forward_operator(inv_tid_1024, inv_bary_1024, bdry.tri,
                                                 bdry.pts.shape[0], valid_mask=valid1024)
    M_1024 = build_normal_matrix(Phi_1024)

    # 6) Geometry tensors following equations: compute per-element K, |K|, G, then rasterize to 256 disk grid
    K_e, absK_e, G_e = compute_element_K_absK_G(bdry.pts, bdry.tri, W_nodes)
    K_256_disk, absK_256_disk, G_256_disk = rasterize_element_tensors_to_disk(inv_tid_256, valid256, K_e, absK_e, G_e)

    # 7) Save HDF5 assets
    save_assets_h5(H5_OUT, bdry, W_nodes,
                   forward_1024, forward_256,
                   inverse_1024, inverse_256,
                   Phi_1024, M_1024,
                   K_256_disk, absK_256_disk, G_256_disk)


if __name__ == "__main__":
    main()
