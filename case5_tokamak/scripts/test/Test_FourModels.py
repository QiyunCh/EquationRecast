#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test_FourModels.py

Compare 4 models (FNO-M, FNO-L, LocalNO-M, LocalNO-L) on val splits of
4 prefixes (CMOD, CMOD_NewGeo_SO, SPARC, ARC_V02). For each (model, prefix),
evaluate relative L2 on T_e in the PHYSICAL domain over the SAME 20 randomly
chosen val samples. Output: one grouped bar chart (linear + log y-scale).

No fixed-point iteration: one-shot prediction with the pre-computed effective
source already in the dataset.

Pipeline mirrors Plot_FNO.py:
  raw H5 -> normalize -> model -> denormalize dT
  -> T_pred_disk = IC + dT_pred  (256x256 disk)
  -> map_back via HM -> physical grid -> relative L2 on T_e
"""

from __future__ import annotations

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import csv
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

from Data import load_stats, normalize_ic, normalize_source

from Model_FNO_M       import SingleFNO     as FNO_M_Cls,    FNOConfig      as FNO_M_Cfg
from Model_FNO_L       import SingleFNO     as FNO_L_Cls,    FNOConfig      as FNO_L_Cfg
from Model_LocalNO_M   import SingleLocalNO as LNO_M_Cls,    LocalNOConfig  as LNO_M_Cfg
from Model_LocalNO_L_2 import SingleLocalNO as LNO_L_Cls,    LocalNOConfig  as LNO_L_Cfg


# ======================================================================
# CONFIG
# ======================================================================

CONFIG = {
    "data_h5":    Path("Data/Merge/Data_ML_Merged.h5"),
    "stats_json": Path("Data/CalStat/stats_train.json"),

    "models": [
        {"name": "FNO-M",     "cls": FNO_M_Cls, "cfg": FNO_M_Cfg,
         "ckpt": Path("Train/FNO_Med/best_model.pt")},
        {"name": "FNO-L",     "cls": FNO_L_Cls, "cfg": FNO_L_Cfg,
         "ckpt": Path("Train/Model_Large/final_model.pt")},
        {"name": "LocalNO-M", "cls": LNO_M_Cls, "cfg": LNO_M_Cfg,
         "ckpt": Path("Train/LocalNO_Med/best_model.pt")},
        {"name": "LocalNO-L", "cls": LNO_L_Cls, "cfg": LNO_L_Cfg,
         "ckpt": Path("Train/LocalNO_Large/Train_2/best_model.pt")},
    ],

    # display name -> H5 case_id prefix
    "prefixes": [
        ("CMOD",            "CMOD_AT_"),
        ("CMOD_NewGeo_SO",  "CMOD_NewGeo_SO_"),
        ("SPARC",           "SPARC_"),
        ("ARC_V02",         "ARC_"),
    ],

    "hm_paths": {
        "CMOD_AT_":        Path("HM/CMOD_HM.h5"),
        "CMOD_NewGeo_SO_": Path("HM/CMOD_NewGeo_HM.h5"),
        "SPARC_":          Path("HM/SPARC_HM.h5"),
        "ARC_":            Path("HM/ARC_V02_HM.h5"),
    },

    "n_samples":      20,
    "seed":           1234,
    "T_floor":        1e-4,
    "device":         "cuda" if torch.cuda.is_available() else "cpu",

    "phys_res":       512,
    "phys_pad_frac":  0.01,
    "disk_group_256": "disk_256",

    "out_dir":        Path("plot/compare_4models"),
}


# ======================================================================
# DENORMALIZATION  (same as Plot_FNO.py)
# ======================================================================

def denormalize_label(y_norm: np.ndarray, stats: Dict) -> np.ndarray:
    mu    = stats["mu_sigma"]["Y"]["mu"]
    sigma = stats["mu_sigma"]["Y"]["sigma"]
    y0    = stats["scales"]["y0"]
    return np.sinh(np.asarray(y_norm, dtype=np.float64) * sigma + mu) * y0


# ======================================================================
# HARMONIC MAPPING  (256 grid -> physical, same as Plot_FNO.py)
# ======================================================================

def load_hm_mapping(hm_path: Path, grp256: str) -> Dict[str, Any]:
    with h5py.File(hm_path, "r") as f:
        pts = np.asarray(f["mesh/pts"][...], dtype=np.float64)
        tri = np.asarray(f["mesh/tri"][...], dtype=np.int64)
        boundary_nodes = np.asarray(f["mesh/boundary_nodes"][...], dtype=np.int64)
        g = f[f"disk/{grp256}"]
        d256 = {
            "xd":         np.asarray(g["xd"][...], dtype=np.float64),
            "yd":         np.asarray(g["yd"][...], dtype=np.float64),
            "inv_tri_id": np.asarray(g["inv_tri_id"][...], dtype=np.int32),
            "inv_bary":   np.asarray(g["inv_bary"][...], dtype=np.float32),
            "valid_mask": np.asarray(g["valid_mask"][...], dtype=np.uint8),
        }
    return {"pts": pts, "tri": tri, "boundary_nodes": boundary_nodes, "d256": d256}


def map_back_disk_to_physical(pts, tri, inv_tri_id, inv_bary, valid_mask):
    Ny, Nx = inv_tri_id.shape
    Rb = np.full((Ny, Nx), np.nan, dtype=np.float64)
    Zb = np.full((Ny, Nx), np.nan, dtype=np.float64)
    ok = valid_mask.astype(bool)
    if not np.any(ok):
        return Rb, Zb
    t   = inv_tri_id[ok].astype(np.int64)
    lam = inv_bary[ok].astype(np.float64)
    a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
    Rb[ok] = lam[:, 0]*pts[a, 0] + lam[:, 1]*pts[b, 0] + lam[:, 2]*pts[c, 0]
    Zb[ok] = lam[:, 0]*pts[a, 1] + lam[:, 1]*pts[b, 1] + lam[:, 2]*pts[c, 1]
    return Rb, Zb


def precompute_physical_grid(hm: Dict, cfg: Dict) -> Dict[str, Any]:
    d256 = hm["d256"]
    pts, tri, bnodes = hm["pts"], hm["tri"], hm["boundary_nodes"]

    Rb, Zb = map_back_disk_to_physical(
        pts, tri, d256["inv_tri_id"], d256["inv_bary"], d256["valid_mask"],
    )
    ok_geo = d256["valid_mask"].astype(bool) & np.isfinite(Rb) & np.isfinite(Zb)

    pad = cfg["phys_pad_frac"]
    rmin, rmax = float(Rb[ok_geo].min()), float(Rb[ok_geo].max())
    zmin, zmax = float(Zb[ok_geo].min()), float(Zb[ok_geo].max())
    pr = pad * (rmax - rmin + 1e-12)
    pz = pad * (zmax - zmin + 1e-12)

    res = cfg["phys_res"]
    R_grid = np.linspace(rmin - pr, rmax + pr, res)
    Z_grid = np.linspace(zmin - pz, zmax + pz, res)
    RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing="xy")

    poly = MplPath(pts[bnodes, :2], closed=True)
    inside = poly.contains_points(np.c_[RR.ravel(), ZZ.ravel()]).reshape(RR.shape)
    bphys = pts[bnodes, :2]

    return {"Rb": Rb, "Zb": Zb, "ok_geo": ok_geo, "RR": RR, "ZZ": ZZ,
            "inside": inside, "bphys": bphys}


def field_to_physical(field256: np.ndarray, phys_pre: Dict) -> np.ndarray:
    ok = phys_pre["ok_geo"] & np.isfinite(field256)
    Rb, Zb = phys_pre["Rb"], phys_pre["Zb"]
    R_sc, Z_sc, V_sc = Rb[ok], Zb[ok], field256[ok]

    bphys = phys_pre["bphys"]
    V_bnd = griddata(np.c_[R_sc, Z_sc], V_sc, bphys, method="nearest")
    pts_all  = np.vstack([np.c_[R_sc, Z_sc], bphys])
    vals_all = np.concatenate([V_sc, V_bnd])

    field_phys = griddata(pts_all, vals_all,
                          (phys_pre["RR"], phys_pre["ZZ"]), method="linear")
    field_phys[~phys_pre["inside"]] = np.nan
    return field_phys


# ======================================================================
# DATA / SPLIT
# ======================================================================

def build_val_indices_per_prefix(stats: Dict, data_h5: Path,
                                 prefixes: List[Tuple[str, str]]):
    val_set = set(np.array(stats["split"]["val_idx"], dtype=np.int64).tolist())
    with h5py.File(data_h5, "r") as f:
        case_ids = [
            (x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x))
            for x in np.asarray(f["case_id"][...])
        ]
    out = {}
    for _disp, pfx in prefixes:
        out[pfx] = sorted(i for i, cid in enumerate(case_ids)
                          if cid.startswith(pfx) and i in val_set)
    return out, case_ids


def load_raw_sample(h5_path: Path, idx: int) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        return {
            "IC":        np.nan_to_num(np.asarray(f["IC"][idx],        dtype=np.float64)),
            "Source":    np.nan_to_num(np.asarray(f["Source"][idx],    dtype=np.float64)),
            "Label":     np.nan_to_num(np.asarray(f["Label"][idx],     dtype=np.float64)),
            "Benchmark": np.nan_to_num(np.asarray(f["Benchmark"][idx], dtype=np.float64)),
        }


# ======================================================================
# INFERENCE  (single forward, returns disk-grid T_pred / T_true)
# ======================================================================

def run_inference(model, device, stats, raw, mask) -> Dict[str, np.ndarray]:
    ic_norm  = normalize_ic(raw["IC"], stats)
    src_norm = normalize_source(raw["Source"], stats)
    x_np = np.stack([ic_norm, src_norm], axis=0).astype(np.float32)
    x_gpu = torch.from_numpy(x_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_gpu = model(x_gpu)
    pred_norm = pred_gpu.cpu().numpy().astype(np.float64)[0]

    dT_pred = denormalize_label(pred_norm[0], stats)
    T_pred_disk = raw["IC"] + dT_pred
    T_true_disk = raw["Benchmark"].copy()

    m = mask.astype(bool)
    T_pred_disk[~m] = np.nan
    T_true_disk[~m] = np.nan
    return {"T_pred_disk": T_pred_disk, "T_true_disk": T_true_disk}


# ======================================================================
# ERROR METRIC  (relative L2 in physical domain, with T_floor)
# ======================================================================

def relative_L2_phys(T_pred_phys: np.ndarray, T_true_phys: np.ndarray,
                     T_floor: float) -> float:
    valid = np.isfinite(T_pred_phys) & np.isfinite(T_true_phys) \
            & (np.abs(T_true_phys) >= T_floor)
    if not valid.any():
        return float("nan")
    diff = (T_pred_phys - T_true_phys)[valid]
    den  = T_true_phys[valid]
    return float(np.sqrt((diff ** 2).sum())
                 / (np.sqrt((den ** 2).sum()) + 1e-30))


# ======================================================================
# MODEL LOADING
# ======================================================================

def load_model(mcfg: Dict, device: torch.device):
    model = mcfg["cls"](mcfg["cfg"]()).to(device)
    ckpt  = torch.load(mcfg["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k: v for k, v in state.items() if not k.startswith("_")}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {mcfg['name']:>10s}  params={n_params:,}  ckpt={mcfg['ckpt']}")
    return model, n_params


# ======================================================================
# MAIN
# ======================================================================

def main():
    cfg = CONFIG
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"])

    stats = load_stats(cfg["stats_json"])

    # --- mask (shared 256x256) ---
    with h5py.File(cfg["data_h5"], "r") as f:
        mask = np.asarray(f["Mask"], dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[0]

    # --- val indices per prefix ---
    val_by_pfx, case_ids = build_val_indices_per_prefix(
        stats, cfg["data_h5"], cfg["prefixes"]
    )
    for disp, pfx in cfg["prefixes"]:
        print(f"[pool] {disp:>16s}  ({pfx})  val pool = {len(val_by_pfx[pfx])}")

    # --- pick same 20 val samples per prefix (model-agnostic) ---
    rng = random.Random(cfg["seed"])
    chosen: Dict[str, List[int]] = {}
    for disp, pfx in cfg["prefixes"]:
        pool = val_by_pfx[pfx]
        n = min(cfg["n_samples"], len(pool))
        chosen[pfx] = sorted(rng.sample(pool, n))
        print(f"[pick] {disp:>16s}  picked {n} val samples")

    # --- precompute HM physical grids per prefix (shared across models) ---
    print("[HM] loading harmonic maps and precomputing physical grids ...")
    phys_cache: Dict[str, Dict] = {}
    for disp, pfx in cfg["prefixes"]:
        hm = load_hm_mapping(cfg["hm_paths"][pfx], cfg["disk_group_256"])
        phys_cache[pfx] = precompute_physical_grid(hm, cfg)
        print(f"  {disp:>16s} HM ready")

    # --- preload raw H5 samples once (4 models share them) ---
    print("[data] preloading raw samples ...")
    raw_cache: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for _disp, pfx in cfg["prefixes"]:
        for gidx in chosen[pfx]:
            raw_cache[(pfx, gidx)] = load_raw_sample(cfg["data_h5"], gidx)

    # --- evaluate ---
    results: Dict[Tuple[str, str], float] = {}
    raw_rows: List[Tuple[str, str, int, str, float]] = []
    param_counts: Dict[str, int] = {}

    for mcfg in cfg["models"]:
        model, n_params = load_model(mcfg, device)
        param_counts[mcfg["name"]] = n_params

        for disp, pfx in cfg["prefixes"]:
            errs = []
            for gidx in chosen[pfx]:
                raw = raw_cache[(pfx, gidx)]
                res = run_inference(model, device, stats, raw, mask)
                T_pred_phys = field_to_physical(res["T_pred_disk"], phys_cache[pfx])
                T_true_phys = field_to_physical(res["T_true_disk"], phys_cache[pfx])
                e = relative_L2_phys(T_pred_phys, T_true_phys, cfg["T_floor"])
                errs.append(e)
                raw_rows.append((mcfg["name"], disp, gidx, case_ids[gidx], e))
            mean_err = float(np.nanmean(errs))
            results[(mcfg["name"], disp)] = mean_err
            print(f"  {mcfg['name']:>10s} | {disp:>16s} | mean rel L2 = {mean_err:.4e}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- save raw CSV ---
    csv_path = out_dir / "results_raw.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "prefix", "gidx", "case_id", "rel_L2"])
        for r in raw_rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6e}"])
    print(f"[save] raw CSV  -> {csv_path}")

    # --- save summary JSON ---
    summary = {
        "param_counts": param_counts,
        "mean_rel_L2": {f"{m}__{p}": v for (m, p), v in results.items()},
        "n_samples_per_prefix": {disp: len(chosen[pfx])
                                 for disp, pfx in cfg["prefixes"]},
        "T_floor": cfg["T_floor"],
        "seed":    cfg["seed"],
    }
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] summary  -> {out_dir/'results_summary.json'}")

    # --- bar plot (linear + log) ---
    model_names  = [m["name"] for m in cfg["models"]]
    prefix_names = [p[0] for p in cfg["prefixes"]]
    n_pre, n_mod = len(prefix_names), len(model_names)
    width = 0.18
    x = np.arange(n_pre)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for yscale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, mname in enumerate(model_names):
            vals = [results[(mname, p)] for p in prefix_names]
            offset = (i - (n_mod - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=mname, color=colors[i])
            for b, v in zip(bars, vals):
                if np.isfinite(v) and v > 0:
                    ax.text(b.get_x() + b.get_width()/2, v, f"{v*100:.2f}%",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(prefix_names)
        ax.set_ylabel(r"Mean relative $L^2$ on $T_e$ (physical, 20 val samples)")
        ax.set_yscale(yscale)
        ax.legend(frameon=False, ncol=n_mod, loc="upper center",
                  bbox_to_anchor=(0.5, 1.10))
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        out_png = out_dir / f"compare_4models_{yscale}.png"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] plot ({yscale}) -> {out_png}")

    print("[DONE]")


if __name__ == "__main__":
    main()
