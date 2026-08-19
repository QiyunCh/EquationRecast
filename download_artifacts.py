#!/usr/bin/env python3
"""
download_artifacts.py — fetch datasets, trained weights, and saved evaluation
outputs from the Zenodo archive and place them where the scripts expect them.

The repository holds code only. Everything else lives in one Zenodo record,
laid out exactly like the ``case*/`` tree here.

Usage
-----
    python download_artifacts.py --list                  # show what is available
    python download_artifacts.py --case all              # everything (~6.8 GB)
    python download_artifacts.py --case case4_ns_2d      # one case
    python download_artifacts.py --case case4_ns_2d --skip-budget   # omit the 5.8 GB ablation
    python download_artifacts.py --case case1_adr_1d --kind checkpoints

Nothing is downloaded for ``case5_tokamak``: the M3D-C1 simulation dataset and
the weights derived from it are not part of the release.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Zenodo record id, assigned on publication.
ZENODO_RECORD = "22016990"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

ROOT = Path(__file__).resolve().parent

# archive path -> (destination relative to repo root, approximate size in MB)
FILES: dict[str, tuple[str, int]] = {
    # ---------------- case 1: ADR ----------------
    "case1_adr_1d/data/data_canonical.h5": ("case1_adr_1d/scripts/data_canonical.h5", 3),
    "case1_adr_1d/checkpoints/best_fno1d.pt": ("case1_adr_1d/scripts/best_fno1d.pt", 6),
    "case1_adr_1d/peda_scan_bandlimit_results.h5": ("case1_adr_1d/scripts/peda_scan_bandlimit_results.h5", 1),
    # ---------------- case 2: reaction-diffusion ----------------
    "case2_rd_1d/data/data_beta_0.785.h5": ("case2_rd_1d/scripts/data_beta_0.785.h5", 6),
    "case2_rd_1d/data/data_beta_0.050.h5": ("case2_rd_1d/scripts/data_beta_0.050.h5", 3),
    "case2_rd_1d/data/data_merged_beta_0.785_0.050.h5": ("case2_rd_1d/scripts/data_merged_beta_0.785_0.050.h5", 6),
    "case2_rd_1d/checkpoints/best_fno1d_model1.pt": ("case2_rd_1d/scripts/best_fno1d_model1.pt", 6),
    "case2_rd_1d/checkpoints/best_fno1d_model2.pt": ("case2_rd_1d/scripts/best_fno1d_model2.pt", 6),
    "case2_rd_1d/results/recast_rd_compare_2models.h5": ("case2_rd_1d/scripts/recast_rd_compare_2models.h5", 4),
    # ---------------- case 3: Helmholtz ----------------
    "case3_helmholtz_1d/data/data_0.785.h5": ("case3_helmholtz_1d/scripts/data_0.785.h5", 3),
    "case3_helmholtz_1d/data/data_1.099.h5": ("case3_helmholtz_1d/scripts/data_1.099.h5", 3),
    "case3_helmholtz_1d/data/data_merged_0.785_1.099.h5": ("case3_helmholtz_1d/scripts/data_merged_0.785_1.099.h5", 6),
    "case3_helmholtz_1d/checkpoints/best_fno1d_model1.pt": ("case3_helmholtz_1d/scripts/best_fno1d_model1.pt", 6),
    "case3_helmholtz_1d/checkpoints/best_fno1d_model2.pt": ("case3_helmholtz_1d/scripts/best_fno1d_model2.pt", 6),
    "case3_helmholtz_1d/results/recast_helmholtz_compare_2models.h5": ("case3_helmholtz_1d/scripts/recast_helmholtz_compare_2models.h5", 4),
    # ---------------- case 4: Navier-Stokes ----------------
    "case4_ns_2d/data/data_canonical.h5": ("case4_ns_2d/scripts/data_canonical.h5", 52),
    "case4_ns_2d/data/data_parametric.h5": ("case4_ns_2d/scripts/data_parametric.h5", 53),
    "case4_ns_2d/checkpoints/best_fno2d_canonical_dataonly.pt": ("case4_ns_2d/scripts/models/best_fno2d_canonical_dataonly.pt", 256),
    "case4_ns_2d/checkpoints/best_fno2d_parametric.pt": ("case4_ns_2d/scripts/models/best_fno2d_parametric.pt", 256),
    "case4_ns_2d/checkpoints/best_fno2d_pino.pt": ("case4_ns_2d/scripts/models/best_fno2d_pino.pt", 256),
    "case4_ns_2d/results/test3_compare.h5": ("case4_ns_2d/scripts/results/test3_compare.h5", 3),
    "case4_ns_2d/results/test3_fields.h5": ("case4_ns_2d/scripts/results/test3_fields.h5", 2),
    "case4_ns_2d/results/test3_matched_accuracy_time.h5": ("case4_ns_2d/scripts/results/test3_matched_accuracy_time.h5", 1),
    "case4_ns_2d/results/test3_aitken_vs_anderson.h5": ("case4_ns_2d/scripts/results/test3_aitken_vs_anderson.h5", 1),
}

# The data-budget ablation is large and separable.
BUDGET_FILES: dict[str, tuple[str, int]] = {
    "case4_ns_2d/data_budget/data/data_canonical_N1500.h5": ("case4_ns_2d/scripts/data_budget/canonical/data_canonical_N1500.h5", 469),
    "case4_ns_2d/data_budget/data/data_parametric_N1500.h5": ("case4_ns_2d/scripts/data_budget/parametric/data_parametric_N1500.h5", 469),
    "case4_ns_2d/data_budget/data/data_redistributed.h5": ("case4_ns_2d/scripts/data_budget/redistributed/data_redistributed.h5", 469),
    "case4_ns_2d/data_budget/results/test4_compare.h5": ("case4_ns_2d/scripts/data_budget/results/test4_compare.h5", 1),
    "case4_ns_2d/data_budget/results/recast_convergence.h5": ("case4_ns_2d/scripts/data_budget/results/recast_convergence.h5", 1),
}
for _budget in (200, 500, 1000, 1500):
    BUDGET_FILES[f"case4_ns_2d/data_budget/checkpoints/canonical_data_N{_budget}.pt"] = (
        f"case4_ns_2d/scripts/data_budget/models/canonical/canonical_data_N{_budget}.pt", 256)
    BUDGET_FILES[f"case4_ns_2d/data_budget/checkpoints/parametric_N{_budget}.pt"] = (
        f"case4_ns_2d/scripts/data_budget/models/parametric/parametric_N{_budget}.pt", 256)
    BUDGET_FILES[f"case4_ns_2d/data_budget/checkpoints/pino_N{_budget}.pt"] = (
        f"case4_ns_2d/scripts/data_budget/models/pino/pino_N{_budget}.pt", 256)
BUDGET_FILES["case4_ns_2d/data_budget/checkpoints/redistributed_data_N200.pt"] = (
    "case4_ns_2d/scripts/data_budget/models/redistributed/redistributed_data_N200.pt", 256)
BUDGET_FILES["case4_ns_2d/data_budget/checkpoints/redistributed_data_N500.pt"] = (
    "case4_ns_2d/scripts/data_budget/models/redistributed/redistributed_data_N500.pt", 256)
BUDGET_FILES["case4_ns_2d/data_budget/checkpoints/redistributed_data_N1000.pt"] = (
    "case4_ns_2d/scripts/data_budget/models/redistributed/redistributed_data_N1000.pt", 256)
BUDGET_FILES["case4_ns_2d/data_budget/checkpoints/redistributed_data.pt"] = (
    "case4_ns_2d/scripts/data_budget/models/redistributed/redistributed_data.pt", 256)

# ADR canonical-point ablation (six reference configurations).
for _p in ("Pe1_Da1", "Pe2_Da2", "Pe2_Da4", "Pe2_Da6", "Pe10_Da10", "Pe25_Da2"):
    base = "case1_adr_1d/ablation_canonical_point"
    dest = "case1_adr_1d/scripts/ablation_canonical_point"
    FILES[f"{base}/data/data_{_p}.h5"] = (f"{dest}/data/data_{_p}.h5", 4)
    FILES[f"{base}/checkpoints/fno_{_p}_stage1.pt"] = (f"{dest}/models/fno_{_p}_stage1.pt", 6)
    FILES[f"{base}/results/scan_{_p}_stage1.h5"] = (f"{dest}/results/scan_{_p}_stage1.h5", 1)

CASES = ("case1_adr_1d", "case2_rd_1d", "case3_helmholtz_1d", "case4_ns_2d")


def selected(case: str, kind: str | None, skip_budget: bool) -> dict[str, tuple[str, int]]:
    table = dict(FILES)
    if not skip_budget:
        table.update(BUDGET_FILES)
    out = {}
    for src, (dst, mb) in table.items():
        if case != "all" and not src.startswith(case):
            continue
        if kind and f"/{kind}/" not in f"/{src}":
            continue
        out[src] = (dst, mb)
    return out


def zenodo_name(src: str) -> str:
    """Zenodo stores files flat, so the archive tree is encoded in the file name:
    ``case4_ns_2d/checkpoints/x.pt`` is deposited as
    ``case4_ns_2d__checkpoints__x.pt``."""
    return src.replace("/", "__")


def fetch(src: str, dst: Path) -> None:
    if dst.exists():
        print(f"  exists, skipping: {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ZENODO_BASE}/{zenodo_name(src)}?download=1"
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  {src} -> {dst.relative_to(ROOT)}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"download failed ({exc.code}) for {src}.\n"
            f"If the record id is still the placeholder, edit ZENODO_RECORD "
            f"at the top of this script."
        ) from exc
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="all", choices=("all",) + CASES)
    ap.add_argument("--kind", choices=("data", "checkpoints", "results"),
                    help="restrict to one artifact kind")
    ap.add_argument("--skip-budget", action="store_true",
                    help="omit the 5.8 GB Navier-Stokes data-budget ablation")
    ap.add_argument("--list", action="store_true", help="list files without downloading")
    args = ap.parse_args()

    table = selected(args.case, args.kind, args.skip_budget)
    if not table:
        raise SystemExit("nothing selected")
    total = sum(mb for _, mb in table.values())
    print(f"{len(table)} files, approximately {total / 1024:.2f} GB")
    if args.list:
        for src, (dst, mb) in sorted(table.items()):
            print(f"  {mb:5d} MB  {zenodo_name(src)}")
        return
    if ZENODO_RECORD == "XXXXXXX":
        raise SystemExit("ZENODO_RECORD is still the placeholder; set it first.")
    for src, (dst, _) in sorted(table.items()):
        fetch(src, ROOT / dst)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
