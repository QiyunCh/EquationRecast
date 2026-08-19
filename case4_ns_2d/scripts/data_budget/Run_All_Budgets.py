#!/usr/bin/env python3
"""
Run_All_Budgets.py — Sequentially train all Test 4 models for budgets [200,500,1000,1500].

For each budget:
  - canonical FNO (data) → canonical FNO (PINN finetune)
  - parametric FNO (Re~U[50,400])
  - PINO (Re~U[50,400], λ=0.5)
At budget=1500 also:
  - redistributed recast (data) → PINN finetune

Models saved to models/. Logs to results/training_log.txt.
"""
from __future__ import annotations
import os, subprocess, sys, time


BUDGETS = [200, 500, 1000, 1500]
CAN_DATA = "canonical/data_canonical_N1500.h5"
PAR_DATA = "parametric/data_parametric_N1500.h5"
REDISTRIB_DATA = "redistributed/data_redistributed.h5"

PY = sys.executable


def run(cmd):
    print(f">>> {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    print(f"    (rc={proc.returncode}, {time.time()-t0:.0f}s)")
    return proc.returncode


def main():
    os.makedirs("models/canonical", exist_ok=True)
    os.makedirs("models/parametric", exist_ok=True)
    os.makedirs("models/pino", exist_ok=True)
    os.makedirs("models/pino_ext", exist_ok=True)
    os.makedirs("models/redistributed", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Required data
    if not os.path.exists(CAN_DATA):
        print(f"MISSING {CAN_DATA}: run GenData.py --target canonical first"); sys.exit(1)
    if not os.path.exists(PAR_DATA):
        print(f"MISSING {PAR_DATA}: run GenData.py --target parametric first"); sys.exit(1)

    for B in BUDGETS:
        print(f"\n========== BUDGET {B} ==========")

        # Canonical FNO (data only)
        can_data_ck = f"models/canonical/canonical_data_N{B}.pt"
        if not os.path.exists(can_data_ck):
            run([PY, "Train_Canonical_Data.py",
                 "--data", CAN_DATA, "--budget", str(B), "--out", can_data_ck])
        else:
            print(f"  skip (exists): {can_data_ck}")

        # Canonical FNO (PINN finetune from above)
        can_pinn_ck = f"models/canonical/canonical_pinn_N{B}.pt"
        if not os.path.exists(can_pinn_ck) and os.path.exists(can_data_ck):
            run([PY, "Train_Canonical_PINN.py",
                 "--data", CAN_DATA, "--budget", str(B),
                 "--in_ckpt", can_data_ck, "--out", can_pinn_ck])
        else:
            print(f"  skip (exists): {can_pinn_ck}")

        # Parametric FNO (Re~U[50,400])
        par_ck = f"models/parametric/parametric_N{B}.pt"
        if not os.path.exists(par_ck):
            run([PY, "Train_Parametric.py",
                 "--data", PAR_DATA, "--budget", str(B), "--out", par_ck])
        else:
            print(f"  skip (exists): {par_ck}")

        # PINO (Re~U[50,400])
        pino_ck = f"models/pino/pino_N{B}.pt"
        if not os.path.exists(pino_ck):
            run([PY, "Train_PINO.py",
                 "--data", PAR_DATA, "--budget", str(B), "--out", pino_ck,
                 "--lambda_res", "0.5"])
        else:
            print(f"  skip (exists): {pino_ck}")

        # ext-PINO (Re~U[50,400], same data as our PINO; uses neuraloperator/physics_informed FNO2d)
        pino_ext_ck = f"models/pino_ext/pino_ext_N{B}.pt"
        if not os.path.exists(pino_ext_ck):
            run([PY, "Train_PINO_ExtRepo.py",
                 "--data", PAR_DATA, "--budget", str(B), "--out", pino_ext_ck,
                 "--lambda_res", "0.5"])
        else:
            print(f"  skip (exists): {pino_ext_ck}")

    # Redistributed recast (only at B=1500)
    # NOTE: skipping PINN finetune for redistributed because input normalization
    # uses the effective-source range, which differs from canonical S range —
    # PINN-finetuning with canonical data would feed mis-normalized inputs.
    # The data-trained redistributed model is used as-is.
    if os.path.exists(REDISTRIB_DATA):
        print(f"\n========== REDISTRIBUTED RECAST (B=1500) ==========")
        rec_data_ck = "models/redistributed/redistributed_data.pt"
        if not os.path.exists(rec_data_ck):
            run([PY, "Train_Redistributed_Recast.py",
                 "--data", REDISTRIB_DATA, "--out", rec_data_ck,
                 "--canonical_data", CAN_DATA])

    print("\nAll trainings done.")


if __name__ == "__main__":
    main()
