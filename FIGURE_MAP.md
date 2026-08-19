# Figure and table map

Every display item in the manuscript, the script that produces it, and the
Zenodo artifacts it needs. Paths are relative to the repository root; artifact
paths are relative to the Zenodo archive (see `download_artifacts.py`).

## Main text

| Item | Content | Script | Artifacts needed |
|---|---|---|---|
| Fig. 1 | Schematic of the recast formulation | — (drawn, no code) | — |
| Fig. 2a | ADR mean relative $L^2$ over $(\mathrm{Pe},\mathrm{Da})\in[1,20]^2$ | `case1_adr_1d/scripts/Test_Recast.py` → `Plot_3Contours.py` | `case1_adr_1d/checkpoints/best_fno1d.pt`, `case1_adr_1d/data/data_canonical.h5` |
| Fig. 2b | ADR mean fixed-point iteration count, same domain | same run as Fig. 2a (`Plot_3Contours.py`) | `case1_adr_1d/peda_scan_bandlimit_results.h5` |
| Fig. 3a | Reaction–diffusion, Model 1 vs Model 2 over $k$ | `case2_rd_1d/scripts/Test_Compare.py` | `case2_rd_1d/checkpoints/*.pt` |
| Fig. 3b | Helmholtz, error and iteration count near resonances | `case3_helmholtz_1d/scripts/Test_Compare.py` | `case3_helmholtz_1d/checkpoints/*.pt` |
| Fig. 4 | NS: accuracy and PDE residual vs Re; error spectra and error fields | `case4_ns_2d/scripts/Test_Compare.py` → `Plot_NS_Baseline_Variants.py` (`make_variant("A_errspec", fields_mode="absdiff")`) | `case4_ns_2d/checkpoints/*.pt`, `case4_ns_2d/results/test3_compare.h5`, `test3_fields.h5` |
| Fig. 5 | Tokamak: four geometries, canonical-domain prediction and physical-domain error | `case5_tokamak/scripts/test/Plot_LocalNO_Panel.py` + `Mesh_Plot.py` | M3D-C1 data — **not released** |

## Supplementary Information

| Item | Content | Script | Artifacts needed |
|---|---|---|---|
| Fig. S.1 | ADR canonical-point ablation, six reference configurations over $[1,50]^2$ | `case1_adr_1d/scripts/ablation_canonical_point/Test_Scan.py` → `Plot_1pct_Boundary.py` | `case1_adr_1d/ablation_canonical_point/{checkpoints,results}` |
| Table S.1 | NS relaxation-scheme ablation (Aitken / Anderson / under-relaxation) | `case4_ns_2d/scripts/ablations/Test_AitkenAnderson.py` | `case4_ns_2d/checkpoints/best_fno2d_canonical_dataonly.pt` |
| Fig. S.2 | Same ablation vs Reynolds number | `case4_ns_2d/scripts/ablations/Test_AitkenAnderson.py` → `Plot_Test3.py` | `case4_ns_2d/results/test3_aitken_vs_anderson.h5` |
| Fig. S.3 | Convergence trajectories and inference cost at matched accuracy | `case4_ns_2d/scripts/data_budget/Test_RecastConvergence.py` | `case4_ns_2d/data_budget/results/recast_convergence.h5` |
| Table S.2 | NS data-budget ablation, four model families × four budgets | `case4_ns_2d/scripts/data_budget/Run_All_Budgets.py` → `Test_Compare.py` | `case4_ns_2d/data_budget/checkpoints/*` (16), `data_budget/data/*` |
| Fig. S.4 | Budget ablation resolved in Reynolds number | `case4_ns_2d/scripts/data_budget/Plot_Test4.py` | `case4_ns_2d/data_budget/results/test4_compare.h5` |
| Table S.3, Fig. S.5 | Tokamak architecture ablation, FNO/LocalNO at two scales | `case5_tokamak/scripts/test/Compare_4Models.py`, `Test_FourModels.py` | M3D-C1 data — **not released** |

## Supporting analyses referenced in the text but not shown

| Claim | Script |
|---|---|
| NS error growth is driven by higher-amplitude effective sources and more nonlinear solution regimes, not by unresolved small scales | `case4_ns_2d/scripts/ablations/Test_EffSourceDiag.py` |
| Error fields and difference maps between methods | `case4_ns_2d/scripts/ablations/Plot_NS_Diff.py` |
| Radial energy/error spectra of the vorticity fields | `case4_ns_2d/scripts/ablations/Test_FieldSpectrum.py` |
| Recast self-consistency check at the canonical parameter | `case4_ns_2d/scripts/data_budget/Test_SelfConsistency.py` |
