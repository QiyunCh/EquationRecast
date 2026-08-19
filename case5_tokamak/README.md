# Case 5 — Multi-device tokamak electron-temperature equation (Fig. 5, SI S5)

> **The M3D-C1 simulation dataset is not released, and neither are weights
> trained on it.** The complete pipeline is provided so the method can be
> reproduced on any equivalent finite-element dataset. The M3D-C1 outputs and
> the device configurations they encode are subject to third-party and
> machine-specific restrictions; requests should be directed to the
> corresponding authors.

Electron-temperature (energy) equation extracted from high-fidelity M3D-C1
simulations of four devices: Alcator C-Mod, Alcator C-Mod with a flat divertor,
SPARC, and ARC_V2A.

Each physical domain is mapped harmonically to a shared unit-disk canonical
domain with coordinates $\boldsymbol{\xi}$ and Jacobian
$\mathrm{K}=\partial\mathbf{x}/\partial\boldsymbol{\xi}$, so the canonical-domain
equation is

$$\frac{\partial \widetilde{T}_e}{\partial t}
= \widetilde{Q}_{\mathrm{tot}}
- \frac{\tilde\sigma_e}{\tilde n_e}\widetilde{T}_e
+ \frac{\gamma-1}{\tilde n_e}\,\mathcal{L}_{\mathrm{K}}\!\left[\tilde\kappa_\perp,\widetilde{T}_e\right],
\qquad
\mathcal{L}_{\mathrm{K}}[\cdot]=\frac{1}{|\mathrm{K}|}\nabla_{\boldsymbol{\xi}}\!\cdot\!\left(|\mathrm{K}|\,\tilde\kappa_\perp\,\mathrm{K}^{-1}\mathrm{K}^{-\mathsf{T}}\nabla_{\boldsymbol{\xi}}\,\cdot\right).$$

Geometry and plasma-coefficient variation then enter the *same* effective source,

$$\widetilde{S}_{\mathrm{eff}}
= \widetilde{Q}_{\mathrm{tot}}
- \widetilde{\mathcal{O}}_{\delta}(\delta\mathbf{p},\delta\mathrm{K})\!\left[\widetilde{T}_e\right],$$

and the learned canonical object is the one-step update
$(\widetilde{T}_e^{\,n},\widetilde{S}_{\mathrm{eff}}^{\,n})\mapsto\Delta\widetilde{T}_e^{\,n}$.
Alcator C-Mod provides the reference geometry $\mathrm{K}^*$ and the reference
coefficients $\mathbf{p}^*=(\tilde n_e^*,\tilde\sigma_e^*,\tilde\kappa_\perp^*)$.

All four geometries are included in training: the benchmark tests multi-device
data unification, not extrapolation to an unseen device.

## Contents

| Path | Role |
|---|---|
| `scripts/mesh/Mesh.py`, `FEInterp.py`, `BD.py` | 2D finite-element mesh container, high-order FE basis evaluation, M3D-C1 field/boundary containers |
| `scripts/mesh/HM_Mapping.py` | harmonic map of a device domain to the unit disk; forward and inverse tables |
| `scripts/mesh/HM_Test.py`, `Plot_Mesh.py` | mapping checks and mesh figures (physical vs canonical) |
| `scripts/data/Data.py` | canonical-domain dataset assembly: mapping, Jacobian metrics, effective sources, masked normalization, `Dataset`/`DataLoader` |
| `scripts/data/Compute_train_stats.py` | normalization statistics over the training split |
| `scripts/models/Model_FNO_{M,L}.py`, `Model_LocalNO_{M,L}.py` | the four architectures of Table S.3 |
| `scripts/train/Train_Server.py`, `Model.py`, `Train.sbatch` | multi-GPU DDP training (masked MSE, Adam, lr $10^{-3}$, ×0.5 every 1000 epochs) |
| `scripts/test/Plot_LocalNO_Panel.py`, `Mesh_Plot.py` | Fig. 5 panels: canonical-domain $\Delta T_e$, mapped-back $T_e$, benchmark, pointwise error, and the mesh columns |
| `scripts/test/Compare_4Models.py`, `Test_FourModels.py` | architecture ablation on $\Delta T_e$ (Table S.3, Fig. S.5b) and on the full $T_e$ field (Fig. S.5a) |

## Settings (SI S5)

| | |
|---|---|
| Pairs | 3,464 one-step pairs across four devices after removing initialization transients and numerically corrupted states |
| Canonical grid | $256\times256$ unit disk; loss evaluated only on valid mapped pixels via a binary mask |
| Split | 10% held out for validation |
| Medium models | 32 modes, width 64 |
| Large models | 48 modes, width 128 |
| Main text | LocalNO-L |

`Train.sbatch` targets a SLURM cluster (2 nodes × 4 GPUs); change the partition
name and the conda environment before use.

## Expected inputs

`scripts/data/Data.py` reads per-device HDF5 files containing, on the FE mesh,
the electron temperature $T_e$, density $n_e$, ionization term $\sigma_e$,
perpendicular thermal diffusivity $\kappa_\perp$, the total heating/cooling
source $Q_{\mathrm{tot}}$, and the mesh connectivity, together with the harmonic
map produced by `scripts/mesh/HM_Mapping.py`. Normalization statistics are
stored alongside as JSON. Substituting an equivalent dataset with the same field
names is sufficient to run the pipeline unchanged.

## Usage (with a compatible dataset in place)

```bash
cd scripts
python mesh/HM_Mapping.py     # harmonic map + Jacobian tables, per device
python data/Data.py           # canonical-domain one-step pairs and effective sources
python data/Compute_train_stats.py
sbatch train/Train.sbatch     # or: torchrun --nproc_per_node=<n> train/Train_Server.py
python test/Plot_LocalNO_Panel.py   # Fig. 5 result columns
python test/Mesh_Plot.py            # Fig. 5 mesh columns
python test/Compare_4Models.py      # Table S.3 / Fig. S.5b
python test/Test_FourModels.py      # Fig. S.5a
```

The mesh utilities are shared by all four devices; the copies here are the
Alcator C-Mod instances, which differ from the other devices only in input paths
and boundary-extraction parameters.
