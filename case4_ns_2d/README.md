# Case 4 — 2D steady Navier–Stokes, vorticity form (Fig. 4, SI S4)

$$\mathbf{u}\cdot\nabla\omega = \frac{1}{\mathrm{Re}}\Delta\omega + S,
\qquad (x,y)\in[0,1)^2\ \text{periodic},\quad \Delta\psi=-\omega,\ \mathbf{u}=\nabla^\perp\psi$$

Canonical Reynolds number $\mathrm{Re}^*=250$; the recast moves the viscous
difference into the source,

$$\mathcal{O}^*[\omega]=S_{\mathrm{eff}},\qquad
S_{\mathrm{eff}} = S+\left(\frac{1}{\mathrm{Re}}-\frac{1}{\mathrm{Re}^*}\right)\Delta\omega,$$

with the Laplacian evaluated spectrally and the fixed point solved by Aitken
$\Delta^2$ relaxation.

## Settings (SI S4)

| | |
|---|---|
| Reference solver | Fourier pseudo-spectral, $128\times128$, 2/3 de-aliasing, IMEX pseudo-time, $\Delta t=0.2$, stop at relative residual $10^{-6}$ or 2500 steps |
| Forcing | filtered white noise, power $\propto\lvert k\rvert^{-2}$, radial band $2\le\lvert k\rvert\le21$, zero mean, unit standard deviation |
| Canonical data | 200 samples at $\mathrm{Re}^*=250$ |
| Baseline data | 200 samples, $\mathrm{Re}\sim U[200,300]$, Re supplied as a constant input channel |
| Model | FNO2d, 4 Fourier layers, $32\times32$ modes, width 64; 90/10 train/validation split |
| PINO | same samples and inputs plus a PDE-residual term, $\lambda=0.5$ |
| Evaluation | 20 held-out forcing fields, $\mathrm{Re}\in[50,400]$; relative $L^2$ and normalized PDE residual $\lVert R\rVert_2/\lVert S\rVert_2$ |

## Main comparison (Fig. 4)

```bash
cd scripts
python VorticityNS_2D.py             # canonical dataset at Re* = 250
python VorticityNS_2D_parametric.py  # parametric dataset, Re ~ U[200,300]
python Train_Canonical.py            # canonical FNO       -> models/best_fno2d_canonical_dataonly.pt
python Train_Parametric.py           # parametric FNO      -> models/best_fno2d_parametric.pt
python Train_PINO.py                 # PINO (lambda = 0.5) -> models/best_fno2d_pino.pt
python Test_Compare.py               # Re scan -> results/test3_compare.h5, test3_fields.h5
python -c "import Plot_NS_Baseline_Variants as P; P.make_variant('A_errspec', fields_mode='absdiff')"
```

The last command writes Fig. 4: accuracy and PDE residual versus Reynolds
number on top, and per-Reynolds error spectra with absolute error fields below.
`Plot_NS_Baseline.py` produces an earlier single-panel-per-row layout of the
same data.

> `Test_Compare.py` also lists a `canonical_pinn` entry — a canonical model
> fine-tuned with a PDE-residual term. It is not reported in the manuscript;
> delete the entry from `MODELS` or ignore its columns.

## Relaxation-scheme ablation (Table S.1, Fig. S.2)

```bash
cd scripts/ablations
python Test_AitkenAnderson.py   # Aitken vs Anderson (m=3) vs under-relaxation (omega=0.5)
python Plot_Test3.py            # Fig. S.2
```

15 Reynolds numbers × 20 sources = 300 cases, tolerance $10^{-5}$, maximum 300
iterations.

## Convergence and inference cost (Fig. S.3)

```bash
cd scripts/data_budget
python Test_RecastConvergence.py   # convergence trajectories + matched-accuracy timing
```

The recast is timed **batched over the 20 test sources and reported per
source**; the numerical solver is timed per source because its iteration count
adapts to each source. The matched-accuracy target is the per-Reynolds 95%
plateau of the recast error trajectory.

## Data-budget and heterogeneous-data ablation (Table S.2, Fig. S.4)

Budgets $N\in\{200,500,1000,1500\}$ for four model families: canonical recast,
full-range parametric FNO ($\mathrm{Re}\in[50,400]$), PINO, and the
**redistributed recast**, whose training inputs are heterogeneous samples from
$\mathrm{Re}\in\{50,100,200,300,400\}$ mapped into the canonical representation,

$$S_{\mathrm{eff},i} = S_i + \left(\frac{1}{\mathrm{Re}_i}-\frac{1}{\mathrm{Re}^*}\right)\Delta\omega_i .$$

```bash
cd scripts/data_budget
python GenData.py           # canonical / full-range / redistributed datasets (N = 1500 each)
python Run_All_Budgets.py   # trains every family at every budget
python Test_Compare.py      # Table S.2 -> results/test4_compare.h5
python Plot_Test4.py        # Fig. S.4
```

`Run_All_Budgets.py` additionally trains the PINN-finetuned canonical variant,
which is not reported in the manuscript.

## Artifacts (Zenodo)

```
case4_ns_2d/data/{data_canonical.h5, data_parametric.h5}
case4_ns_2d/checkpoints/{best_fno2d_canonical_dataonly.pt, best_fno2d_parametric.pt, best_fno2d_pino.pt}
case4_ns_2d/results/{test3_compare.h5, test3_fields.h5, test3_matched_accuracy_time.h5, test3_aitken_vs_anderson.h5}
case4_ns_2d/data_budget/data/{data_canonical_N1500.h5, data_parametric_N1500.h5, data_redistributed.h5}
case4_ns_2d/data_budget/checkpoints/   (16 checkpoints: 4 families x 4 budgets)
case4_ns_2d/data_budget/results/{test4_compare.h5, recast_convergence.h5}
```

Each FNO2d checkpoint is ~256 MB (67.1 M parameters). Place `checkpoints/`
contents in `scripts/models/` and `data_budget/checkpoints/` contents in
`scripts/data_budget/models/<family>/`, or let `download_artifacts.py` do it.

Runtime: dataset generation ~2 h per 200 samples on one GPU; each training run
~1–3 h on one A100; the full Re scan for three models ~20 min.
