# Equation Recast for Canonical Operator Learning Across Parametric PDEs

Code accompanying the manuscript

> Q. Cheng, V. Duruisseaux, C. F. Clauser, M. H. Sahadath, H. Yang, S. Pan,
> N. Ferraro, A. Anandkumar, W. Ji, C. Rea,
> *Equation Recast for Canonical Operator Learning Across Parametric PDEs*.

Equation recast reformulates parametric operator learning as the learning of a
**single canonical operator**. For a parametric PDE $\mathcal{O}(\mathbf{p})[u]=S$
and a reference configuration $\mathbf{p}^*$, the analytically known operator
difference $\mathcal{O}_\delta(\delta\mathbf{p})=\mathcal{O}(\mathbf{p})-\mathcal{O}^*$
is moved into an effective source,

$$\mathcal{O}^*[u] = S - \mathcal{O}_\delta(\delta\mathbf{p})[u] \equiv S_{\mathrm{eff}},$$

and a target configuration is resolved by iterating a *fixed* learned canonical
inverse $\mathcal{G}^*_N\approx(\mathcal{O}^*)^{-1}$:

$$u^{k+1} = \mathcal{G}^*_N\!\left[S_{\mathrm{eff}}^{\,k}\right],
\qquad S_{\mathrm{eff}}^{\,k} = S - \mathcal{O}_\delta(\delta\mathbf{p})[u^{k}].$$

## Repository layout

| Directory | Benchmark | Manuscript items |
|---|---|---|
| [`case1_adr_1d/`](case1_adr_1d) | 1D advection–diffusion–reaction, two-parameter extrapolation | Fig. 2, SI S1, Fig. S.1 |
| [`case2_rd_1d/`](case2_rd_1d) | 1D reaction–diffusion, heterogeneous-data enrichment | Fig. 3a, SI S2 |
| [`case3_helmholtz_1d/`](case3_helmholtz_1d) | 1D Helmholtz, resonance failure diagnostic | Fig. 3b, SI S3 |
| [`case4_ns_2d/`](case4_ns_2d) | 2D steady Navier–Stokes (vorticity), nonlinear extrapolation | Fig. 4, SI S4, Table S.1–S.2, Figs. S.2–S.4 |
| [`case5_tokamak/`](case5_tokamak) | Multi-device tokamak electron-temperature equation | Fig. 5, SI S5, Table S.3, Fig. S.5 |

[`FIGURE_MAP.md`](FIGURE_MAP.md) maps every figure and table in the manuscript
to the script that produces it.

## Data and trained weights

This repository contains **code only**. Datasets, trained checkpoints, and the
saved evaluation outputs are archived on Zenodo:

> DOI: `10.5281/zenodo.22016990`

Fetch and unpack them into the expected locations with

```bash
python download_artifacts.py --case all       # or --case case4_ns_2d
```

Every benchmark can also be regenerated from scratch — each case directory
documents the `GenData → Train → Test → Plot` sequence and the reference solver
is included.

**M3D-C1 simulation data are not released.** The tokamak case
(`case5_tokamak/`) ships the complete pipeline — harmonic mapping, canonical-domain
assembly, models, training, and evaluation — together with the trained weights for
all four architectures and their normalization statistics on Zenodo. Only the
underlying simulation outputs are withheld: they and the device configurations
they encode are subject to third-party and machine-specific restrictions, and
requests should be directed to the corresponding authors. The pipeline runs
unchanged on any equivalent finite-element dataset exposing the same field names
(see [`case5_tokamak/README.md`](case5_tokamak/README.md)).

## Installation

```bash
conda create -n recast python=3.10
conda activate recast
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

A GPU is required in practice for `case4_ns_2d` and `case5_tokamak`; the 1D
benchmarks (cases 1–3) run on CPU in minutes.

## Conventions shared by all cases

* Test sources are drawn with a fixed seed (13) so that every method is
  evaluated on identical inputs.
* The recast fixed point uses Aitken $\Delta^2$ relaxation everywhere. No
  relaxation weight or damping factor is tuned per parameter value in any case.
  Tolerances and iteration caps are stated per case in the SI and repeated in
  each case README.
* Accuracy is the relative $L^2$ error against the high-fidelity reference. The
  relative PDE residual $\lVert R\rVert_2/\lVert S\rVert_2$ is computed
  spectrally from the prediction alone and needs no reference solution.
* Neural-operator architectures follow the *NeuralOperator* library
  conventions; FNO is the primary learner, LocalNO is additionally evaluated in
  the tokamak case.

## License

MIT — see [LICENSE](LICENSE).
