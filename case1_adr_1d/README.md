# Case 1 — 1D advection–diffusion–reaction (Fig. 2, SI S1)

$$-u''(x) + \mathrm{Pe}\,u'(x) + \mathrm{Da}\,u(x) = S(x), \qquad x\in[0,1)\ \text{periodic}$$

Canonical configuration $(\mathrm{Pe}^*,\mathrm{Da}^*) = (4,2)$. Parameter
variation enters only through the effective source,

$$\mathcal{O}^*[u] = S - (\mathrm{Pe}-\mathrm{Pe}^*)\,u' - (\mathrm{Da}-\mathrm{Da}^*)\,u,$$

evaluated spectrally and solved by fixed-point iteration with Aitken $\Delta^2$
relaxation.

## Settings (SI S1)

| | |
|---|---|
| Grid | $N=201$, Fourier spectral reference solver |
| Sources | periodic GRF, $E(\kappa)\propto\exp[-(\ell\kappa)^2/2]$, $\ell=0.08$, rescaled to $[-1,1]$ |
| Training data | 1000 canonical samples (900 train / 100 validation) |
| Model | FNO1d, 3 Fourier layers, 64 modes, width 64; Adam, MSE, 1000 epochs |
| Scan | $(\mathrm{Pe},\mathrm{Da})\in[1,20]^2$, unit spacing (400 configurations), 20 test sources each |
| Iteration | Aitken $\Delta^2$, tolerance $10^{-5}$, max 100 iterations |

## Reproduce

```bash
cd scripts
python ADR.py           # canonical dataset at (Pe*, Da*) = (4, 2)
python Train.py         # canonical FNO1d  -> best_fno1d.pt
python Test_Recast.py   # Pe-Da scan -> peda_scan_bandlimit_results.h5
python Plot_3Contours.py  # Fig. 2a (error + marginals), Fig. 2b (iteration count)
```

`Plot_3Contours.py` writes the two panels that are composited into Fig. 2, plus
the effective-source ratio contour used in the discussion of the recast
correction magnitude.

## Canonical-point ablation (SI S1.1, Fig. S.1)

Six reference configurations
$(\mathrm{Pe}^*,\mathrm{Da}^*)\in\{(1,1),(2,2),(2,4),(2,6),(10,10),(25,2)\}$
over the extended domain $(\mathrm{Pe},\mathrm{Da})\in[1,50]^2$, all trained on
the same source distribution and dataset size, with the iteration cap raised to
300.

```bash
cd scripts/ablation_canonical_point
python GenData.py        # one dataset per canonical configuration
python Train_Stage1.py   # one FNO per canonical configuration (1000 epochs, Adam, lr 1e-3)
python Test_Scan.py      # recast scan over [1,50]^2 -> results/scan_<Pe>_<Da>_stage1.h5
python Plot_1pct_Boundary.py   # Fig. S.1
```

## Artifacts (Zenodo)

```
case1_adr_1d/data/data_canonical.h5
case1_adr_1d/checkpoints/best_fno1d.pt
case1_adr_1d/peda_scan_bandlimit_results.h5
case1_adr_1d/ablation_canonical_point/{data,checkpoints,results}/
```

Runtime: dataset ~1 min, training ~10 min on one GPU, full scan ~15 min.
