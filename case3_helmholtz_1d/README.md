# Case 3 — 1D Helmholtz (Fig. 3b, SI S3)

$$u''(x) + k^2 u(x) = -S(x), \qquad x\in(0,10),\quad u(0)=u(10)=0$$

Dirichlet resonances at $k_n = n\pi/10$, so $k_2=0.628$, $k_3=0.942$,
$k_4=1.257$. The canonical wavenumber sits between the second and third
resonance, $k^*=0.785=\tfrac12(k_2+k_3)$, and the recast is

$$\mathcal{O}^*[u] = -S - \bigl(k^2-(k^*)^2\bigr)u .$$

This case is the **failure diagnostic**: as a resonance is approached the recast
fixed-point map loses contractivity, the iteration count rises, and beyond
$k_3$ the iteration hits the cap while the error grows by orders of magnitude —
a structural limit that additional training data cannot remove.

Two models at equal budget (1000 samples), as in case 2:

* **Model 1** — canonical only, $k^*=0.785$.
* **Model 2** — 500 canonical + 500 recast pairs from $k_{\mathrm{new}}=1.099=\tfrac12(k_3+k_4)$.

With the positive-source convention used for the network input,

$$S_{\mathrm{eff}} = S + \bigl(k_{\mathrm{new}}^2-(k^*)^2\bigr)u,
\qquad \mathcal{O}^*[u] = -S_{\mathrm{eff}} .$$

## Settings (SI S3)

| | |
|---|---|
| Grid | $N=201$, second-order finite differences, sparse direct solve |
| Sources | GRF, squared-exponential covariance, $\ell=0.5$, $\sigma=1$, rescaled to $[-1,1]$ |
| Model | FNO1d, 3 Fourier layers, 64 modes, width 64, source + coordinate channel |
| Evaluation | 20 test sources, scan $k\in[0.630,1.200]$; Aitken $\Delta^2$; iteration count reported alongside the error |

## Reproduce

```bash
cd scripts
python Helmholtz.py       # datasets at k* = 0.785 and k_new = 1.099
python data_merge.py      # recast the k_new pairs -> merged canonical dataset
python Train_Model1.py
python Train_Model2.py
python Test_Compare.py    # k-scan, error + iteration count -> Fig. 3b
```

`Test_Compare.py` reuses its saved HDF5 if present, so re-plotting is instant.

## Artifacts (Zenodo)

```
case3_helmholtz_1d/data/{data_0.785.h5, data_1.099.h5, data_merged_0.785_1.099.h5}
case3_helmholtz_1d/checkpoints/{best_fno1d_model1.pt, best_fno1d_model2.pt}
case3_helmholtz_1d/results/recast_helmholtz_compare_2models.h5
```
