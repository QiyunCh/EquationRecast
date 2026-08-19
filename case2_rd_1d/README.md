# Case 2 — 1D reaction–diffusion (Fig. 3a, SI S2)

$$-u''(x) + k\,u(x) = S(x), \qquad x\in(0,10),\quad u(0)=u(10)=0$$

Canonical coefficient $k^*=0.785$; the recast is

$$\mathcal{O}^*[u] = S - (k-k^*)\,u .$$

This case tests **heterogeneous-data enrichment**: off-canonical training pairs
$(S,u)$ generated at $k_{\mathrm{new}}=0.05$ are recast into the canonical
effective-source representation,

$$S_{\mathrm{eff}} = S - (k_{\mathrm{new}}-k^*)\,u,$$

used as the network input while the high-fidelity solution $u$ remains the
supervision target. Both models get the same total budget of 1000 samples:

* **Model 1** — 1000 canonical samples at $k^*$.
* **Model 2** — 500 canonical + 500 recast samples from $k_{\mathrm{new}}=0.05$.

## Settings (SI S2)

| | |
|---|---|
| Grid | $N=201$, second-order finite differences, sparse direct solve |
| Sources | GRF, squared-exponential covariance, $\ell=0.5$, $\sigma=1$, rescaled to $[-1,1]$ |
| Model | FNO1d, 3 Fourier layers, 64 modes, width 64, source + coordinate channel |
| Split | 90% train / 10% validation; Adam, MSE |
| Evaluation | 20 test sources, 27 values of $k\in[0.01,10]$; Aitken $\Delta^2$ |

## Reproduce

```bash
cd scripts
python ReactionDiffusion.py   # datasets at k* = 0.785 and k_new = 0.05
python data_merge.py          # recast the k_new pairs -> merged canonical dataset
python Train_Model1.py        # canonical-only model
python Train_Model2.py        # canonical + recast heterogeneous model
python Test_Compare.py        # k-scan for both models -> Fig. 3a
```

`data_merge.py` inherits the solution normalization range from the canonical
dataset, so both models see an identical output scale.

## Artifacts (Zenodo)

```
case2_rd_1d/data/{data_beta_0.785.h5, data_beta_0.050.h5, data_merged_beta_0.785_0.050.h5}
case2_rd_1d/checkpoints/{best_fno1d_model1.pt, best_fno1d_model2.pt}
case2_rd_1d/results/recast_rd_compare_2models.h5
```

Runtime: a few minutes end-to-end on one GPU; CPU is workable.
