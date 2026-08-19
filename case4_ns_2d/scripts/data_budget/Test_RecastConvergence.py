#!/usr/bin/env python3
"""
Test_RecastConvergence.py — recast convergence + inference-cost diagnostic for
the redistributed-recast model at five test Reynolds numbers.

Left panel : relative L2 error vs recast iteration (median over 20 sources),
             with the 95%-plateau iteration (k95) marked.
Right panel: wall time at matched accuracy (the per-Re 95% plateau). The
             equation recast is timed as a BATCH of all 20 sources processed
             in parallel on the GPU (its natural deployment; a single fixed
             iteration structure batches trivially), reported as time per
             source = batch time / 20. The numerical solver (NumPy CPU and
             torch GPU) is timed per source, as its adaptive per-source
             control does not batch in the same way.

Output: results/recast_convergence.h5 + Fig_Test4_recast_convergence.png
"""
from __future__ import annotations
import os, time, h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from FNO2D import FNO2d
from Train_Canonical_PINN import make_kgrids
import VorticityNS_2D as ns

RE_LIST = [50, 150, 250, 350, 400]
RE_STAR = 250.0
N = 128
L_DOMAIN = 1.0
N_SRC = 20
SEED = 13
K_HARD = 21
MAX_ITERS = 80
EPS = 1e-12
AITKEN_INIT = 0.35
AITKEN_MIN = 0.02
AITKEN_MAX = 0.85
CKPT = "models/redistributed/redistributed_data.pt"
SOLVER_H5 = "../Test3_NS_PaperBaseline/results/test3_matched_accuracy_time.h5"

plt.rcParams.update({
    "font.size": 20, "axes.titlesize": 26, "axes.titleweight": "bold",
    "axes.labelsize": 23, "axes.labelweight": "bold",
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
    "lines.linewidth": 3.0, "savefig.dpi": 300,
})


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def batched_recast(model, S_b, refs, Re_target, stats, Kx, Ky, mask):
    """Batched recast over B sources. Returns:
       err_traj (MAX+1, B) relative L2 error per source per iterate,
       cum_time (MAX+1,)   cumulative batch wall time (s)."""
    s_min, s_max = stats["S_min"], stats["S_max"]
    o_min, o_max = stats["omega_min"], stats["omega_max"]
    B = S_b.shape[0]
    ref_n = refs.flatten(1).norm(dim=1) + EPS

    def apply_bl(f):
        Fh = torch.fft.rfft2(f, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(Fh * mask, s=f.shape[-2:], dim=(-2, -1), norm="backward")

    def lap(f):
        Fh = torch.fft.rfft2(f, dim=(-2, -1), norm="backward")
        return torch.fft.irfft2(-(Kx ** 2 + Ky ** 2) * Fh, s=f.shape[-2:], dim=(-2, -1), norm="backward")

    def predict(S_field):
        s_n = (2.0 * (S_field - s_min) / (s_max - s_min) - 1.0).unsqueeze(1).to(torch.float32)
        with torch.no_grad():
            o = model(s_n).squeeze(1).to(torch.float64)
        return 0.5 * (o + 1.0) * (o_max - o_min) + o_min

    def relerr(w):
        return ((w - refs).flatten(1).norm(dim=1) / ref_n).cpu().numpy()

    _sync(); t0 = time.perf_counter()
    omega = apply_bl(predict(S_b))
    _sync()
    errs = [relerr(omega)]; times = [time.perf_counter() - t0]
    omega_relax = torch.full((B,), AITKEN_INIT, dtype=torch.float64, device=S_b.device)
    r_prev = None
    for it in range(1, MAX_ITERS + 1):
        S_eff = apply_bl(S_b + (1.0 / Re_target - 1.0 / RE_STAR) * lap(omega))
        omega_hat = apply_bl(predict(S_eff))
        r = omega_hat - omega
        if r_prev is not None:
            dr = r - r_prev
            num = (r_prev * dr).sum(dim=(-2, -1))
            den = (dr * dr).sum(dim=(-2, -1)).clamp(min=EPS)
            cand = -omega_relax * num / den
            cand = torch.where(torch.isfinite(cand), cand, omega_relax)
            omega_relax = cand.clamp(AITKEN_MIN, AITKEN_MAX)
        omega = omega + omega_relax.view(B, 1, 1) * r
        r_prev = r
        _sync()
        errs.append(relerr(omega)); times.append(time.perf_counter() - t0)
    return np.stack(errs), np.array(times)   # (MAX+1, B), (MAX+1,)


def ns_solve(S_np, Re, kx, ky, ikx, iky, lap_symbol, inv_ksq, dealias):
    saved = ns.RE; ns.RE = float(Re)
    try:
        omega, _, _, _, _ = ns.solve_steady_vorticity(S_np, ikx, iky, lap_symbol, inv_ksq, dealias)
    finally:
        ns.RE = saved
    return omega


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RecastConvergence] device={device} ckpt={CKPT} N_SRC={N_SRC}")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = FNO2d(modes_x=ck.get("modes", 32), modes_y=ck.get("modes", 32),
                  width=ck.get("width", 64), in_channels=1, out_channels=1,
                  n_layers=ck.get("n_layers", 4)).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stats = {"S_min": ck["S_min"], "S_max": ck["S_max"],
             "omega_min": ck["omega_min"], "omega_max": ck["omega_max"]}

    kx_np, ky_np, ikx_np, iky_np, lap_sym_np, inv_ksq_np, dealias_np = ns.make_spectral_operators(N)
    Kx, Ky, _ = make_kgrids(N, L_DOMAIN, device)
    Kx = Kx.to(torch.float64); Ky = Ky.to(torch.float64)
    two_pi = 2.0 * np.pi
    k_rad = ((Kx / two_pi) ** 2 + (Ky / two_pi) ** 2).sqrt()
    mask = (k_rad <= float(K_HARD)).to(torch.complex128)

    rng = np.random.default_rng(SEED)
    sources = np.stack([ns.generate_source(N, kx_np, ky_np, dealias_np, rng)
                        for _ in range(N_SRC)], axis=0)
    S_b = torch.tensor(sources, dtype=torch.float64, device=device)

    # warmup (stabilize GPU timing)
    batched_recast(model, S_b, S_b, 250.0, stats, Kx, Ky, mask)

    traj = {}; k95 = {}; recast_ms = {}; Tplat = {}
    for Re in RE_LIST:
        refs_np = np.stack([ns_solve(sources[s], Re, kx_np, ky_np, ikx_np, iky_np,
                                     lap_sym_np, inv_ksq_np, dealias_np) for s in range(N_SRC)])
        refs = torch.tensor(refs_np, dtype=torch.float64, device=device)
        # median over a few batched timing repeats for a stable wall-time
        err_traj, t1 = batched_recast(model, S_b, refs, float(Re), stats, Kx, Ky, mask)
        _, t2 = batched_recast(model, S_b, refs, float(Re), stats, Kx, Ky, mask)
        cum_time = np.minimum(t1, t2)               # best-of-2, batch cumulative s
        # per-source plateau / k95 / amortized time
        k_src = []; rt_src = []; plat_src = []
        for s in range(N_SRC):
            e = err_traj[:, s]; E = e.min()
            k = int(np.argmax(e <= 1.05 * E))
            k_src.append(k); plat_src.append(E)
            rt_src.append(cum_time[k] / N_SRC)       # batch time / 20 = per source
        traj[Re] = np.median(err_traj, axis=1) * 100.0
        k95[Re] = int(np.median(k_src))
        recast_ms[Re] = float(np.median(rt_src) * 1000.0)
        Tplat[Re] = float(np.median(plat_src))
        print(f"  Re={Re}: plateau={Tplat[Re]*100:.2f}%  k95={k95[Re]}  "
              f"batched recast/src={recast_ms[Re]:.2f}ms")

    # ---- solver Pareto (per source) at the same target accuracy ----
    cpu_ms = {}; gpu_ms = {}
    def time_at_T(tt, ee, T):
        idx = np.where(np.asarray(ee) <= T)[0]
        return float(tt[idx[0]]) if len(idx) else float(tt[-1])
    if os.path.exists(SOLVER_H5):
        with h5py.File(SOLVER_H5, "r") as f:
            nsrc = int(f.attrs["n_sources"])
            for Re in RE_LIST:
                cc = []; gg = []
                for si in range(nsrc):
                    g = f[f"Re{int(Re)}_src{si:02d}"]
                    cc.append(time_at_T(g["solver_cpu_time_s"][:], g["solver_cpu_err"][:], Tplat[Re]))
                    gg.append(time_at_T(g["solver_gpu_time_s"][:], g["solver_gpu_err"][:], Tplat[Re]))
                cpu_ms[Re] = float(np.median(cc) * 1000.0)
                gpu_ms[Re] = float(np.median(gg) * 1000.0)
    else:
        print(f"  WARNING: {SOLVER_H5} missing; timing panel omits solver")

    os.makedirs("results", exist_ok=True)
    with h5py.File("results/recast_convergence.h5", "w") as f:
        for Re in RE_LIST:
            f.create_dataset(f"err_Re{Re}", data=traj[Re])
        f.attrs["Re_list"] = np.array(RE_LIST)
        f.attrs["k95"] = np.array([k95[Re] for Re in RE_LIST])
        f.attrs["recast_ms_per_src_batched"] = np.array([recast_ms[Re] for Re in RE_LIST])
        f.attrs["plateau_pct"] = np.array([Tplat[Re] * 100 for Re in RE_LIST])
        if cpu_ms:
            f.attrs["cpu_ms"] = np.array([cpu_ms[Re] for Re in RE_LIST])
            f.attrs["gpu_ms"] = np.array([gpu_ms[Re] for Re in RE_LIST])

    # ---- combined figure ----
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(RE_LIST)))
    fig, (axc, axt) = plt.subplots(1, 2, figsize=(20, 7.5))

    for c, Re in zip(colors, RE_LIST):
        y = traj[Re]
        lab = f"Re = {Re}" + ("  (canonical)" if Re == int(RE_STAR) else "")
        axc.plot(np.arange(len(y)), y, color=c, marker="o", markersize=4, markevery=10, label=lab)
        axc.plot(k95[Re], y[k95[Re]], marker="*", color=c, markersize=22,
                 markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    axc.plot([], [], marker="*", color="gray", markersize=18, markeredgecolor="black",
             linestyle="none", label="95\\% plateau")
    axc.set_xlabel("recast iteration"); axc.set_ylabel("Rel. $L^2$ error (%)")
    axc.set_yscale("log"); axc.set_xlim(0, 40)
    axc.grid(True, which="both", alpha=0.3)
    axc.set_title("Convergence trajectory", fontsize=26, fontweight="bold")
    axc.legend(loc="upper right", framealpha=0.95)

    Re_arr = np.array(RE_LIST, dtype=float)
    axt.plot(Re_arr, [recast_ms[Re] for Re in RE_LIST], color="#1f77b4",
             marker="s", linewidth=3.0, markersize=11, label="Equation recast")
    if cpu_ms:
        axt.plot(Re_arr, [cpu_ms[Re] for Re in RE_LIST], color="tab:red",
                 marker="o", linestyle="--", linewidth=3.0, markersize=10, label="NumPy CPU")
        axt.plot(Re_arr, [gpu_ms[Re] for Re in RE_LIST], color="tab:purple",
                 marker="o", linestyle="--", linewidth=3.0, markersize=10, label="torch GPU")
    axt.axvline(RE_STAR, color="red", linestyle="--", linewidth=2.0, alpha=0.85)
    axt.set_xlabel("Re"); axt.set_ylabel("wall time per source (ms)")
    axt.set_yscale("linear"); axt.grid(True, which="both", alpha=0.3)
    axt.set_title("Inference cost at matched accuracy", fontsize=26, fontweight="bold")
    axt.legend(loc="best", framealpha=0.95)

    fig.tight_layout()
    out = "results/Fig_Test4_recast_convergence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
