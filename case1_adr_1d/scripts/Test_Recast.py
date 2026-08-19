"""
Test_2D_GPU_Aitken_from_original.py

GPU-batched version of Test_PeDaScan_Recast_Bandlimit.py.

Changes relative to the original CPU script:
  1. Keep the original recast / bandlimit / rhs-scaling algorithmic structure.
  2. Batch all N_SOURCES together on GPU for each (Pe, Da) grid point.
  3. Replace fixed damping by a standard Aitken Delta^2 dynamic damping factor:

        omega_k = - omega_{k-1} * <r_{k-1}, r_k - r_{k-1}> / ||r_k - r_{k-1}||^2,

     where r_k = u_hat_k - u_k is the fixed-point residual. The first step uses
     DAMP_INIT. The factor is clamped to [OMEGA_MIN, OMEGA_MAX] for robustness.
  4. Add transfer-free timing for the recast solver and the exact spectral baseline.
  5. Add statistics for ||S_eff||_2 / ||S||_2 at the final predicted solution, where

        S_eff = S - (Pe - Pe*) u_x - (Da - Da*) u.

PDE, periodic on [0,1):
    -u_xx + Pe*u_x + Da*u = S(x)

Canonical operator:
    Pe* = 4, Da* = 2

Recast:
    L* u = S - (Pe - Pe*) u_x - (Da - Da*) u.

Outputs:
  - HDF5: peda_scan_bandlimit_results_gpu_aitken.h5
  - PNGs:
      peda_scan_bandlimit_mean_relL2_contour_gpu_aitken.png
      peda_scan_bandlimit_Seff_ratio_contour_gpu_aitken.png
      peda_scan_bandlimit_time_recast_contour_gpu_aitken.png
      peda_scan_bandlimit_slowdown_contour_gpu_aitken.png
"""

import os
import glob
import time
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.tri import Triangulation, LinearTriInterpolator

from FNO1D import FNO1d


# -----------------------------
# User settings
# -----------------------------
SEED = 13
N_SOURCES = 20

# Canonical parameters
PE_STAR = 4.0
DA_STAR = 2.0

# Scan ranges
PE_MIN, PE_MAX, PE_STEP = 1.0, 20.0, 1.0
DA_MIN, DA_MAX, DA_STEP = 1.0, 20.0, 1.0
PE_LIST = np.arange(PE_MIN, PE_MAX + 0.5 * PE_STEP, PE_STEP, dtype=np.float64)
DA_LIST = np.arange(DA_MIN, DA_MAX + 0.5 * DA_STEP, DA_STEP, dtype=np.float64)

# Aitken Delta^2 damping settings
DAMP_INIT = 0.1
OMEGA_MIN = 0.05
OMEGA_MAX = 1.5

# Iteration controls
TOL = 1e-5
MAX_ITERS = 150
EPS = 1e-12

# Mesh (must match training/data)
L = 1.0
N = 201
x = np.linspace(0.0, L, N, endpoint=False)

# GRF
GRF_ELL = 0.08
GRF_SIGMA = 1.0

# Bandlimit rule:
# keep rFFT modes where |S_hat[m]| >= AMP_FRAC * max(|S_hat|)
AMP_FRAC = 1e-3

# Canonical dataset for u_min/u_max (training normalization)
CANON_DATA_H5 = "data_canonical.h5"

# Model checkpoint
CKPT_CANDIDATES = [os.path.join("Results", "best_fno1d.pt"), "best_fno1d.pt"]

# Outputs
OUT_H5 = "peda_scan_bandlimit_results.h5"
OUT_PNG_RELL2 = "peda_scan_bandlimit_mean_relL2_contour_gpu_aitken.png"
OUT_PNG_SEFF = "peda_scan_bandlimit_Seff_ratio_contour_gpu_aitken.png"
OUT_PNG_TIME = "peda_scan_bandlimit_time_recast_contour_gpu_aitken.png"
OUT_PNG_SLOWDOWN = "peda_scan_bandlimit_slowdown_contour_gpu_aitken.png"

# Plot interpolation resolution (only inside [1,10])
PLOT_RES = 0.1

# Timing: median over repeated transfer-free runs
WARMUP_RUNS = 1
TIMING_RUNS = 3


# -----------------------------
# Small utilities
# -----------------------------
def safe_nanmean(a: np.ndarray) -> float:
    a = np.asarray(a)
    b = a[np.isfinite(a)]
    if b.size == 0:
        return np.nan
    return float(np.mean(b))


def safe_nanstd(a: np.ndarray) -> float:
    a = np.asarray(a)
    b = a[np.isfinite(a)]
    if b.size == 0:
        return np.nan
    return float(np.std(b))


def rescale_to_minus1_1(S: np.ndarray) -> np.ndarray:
    smin = float(S.min())
    smax = float(S.max())
    if np.isclose(smin, smax):
        return np.zeros_like(S)
    return 2.0 * (S - smin) / (smax - smin) - 1.0


def find_checkpoint() -> str:
    for p in CKPT_CANDIDATES:
        if os.path.isfile(p):
            return p
    cand = glob.glob(os.path.join("Results", "best_fno1d*.pt")) + glob.glob("best_fno1d*.pt")
    if cand:
        return cand[0]
    raise FileNotFoundError("Cannot find FNO checkpoint (best_fno1d.pt).")


def infer_in_channels_from_state_dict(state: dict) -> int:
    if "fc0.weight" not in state:
        raise KeyError("State dict missing 'fc0.weight'.")
    return int(state["fc0.weight"].shape[1])


def cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# -----------------------------
# Periodic GRF sampler (SE) via Fourier spectrum, CPU setup only
# -----------------------------
def sample_periodic_grf_se(rng: np.random.Generator, N: int, L: float, ell: float, sigma: float) -> np.ndarray:
    n = np.arange(0, N // 2 + 1, dtype=np.float64)
    c = (sigma ** 2) * np.sqrt(2.0 * np.pi) * ell * np.exp(
        -(2.0 * (np.pi ** 2) * (ell ** 2) * (n ** 2)) / (L ** 2)
    )

    coeff = np.zeros(n.shape[0], dtype=np.complex128)
    coeff[0] = rng.normal(0.0, 1.0) * np.sqrt(c[0])

    if coeff.shape[0] > 2:
        real = rng.normal(0.0, 1.0, size=coeff.shape[0] - 2)
        imag = rng.normal(0.0, 1.0, size=coeff.shape[0] - 2)
        coeff[1:-1] = (real + 1j * imag) * np.sqrt(0.5 * c[1:-1])

    if N % 2 == 0:
        coeff[-1] = rng.normal(0.0, 1.0) * np.sqrt(c[-1])

    field = np.fft.irfft(coeff, n=N).astype(np.float64)
    field -= np.mean(field)
    return field


# -----------------------------
# CPU mask setup
# -----------------------------
def bandlimit_mask_from_source(S: np.ndarray, amp_frac: float) -> np.ndarray:
    Sh = np.fft.rfft(S)
    amp = np.abs(Sh)
    amax = float(np.max(amp))
    if (not np.isfinite(amax)) or amax <= 0.0:
        mask = np.zeros_like(amp, dtype=bool)
        mask[0] = True
        return mask
    thr = amp_frac * amax
    mask = amp >= thr
    mask[0] = True
    return mask


# -----------------------------
# Torch spectral utilities, batched on device
# -----------------------------
def rfft_wavenumbers_torch(N: int, L: float, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    freq = torch.fft.rfftfreq(N, d=L / N, dtype=dtype, device=device)
    return 2.0 * float(np.pi) * freq


def spectral_derivative_torch(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # u: (B, N), real; k: (N//2+1,), real
    uh = torch.fft.rfft(u, dim=-1)
    ux_hat = (1j * k) * uh
    return torch.fft.irfft(ux_hat, n=u.shape[-1], dim=-1)


def spectral_solve_adr_torch(S: torch.Tensor, Pe: float, Da: float, k: torch.Tensor) -> torch.Tensor:
    # Solve (-dxx + Pe*dx + Da)u = S in Fourier space.
    Sh = torch.fft.rfft(S, dim=-1)
    denom = (k ** 2) + 1j * float(Pe) * k + float(Da)
    uh = Sh / denom
    return torch.fft.irfft(uh, n=S.shape[-1], dim=-1)


def apply_bandlimit_torch(u: torch.Tensor, mask_c: torch.Tensor) -> torch.Tensor:
    # u: (B, N), real; mask_c: (B, N//2+1), complex/real with 1 keep and 0 drop
    uh = torch.fft.rfft(u, dim=-1)
    uh = uh * mask_c
    return torch.fft.irfft(uh, n=u.shape[-1], dim=-1)


# -----------------------------
# FNO apply, batched and transfer-free
# -----------------------------
def fno_apply_G_star_batched(
    model: torch.nn.Module,
    rhs: torch.Tensor,
    u_min: float,
    u_max: float,
) -> torch.Tensor:
    """
    rhs: (B, N), on device. Model expects (B, 1, N), float32.
    Returns denormalized u: (B, N), float64, on device.
    """
    xb = rhs.unsqueeze(1).to(torch.float32)
    with torch.no_grad():
        yb = model(xb)
    u_norm = yb.squeeze(1).to(torch.float64)
    u = 0.5 * (u_norm + 1.0) * (float(u_max) - float(u_min)) + float(u_min)
    return u


# -----------------------------
# Full batched recast + bandlimit fixed-point with Aitken Delta^2 damping
# -----------------------------
def fixed_point_peda_recast_bandlimit_batched(
    model: torch.nn.Module,
    S_batch: torch.Tensor,
    Pe: float,
    Da: float,
    k: torch.Tensor,
    mask_c: torch.Tensor,
    u_min: float,
    u_max: float,
    tol: float,
    max_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched equivalent of the original per-source solver.

    The recast equation, strict bandlimit, per-iteration c=max|rhs| scaling,
    FNO application, and undo scaling are preserved. The only intended algorithmic
    change is replacing fixed damping by Aitken Delta^2 dynamic damping.

    Already-converged sources are frozen so the batched execution remains close
    to the original per-source early stopping behavior.

    Returns:
        u_final       (B, N), float64
        iters_first   (B,), long, first convergence iteration or max_iters
        relchg_last   (B,), float64
        c_mean        (B,), float64
        omega_mean    (B,), float64
    """
    B, _ = S_batch.shape
    device = S_batch.device

    dPe = float(Pe - PE_STAR)
    dDa = float(Da - DA_STAR)

    # Initial guess: canonical inverse on S, then bandlimit.
    u0 = fno_apply_G_star_batched(model, S_batch, u_min, u_max)
    finite0 = torch.isfinite(u0).all(dim=-1)
    u = torch.where(finite0[:, None], u0, torch.zeros_like(u0))
    u = apply_bandlimit_torch(u, mask_c)

    omega = torch.full((B,), DAMP_INIT, dtype=torch.float64, device=device)
    omega_sum = torch.zeros(B, dtype=torch.float64, device=device)
    c_sum = torch.zeros(B, dtype=torch.float64, device=device)
    n_steps = torch.zeros(B, dtype=torch.long, device=device)

    iters_first = torch.full((B,), max_iters, dtype=torch.long, device=device)
    relchg_last = torch.full((B,), float("nan"), dtype=torch.float64, device=device)
    converged = torch.zeros(B, dtype=torch.bool, device=device)
    failed = ~finite0

    r_prev = None
    u_new = u.clone()

    for it in range(1, max_iters + 1):
        active = (~converged) & (~failed)
        if not bool(active.any().item()):
            break

        # Enforce strict bandlimit before derivative, as in the original script.
        u_bl = apply_bandlimit_torch(u, mask_c)

        ux = spectral_derivative_torch(u_bl, k)
        rhs = S_batch - dPe * ux - dDa * u_bl

        c = rhs.abs().amax(dim=-1).clamp(min=1e-12)
        rhs_in = rhs / c[:, None]

        u_hat_scaled = fno_apply_G_star_batched(model, rhs_in, u_min, u_max)
        u_hat_scaled = apply_bandlimit_torch(u_hat_scaled, mask_c)
        u_hat = c[:, None] * u_hat_scaled

        finite_hat = torch.isfinite(u_hat).all(dim=-1)
        newly_failed = active & (~finite_hat)
        failed = failed | newly_failed
        active = active & finite_hat
        if not bool(active.any().item()):
            break

        r = u_hat - u_bl

        # Aitken Delta^2 update of the damping factor, starting from the second residual.
        if r_prev is not None:
            dr = r - r_prev
            num = (r_prev * dr).sum(dim=-1)
            den = (dr * dr).sum(dim=-1).clamp(min=EPS)
            omega_candidate = -omega * num / den
            valid = torch.isfinite(omega_candidate)
            omega = torch.where(valid & active, omega_candidate, omega)
            omega = torch.clamp(omega, OMEGA_MIN, OMEGA_MAX)

        omega_sum = torch.where(active, omega_sum + omega, omega_sum)
        c_sum = torch.where(active, c_sum + c, c_sum)
        n_steps = torch.where(active, n_steps + 1, n_steps)

        candidate = u_bl + omega[:, None] * r
        candidate = apply_bandlimit_torch(candidate, mask_c)

        diff_norm = (candidate - u_bl).norm(dim=-1)
        u_norm = u_bl.norm(dim=-1) + EPS
        relchg = diff_norm / u_norm
        relchg_last = torch.where(active, relchg, relchg_last)

        # Freeze converged/inactive sources; update only active sources.
        u_new = torch.where(active[:, None], candidate, u)

        newly_converged = active & (relchg < tol)
        iters_first = torch.where(newly_converged, torch.full_like(iters_first, it), iters_first)
        converged = converged | newly_converged

        # Store current residual for Aitken. Inactive rows are harmless because update is masked.
        r_prev = r
        u = u_new

    c_mean = c_sum / n_steps.clamp(min=1).to(torch.float64)
    omega_mean = omega_sum / n_steps.clamp(min=1).to(torch.float64)

    # Match the original behavior for failed rows: output NaNs.
    if bool(failed.any().item()):
        u = torch.where(failed[:, None], torch.full_like(u, float("nan")), u)
        relchg_last = torch.where(failed, torch.full_like(relchg_last, float("nan")), relchg_last)
        c_mean = torch.where(failed, torch.full_like(c_mean, float("nan")), c_mean)
        omega_mean = torch.where(failed, torch.full_like(omega_mean, float("nan")), omega_mean)

    return u, iters_first, relchg_last, c_mean, omega_mean


# -----------------------------
# Timing helpers: transfer-free, warmup + median
# -----------------------------
def time_recast_solver(
    model: torch.nn.Module,
    S_batch: torch.Tensor,
    mask_c: torch.Tensor,
    Pe: float,
    Da: float,
    k: torch.Tensor,
    u_min: float,
    u_max: float,
    tol: float,
    max_iters: int,
    warmup: int = WARMUP_RUNS,
    runs: int = TIMING_RUNS,
):
    last = None
    for _ in range(warmup):
        last = fixed_point_peda_recast_bandlimit_batched(
            model, S_batch, Pe, Da, k, mask_c, u_min, u_max, tol, max_iters
        )
        cuda_sync_if_available()

    times = []
    for _ in range(runs):
        cuda_sync_if_available()
        t0 = time.perf_counter()
        last = fixed_point_peda_recast_bandlimit_batched(
            model, S_batch, Pe, Da, k, mask_c, u_min, u_max, tol, max_iters
        )
        cuda_sync_if_available()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), last


def time_spectral_baseline(
    S_batch: torch.Tensor,
    Pe: float,
    Da: float,
    k: torch.Tensor,
    warmup: int = WARMUP_RUNS,
    runs: int = TIMING_RUNS,
):
    last = None
    for _ in range(warmup):
        last = spectral_solve_adr_torch(S_batch, Pe, Da, k)
        cuda_sync_if_available()

    times = []
    for _ in range(runs):
        cuda_sync_if_available()
        t0 = time.perf_counter()
        last = spectral_solve_adr_torch(S_batch, Pe, Da, k)
        cuda_sync_if_available()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times)), last


# -----------------------------
# Plotting: smooth contour via linear interpolation, no extrapolation
# -----------------------------
def plot_smooth_contour(
    Pe_grid: np.ndarray,
    Da_grid: np.ndarray,
    Z: np.ndarray,
    out_png: str,
    *,
    label: str,
    title: str,
) -> None:
    pe_pts = Pe_grid.ravel()
    da_pts = Da_grid.ravel()
    z_pts = Z.ravel()

    m = np.isfinite(z_pts)
    pe_pts = pe_pts[m]
    da_pts = da_pts[m]
    z_pts = z_pts[m]

    tri = Triangulation(pe_pts, da_pts)
    interp = LinearTriInterpolator(tri, z_pts)

    pe_f = np.arange(PE_MIN, PE_MAX + 0.5 * PLOT_RES, PLOT_RES, dtype=np.float64)
    da_f = np.arange(DA_MIN, DA_MAX + 0.5 * PLOT_RES, PLOT_RES, dtype=np.float64)
    PE_F, DA_F = np.meshgrid(pe_f, da_f)
    ZF = interp(PE_F, DA_F)

    plt.figure(figsize=(9.5, 7.5))
    cf = plt.contourf(PE_F, DA_F, ZF, levels=30)
    plt.colorbar(cf, label=label)
    plt.plot([PE_STAR], [DA_STAR], "ko", markersize=7, label="Canonical (Pe*=4, Da*=2)")
    plt.xlabel("Pe")
    plt.ylabel("Da")
    plt.title(title)
    plt.xlim(PE_MIN - 0.5, PE_MAX + 0.5)
    plt.ylim(DA_MIN - 0.5, DA_MAX + 0.5)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Load training u_min/u_max.
    if not os.path.isfile(CANON_DATA_H5):
        raise FileNotFoundError(f"Cannot find canonical data file: {CANON_DATA_H5}")
    with h5py.File(CANON_DATA_H5, "r") as f:
        u_min = float(f.attrs["solution_min"])
        u_max = float(f.attrs["solution_max"])

    print("[Info] Canonical data:", CANON_DATA_H5)
    print("[Info] u_min =", u_min)
    print("[Info] u_max =", u_max)

    # Load model.
    ckpt = find_checkpoint()
    state = torch.load(ckpt, map_location="cpu")
    in_channels = infer_in_channels_from_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Info] Loaded checkpoint:", ckpt)
    print("[Info] Inferred model input channels =", in_channels)
    print("[Info] Using device:", device)
    if device.type == "cuda":
        print("[Info] GPU:", torch.cuda.get_device_name(0))

    if in_channels != 1:
        raise ValueError(f"This script expects in_channels=1, got {in_channels}")

    model = FNO1d(modes=64, width=64, in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(state)
    model.eval()

    # Device wavenumbers.
    k_torch = rfft_wavenumbers_torch(N=N, L=L, device=device, dtype=torch.float64)

    # Sample sources and masks once on CPU, then upload to device once.
    rng = np.random.default_rng(SEED)
    sources = np.zeros((N_SOURCES, N), dtype=np.float64)
    masks = np.zeros((N_SOURCES, N // 2 + 1), dtype=np.uint8)
    kept_modes = np.zeros(N_SOURCES, dtype=np.int64)

    for i in range(N_SOURCES):
        S_raw = sample_periodic_grf_se(rng=rng, N=N, L=L, ell=GRF_ELL, sigma=GRF_SIGMA)
        S = rescale_to_minus1_1(S_raw)
        sources[i] = S

        m = bandlimit_mask_from_source(S, AMP_FRAC)
        masks[i, :] = m.astype(np.uint8)
        kept_modes[i] = int(np.sum(m))

    print(f"[Info] Sampled {N_SOURCES} sources: GRF ell={GRF_ELL}, each rescaled to [-1,1].")
    print(
        f"[Info] Bandlimit AMP_FRAC={AMP_FRAC} -> kept modes: "
        f"min={int(kept_modes.min())}, max={int(kept_modes.max())}, mean={kept_modes.mean():.2f}"
    )
    print(
        f"[Info] Scan Pe=[{PE_MIN},{PE_MAX}] step {PE_STEP} ({PE_LIST.size} points), "
        f"Da=[{DA_MIN},{DA_MAX}] step {DA_STEP} ({DA_LIST.size} points)"
    )
    print(
        f"[Info] Aitken Delta^2 damping: omega0={DAMP_INIT}, "
        f"clamp=[{OMEGA_MIN},{OMEGA_MAX}], MAX_ITERS={MAX_ITERS}, TOL={TOL}"
    )
    print(f"[Info] Timing: warmup={WARMUP_RUNS}, runs={TIMING_RUNS}, median, transfer-free")

    S_batch = torch.from_numpy(sources).to(device=device, dtype=torch.float64)
    mask_c = torch.from_numpy(masks.astype(np.float64)).to(device=device, dtype=torch.float64).to(torch.complex128)
    S_norm_per = S_batch.norm(dim=-1)

    # Allocate grids: shape (nDa, nPe).
    nPe = PE_LIST.size
    nDa = DA_LIST.size

    mean_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    std_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    nan_count_grid = np.zeros((nDa, nPe), dtype=np.int64)
    nonconv_count_grid = np.zeros((nDa, nPe), dtype=np.int64)
    mean_iters_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    mean_relchg_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    mean_c_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    mean_omega_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)

    mean_seff_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    std_seff_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)

    t_recast_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)
    t_spectral_grid = np.full((nDa, nPe), np.nan, dtype=np.float64)

    all_errs = np.full((nDa, nPe, N_SOURCES), np.nan, dtype=np.float64)
    all_seff = np.full((nDa, nPe, N_SOURCES), np.nan, dtype=np.float64)
    all_iters = np.full((nDa, nPe, N_SOURCES), MAX_ITERS, dtype=np.int64)
    all_relchg = np.full((nDa, nPe, N_SOURCES), np.nan, dtype=np.float64)
    all_omega = np.full((nDa, nPe, N_SOURCES), np.nan, dtype=np.float64)

    grid_total = nDa * nPe
    grid_done = 0
    scan_t0 = time.perf_counter()

    for i_da, Da in enumerate(DA_LIST):
        for i_pe, Pe in enumerate(PE_LIST):
            Pe_f = float(Pe)
            Da_f = float(Da)
            dPe = Pe_f - PE_STAR
            dDa = Da_f - DA_STAR

            t_rec, (u_pred, iters_first, relchg_last, c_mean, omega_mean) = time_recast_solver(
                model=model,
                S_batch=S_batch,
                mask_c=mask_c,
                Pe=Pe_f,
                Da=Da_f,
                k=k_torch,
                u_min=u_min,
                u_max=u_max,
                tol=TOL,
                max_iters=MAX_ITERS,
            )

            t_spec, u_true = time_spectral_baseline(S_batch, Pe_f, Da_f, k_torch)

            # Accuracy: relative L2 vs exact spectral truth, per source.
            finite_pred = torch.isfinite(u_pred).all(dim=-1)
            finite_true = torch.isfinite(u_true).all(dim=-1)
            finite = finite_pred & finite_true
            errs_t = torch.full((N_SOURCES,), float("nan"), dtype=torch.float64, device=device)
            errs_t[finite] = (u_pred[finite] - u_true[finite]).norm(dim=-1) / (u_true[finite].norm(dim=-1) + EPS)
            errs = errs_t.detach().cpu().numpy()

            # Effective source ratio at final u_pred.
            seff_t = torch.full((N_SOURCES,), float("nan"), dtype=torch.float64, device=device)
            if bool(finite_pred.any().item()):
                ux_pred = spectral_derivative_torch(u_pred, k_torch)
                S_eff = S_batch - dPe * ux_pred - dDa * u_pred
                seff_t[finite_pred] = S_eff[finite_pred].norm(dim=-1) / (S_norm_per[finite_pred] + EPS)
            seff = seff_t.detach().cpu().numpy()

            iters_np = iters_first.detach().cpu().numpy()
            relchg_np = relchg_last.detach().cpu().numpy()
            c_np = c_mean.detach().cpu().numpy()
            omega_np = omega_mean.detach().cpu().numpy()

            all_errs[i_da, i_pe, :] = errs
            all_seff[i_da, i_pe, :] = seff
            all_iters[i_da, i_pe, :] = iters_np
            all_relchg[i_da, i_pe, :] = relchg_np
            all_omega[i_da, i_pe, :] = omega_np

            nan_count_grid[i_da, i_pe] = int(np.sum(~np.isfinite(errs)))
            nonconv_count_grid[i_da, i_pe] = int(np.sum(iters_np >= MAX_ITERS))
            mean_grid[i_da, i_pe] = safe_nanmean(errs)
            std_grid[i_da, i_pe] = safe_nanstd(errs)
            mean_iters_grid[i_da, i_pe] = safe_nanmean(iters_np)
            mean_relchg_grid[i_da, i_pe] = safe_nanmean(relchg_np)
            mean_c_grid[i_da, i_pe] = safe_nanmean(c_np)
            mean_omega_grid[i_da, i_pe] = safe_nanmean(omega_np)
            mean_seff_grid[i_da, i_pe] = safe_nanmean(seff)
            std_seff_grid[i_da, i_pe] = safe_nanstd(seff)
            t_recast_grid[i_da, i_pe] = t_rec
            t_spectral_grid[i_da, i_pe] = t_spec

            grid_done += 1
            elapsed = time.perf_counter() - scan_t0
            est_total = elapsed * grid_total / max(grid_done, 1)
            eta = est_total - elapsed

            print(
                f"[{grid_done:3d}/{grid_total}] Pe={Pe_f:4.1f}, Da={Da_f:4.1f} | "
                f"relL2={mean_grid[i_da, i_pe]:.3e}, std={std_grid[i_da, i_pe]:.3e}, "
                f"Seff/S={mean_seff_grid[i_da, i_pe]:.3e}, "
                f"iters={mean_iters_grid[i_da, i_pe]:5.1f}, nonconv={nonconv_count_grid[i_da, i_pe]}/{N_SOURCES}, "
                f"omega={mean_omega_grid[i_da, i_pe]:.3f}, "
                f"t_rec={1e3 * t_rec:8.2f} ms, t_spec={1e6 * t_spec:8.2f} us, "
                f"slowdown={t_rec / max(t_spec, 1e-12):.1f}x | ETA {eta:6.0f}s"
            )

    print(f"[Done] Scan finished in {time.perf_counter() - scan_t0:.1f}s")

    # Save results.
    PE_M, DA_M = np.meshgrid(PE_LIST, DA_LIST)

    with h5py.File(OUT_H5, "w") as f:
        # Metadata.
        f.attrs["seed"] = int(SEED)
        f.attrs["N_sources"] = int(N_SOURCES)
        f.attrs["L"] = float(L)
        f.attrs["N"] = int(N)
        f.attrs["Pe_star"] = float(PE_STAR)
        f.attrs["Da_star"] = float(DA_STAR)

        f.attrs["Pe_min"] = float(PE_MIN)
        f.attrs["Pe_max"] = float(PE_MAX)
        f.attrs["Pe_step"] = float(PE_STEP)
        f.attrs["Da_min"] = float(DA_MIN)
        f.attrs["Da_max"] = float(DA_MAX)
        f.attrs["Da_step"] = float(DA_STEP)

        f.attrs["damping_method"] = "Aitken Delta^2 dynamic damping on fixed-point residual"
        f.attrs["damp_init"] = float(DAMP_INIT)
        f.attrs["omega_min"] = float(OMEGA_MIN)
        f.attrs["omega_max"] = float(OMEGA_MAX)
        f.attrs["tol"] = float(TOL)
        f.attrs["max_iters"] = int(MAX_ITERS)

        f.attrs["GRF_ell"] = float(GRF_ELL)
        f.attrs["GRF_sigma"] = float(GRF_SIGMA)
        f.attrs["AMP_FRAC"] = float(AMP_FRAC)

        f.attrs["canonical_data_h5"] = CANON_DATA_H5
        f.attrs["checkpoint"] = ckpt
        f.attrs["model_in_channels"] = int(in_channels)
        f.attrs["u_min"] = float(u_min)
        f.attrs["u_max"] = float(u_max)
        f.attrs["device"] = str(device)
        f.attrs["timing_warmup"] = int(WARMUP_RUNS)
        f.attrs["timing_runs"] = int(TIMING_RUNS)
        f.attrs["timing_aggregator"] = "median"
        f.attrs["timing_excludes"] = "host<->device transfer for source and mask setup"

        # Axes and inputs.
        f.create_dataset("x", data=x)
        f.create_dataset("Pe_list", data=PE_LIST)
        f.create_dataset("Da_list", data=DA_LIST)
        f.create_dataset("Pe_mesh", data=PE_M)
        f.create_dataset("Da_mesh", data=DA_M)
        f.create_dataset("sources", data=sources)
        f.create_dataset("masks_rfft", data=masks)
        f.create_dataset("kept_modes", data=kept_modes)

        # Accuracy and convergence.
        f.create_dataset("mean_relL2", data=mean_grid)
        f.create_dataset("std_relL2", data=std_grid)
        f.create_dataset("nan_count", data=nan_count_grid)
        f.create_dataset("nonconv_count", data=nonconv_count_grid)
        f.create_dataset("mean_iters", data=mean_iters_grid)
        f.create_dataset("mean_relchg", data=mean_relchg_grid)
        f.create_dataset("mean_c", data=mean_c_grid)
        f.create_dataset("mean_omega", data=mean_omega_grid)
        f.create_dataset("all_relL2", data=all_errs)
        f.create_dataset("all_iters", data=all_iters)
        f.create_dataset("all_relchg", data=all_relchg)
        f.create_dataset("all_omega", data=all_omega)

        # Effective source diagnostic.
        f.create_dataset("mean_Seff_ratio", data=mean_seff_grid)
        f.create_dataset("std_Seff_ratio", data=std_seff_grid)
        f.create_dataset("all_Seff_ratio", data=all_seff)

        # Timing, seconds, batch=N_SOURCES.
        f.create_dataset("t_recast_sec", data=t_recast_grid)
        f.create_dataset("t_spectral_sec", data=t_spectral_grid)
        f.create_dataset("slowdown_recast_over_spectral", data=t_recast_grid / np.maximum(t_spectral_grid, 1e-12))

    print(f"[Done] Saved results: {OUT_H5}")

    # Plots.
    plot_smooth_contour(
        PE_M,
        DA_M,
        mean_grid,
        OUT_PNG_RELL2,
        label="Mean relative L2 error (over sources)",
        title="2D scan: recast + strict bandlimit + Aitken Delta^2 damping",
    )
    print(f"[Done] Saved contour plot: {OUT_PNG_RELL2}")

    plot_smooth_contour(
        PE_M,
        DA_M,
        mean_seff_grid,
        OUT_PNG_SEFF,
        label=r"Mean $\|S_{\mathrm{eff}}\|_2 / \|S\|_2$",
        title=r"Effective source ratio $\|S_{\mathrm{eff}}\|_2 / \|S\|_2$ over (Pe, Da)",
    )
    print(f"[Done] Saved contour plot: {OUT_PNG_SEFF}")

    plot_smooth_contour(
        PE_M,
        DA_M,
        1e3 * t_recast_grid,
        OUT_PNG_TIME,
        label=f"Recast wall time (ms, batch={N_SOURCES})",
        title=f"GPU-batched recast solver time over (Pe, Da), batch={N_SOURCES}",
    )
    print(f"[Done] Saved contour plot: {OUT_PNG_TIME}")

    slowdown = t_recast_grid / np.maximum(t_spectral_grid, 1e-12)
    plot_smooth_contour(
        PE_M,
        DA_M,
        slowdown,
        OUT_PNG_SLOWDOWN,
        label=r"Slowdown $t_{\mathrm{recast}} / t_{\mathrm{spectral}}$",
        title=r"GPU-batched recast vs exact spectral baseline, transfer-free timing",
    )
    print(f"[Done] Saved contour plot: {OUT_PNG_SLOWDOWN}")


if __name__ == "__main__":
    main()
