#!/usr/bin/env python3
"""
ADR_1D_dataset.py

Dataset generator for 1D dimensionless steady advection-diffusion-reaction equation
with periodic BC on [0, 1):

    -u''(x) + Pe * u'(x) + Da * u(x) = S(x)

User requirements:
  (1) Use the dimensionless form with only (Pe, Da).
  (2) Fourier spectral method using NumPy FFT (no CuPy).
  (3) For each source: periodic GRF, exactly zero-mean, length scale l=0.08 (set in __main__),
      then rescale each source to [-1, 1] independently.
  (4) Save global min/max for the entire dataset (for ML normalization).
  (5) Save to HDF5 WITHOUT compression.
  (6) After saving, plot samples #1, #10, #100 in a single figure with 3 panels,
      overlay source and solution using twin y-axes.
  (7) Add one more figure in main: for the same three samples, quantify the relative
      influence (RMS share) of the three ADR terms: -u'', Pe*u', Da*u.
"""

from __future__ import annotations

import os
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Grid and spectral utilities
# -----------------------------
def make_periodic_grid(L: float, N: int) -> tuple[np.ndarray, float]:
    """
    Periodic grid on [0, L) with N points (endpoint excluded for FFT consistency).
    """
    x = np.linspace(0.0, L, N, endpoint=False, dtype=np.float64)
    dx = L / N
    return x, dx


def wavenumbers(L: float, N: int) -> np.ndarray:
    """
    FFT-compatible wavenumbers k (rad/unit length), shape (N,).
    """
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    return k.astype(np.float64)


def rescale_to_minus1_plus1(s: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    """
    Rescale a 1D array to [-1, 1] using its own min/max.
    """
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    rng = s_max - s_min
    if rng < eps:
        return np.zeros_like(s)
    return 2.0 * (s - s_min) / rng - 1.0


def rms(a: np.ndarray) -> float:
    """Root-mean-square over grid points."""
    return float(np.sqrt(np.mean(np.square(a))))


# -----------------------------
# Periodic GRF sampling (Fourier)
# -----------------------------
def sample_grf_periodic_se_fourier(
    rng: np.random.Generator,
    k: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    """
    Sample a real-valued, periodic, exactly zero-mean GRF on [0, 1) using
    Fourier-series coefficients with a squared-exponential (SE/RBF) envelope.

    Envelope:
        envelope(k) = exp( -0.5 * (l*k)^2 )

    Enforce:
      - Hermitian symmetry so that ifft is real
      - k=0 mode = 0 so mean is exactly zero

    Note: absolute scaling is irrelevant because each sample is later rescaled to [-1, 1].
    """
    N = k.size
    envelope = np.exp(-0.5 * (length_scale * k) ** 2).astype(np.float64)

    coeff = np.zeros(N, dtype=np.complex128)
    coeff[0] = 0.0 + 0.0j  # exact zero mean

    half = N // 2  # for N odd, half=(N-1)/2

    for i in range(1, half + 1):
        sigma = math.sqrt(envelope[i])
        a = (sigma / math.sqrt(2.0)) * (rng.standard_normal() + 1j * rng.standard_normal())
        coeff[i] = a
        coeff[-i] = np.conjugate(a)

    s = np.fft.ifft(coeff).real.astype(np.float64)
    s -= np.mean(s)  # exact zero mean guard
    return s


# -----------------------------
# Fourier spectral solver + derivatives
# -----------------------------
def solve_adr_fourier(S: np.ndarray, Pe: float, Da: float, k: np.ndarray) -> np.ndarray:
    """
    Solve:
        -u'' + Pe*u' + Da*u = S
    in Fourier space:
        (k^2 + i*Pe*k + Da) * u_hat = S_hat
    """
    S_hat = np.fft.fft(S)
    denom = (k * k) + 1j * Pe * k + Da

    tiny = 1e-14
    denom = np.where(np.abs(denom) < tiny, tiny + 0j, denom)

    u_hat = S_hat / denom
    u_hat[0] = S_hat[0] / Da  # well-defined since Da>0

    u = np.fft.ifft(u_hat).real.astype(np.float64)
    return u


def spectral_derivatives(u: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute u' and u'' using Fourier spectral differentiation.
    """
    u_hat = np.fft.fft(u)
    up = np.fft.ifft(1j * k * u_hat).real.astype(np.float64)
    upp = np.fft.ifft(-(k * k) * u_hat).real.astype(np.float64)
    return up, upp


# -----------------------------
# Plotting: sample curves (twin axes)
# -----------------------------
def plot_selected_samples_twin_axes(
    x: np.ndarray,
    S_all: np.ndarray,
    u_all: np.ndarray,
    out_png: str,
    indices_1based: list[int],
) -> None:
    """
    Plot selected samples in one figure with multiple panels.
    Each panel uses twin y-axes because u is typically O(1/Da) compared to S in [-1,1].
    """
    idx0 = [i - 1 for i in indices_1based]
    fig, axes = plt.subplots(len(idx0), 1, figsize=(10, 10), sharex=True)

    if len(idx0) == 1:
        axes = [axes]

    for ax, idx, sample_id in zip(axes, idx0, indices_1based):
        ax_u = ax
        ax_s = ax.twinx()

        p_u, = ax_u.plot(x, u_all[idx], lw=1.6, label="u(x)")
        p_s, = ax_s.plot(x, S_all[idx], lw=1.2, ls="--", label="S(x)")

        ax_u.set_ylabel("u(x)")
        ax_s.set_ylabel("S(x)")
        ax_u.grid(True, alpha=0.3)
        ax_u.set_title(f"Sample {sample_id} (u: left axis, S: right axis)")
        ax_u.legend([p_u, p_s], ["u(x)", "S(x)"], loc="upper right", frameon=False)

    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plotting: term contribution shares
# -----------------------------
def plot_term_shares(
    x: np.ndarray,
    S_all: np.ndarray,
    u_all: np.ndarray,
    k: np.ndarray,
    Pe: float,
    Da: float,
    out_png: str,
    indices_1based: list[int],
) -> None:
    """
    For selected samples, compute ADR terms:
      Tdiff = -u''
      Tadv  = Pe*u'
      Treact= Da*u
    and plot RMS-share bar charts.

    Also prints RMS of residual r = Tdiff + Tadv + Treact - S.
    """
    idx0 = [i - 1 for i in indices_1based]

    fig, axes = plt.subplots(len(idx0), 1, figsize=(9, 9))
    if len(idx0) == 1:
        axes = [axes]

    term_names = ["-u'' (diffusion)", "Pe*u' (advection)", "Da*u (reaction)"]

    for ax, idx, sid in zip(axes, idx0, indices_1based):
        S = S_all[idx]
        u = u_all[idx]
        up, upp = spectral_derivatives(u, k)

        Tdiff = -upp
        Tadv = Pe * up
        Treact = Da * u

        # RMS magnitudes
        r_diff = rms(Tdiff)
        r_adv = rms(Tadv)
        r_react = rms(Treact)
        r_sum = r_diff + r_adv + r_react

        shares = np.array([r_diff, r_adv, r_react], dtype=np.float64) / (r_sum + 1e-30)

        # Residual sanity check
        residual = Tdiff + Tadv + Treact - S
        r_res = rms(residual)
        r_rhs = rms(S)

        print(
            f"Sample {sid}: RMS(S)={r_rhs:.6e}, RMS(residual)={r_res:.6e} | "
            f"RMS terms: diff={r_diff:.6e}, adv={r_adv:.6e}, react={r_react:.6e}"
        )

        # Plot shares as bars
        ax.bar([0, 1, 2], shares)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["diff", "adv", "react"])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("RMS share")
        ax.set_title(
            f"Sample {sid} term RMS shares | "
            f"S_RMS={r_rhs:.2e}, residual_RMS={r_res:.2e}"
        )
        ax.grid(True, axis="y", alpha=0.3)

        # Add text labels
        for j, val in enumerate(shares):
            ax.text(j, min(val + 0.03, 0.98), f"{val*100:.1f}%", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    # User configuration
    L = 1.0
    N = 201
    nsamples = 1000

    Pe = 4.0
    Da = 2.0

    grf_length_scale = 0.08
    seed = 12345

    out_h5 = "data_canonical.h5"
    out_png_curves = "adr1d_samples_1_10_100_twinaxes.png"
    out_png_shares = "adr1d_samples_1_10_100_term_shares.png"

    # Setup
    x, dx = make_periodic_grid(L, N)
    k = wavenumbers(L, N)
    rng = np.random.default_rng(seed)

    # Preallocate
    S_all = np.empty((nsamples, N), dtype=np.float64)
    u_all = np.empty((nsamples, N), dtype=np.float64)

    u_min_global = math.inf
    u_max_global = -math.inf

    # Generate dataset
    for i in range(nsamples):
        S_raw = sample_grf_periodic_se_fourier(rng, k=k, length_scale=grf_length_scale)
        S = rescale_to_minus1_plus1(S_raw)
        u = solve_adr_fourier(S, Pe=Pe, Da=Da, k=k)

        S_all[i, :] = S
        u_all[i, :] = u

        u_min_global = min(u_min_global, float(np.min(u)))
        u_max_global = max(u_max_global, float(np.max(u)))

        if (i + 1) % 100 == 0:
            print(f"generated {i+1}/{nsamples}")

    # Save to HDF5 (no compression)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("source", data=S_all)
        f.create_dataset("solution", data=u_all)

        f.attrs["L"] = float(L)
        f.attrs["N"] = int(N)
        f.attrs["dx"] = float(dx)
        f.attrs["Pe"] = float(Pe)
        f.attrs["Da"] = float(Da)
        f.attrs["nsamples"] = int(nsamples)
        f.attrs["grf_length_scale"] = float(grf_length_scale)
        f.attrs["seed"] = int(seed)

        f.attrs["solution_min"] = float(u_min_global)
        f.attrs["solution_max"] = float(u_max_global)

    print("\nSaved dataset:")
    print(f"  HDF5: {os.path.abspath(out_h5)}")
    print(f"  solution_min (global) = {u_min_global:.8e}")
    print(f"  solution_max (global) = {u_max_global:.8e}")

    # Plot selected samples after saving
    plot_selected_samples_twin_axes(
        x=x,
        S_all=S_all,
        u_all=u_all,
        out_png=out_png_curves,
        indices_1based=[1, 10, 100],
    )
    print(f"Saved figure (curves): {os.path.abspath(out_png_curves)}")

    # New plot: ADR term influence shares for the same three sources
    plot_term_shares(
        x=x,
        S_all=S_all,
        u_all=u_all,
        k=k,
        Pe=Pe,
        Da=Da,
        out_png=out_png_shares,
        indices_1based=[1, 10, 100],
    )
    print(f"Saved figure (term shares): {os.path.abspath(out_png_shares)}")


if __name__ == "__main__":
    main()
