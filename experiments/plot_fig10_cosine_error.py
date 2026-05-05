#!/usr/bin/env python3
"""
Figure 10 — Theoretical & empirical cosine-similarity error vs. noise level.

For multiplicative uniform noise n_i ~ U(1-ε, 1+ε) applied element-wise to
d-dimensional unit vectors, the encrypted cosine similarity is:

    cos_enc = (d · q) / (||d⊙n|| · ||q⊙(1/n)||)

The numerator is exactly d·q (noise cancels), but normalization introduces
a systematic negative bias.  By concentration (large d):

    E[cos_enc] ≈ ρ · sqrt((1 - ε²) / (1 + ε²/3))

Panel (a): signed error  (theory line + MC river band)
Panel (b): absolute error (theory line + MC river band)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

FIGS = Path("experiments/figures")
FIGS.mkdir(exist_ok=True)

# ── Style (matches plot_results.py) ───────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

RIVER_ALPHA = 0.18

RHO_PALETTE = {
    0.3: "#2196F3",
    0.6: "#FF9800",
    0.9: "#4CAF50",
}
RHO_MARKERS = {0.3: "o", 0.6: "s", 0.9: "^"}

D = 384          # all-MiniLM-L6-v2 embedding dimension
N_PAIRS = 1000   # Monte Carlo pairs per (ε, ρ) combination
NOISE_PCTS = np.array([1, 5, 10, 20, 30, 50, 70])
EPSILONS = NOISE_PCTS / 100.0
EPS_FINE = np.linspace(0, 0.70, 200)  # smooth theory curve


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    print(f"  saved {name}.png / .pdf")
    plt.close(fig)


def theoretical_factor(eps):
    """Ratio cos_enc / cos_true under concentration approximation."""
    return np.sqrt((1 - eps**2) / (1 + eps**2 / 3))


def make_pair(rho, d, rng):
    """Generate a random unit-vector pair with cosine similarity ≈ rho."""
    q = rng.standard_normal(d)
    q /= np.linalg.norm(q)
    z = rng.standard_normal(d)
    z -= z.dot(q) * q          # orthogonal to q
    z /= np.linalg.norm(z)
    doc = rho * q + np.sqrt(1 - rho**2) * z
    doc /= np.linalg.norm(doc)
    return doc, q


def apply_noise(doc, query, eps, rng):
    """Inject multiplicative uniform noise and return encrypted cosine sim."""
    n = rng.uniform(1 - eps, 1 + eps, size=doc.shape[0])
    enc_doc = doc * n
    enc_doc /= np.linalg.norm(enc_doc)
    dec_query = query / n
    dec_query /= np.linalg.norm(dec_query)
    return enc_doc.dot(dec_query)


# ── Monte Carlo ───────────────────────────────────────────────────────────────
print("Figure 10: Cosine error vs. noise level (Monte Carlo) …")
rng = np.random.default_rng(42)

# mc_errors[rho][i] = array of N_PAIRS signed errors at EPSILONS[i]
mc_errors = {}
for rho in RHO_PALETTE:
    mc_errors[rho] = []
    pairs = [make_pair(rho, D, rng) for _ in range(N_PAIRS)]
    for eps in EPSILONS:
        errs = []
        for doc, query in pairs:
            cos_enc = apply_noise(doc, query, eps, rng)
            errs.append(cos_enc - rho)
        mc_errors[rho].append(np.array(errs))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for rho, color in RHO_PALETTE.items():
    marker = RHO_MARKERS[rho]

    # Theory (smooth curve)
    theory_signed = rho * (theoretical_factor(EPS_FINE) - 1)
    theory_abs = np.abs(theory_signed)

    # MC stats at discrete noise levels
    mc_signed_mean = np.array([e.mean() for e in mc_errors[rho]])
    mc_signed_std  = np.array([e.std()  for e in mc_errors[rho]])
    mc_abs_mean    = np.array([np.abs(e).mean() for e in mc_errors[rho]])
    mc_abs_std     = np.array([np.abs(e).std()  for e in mc_errors[rho]])

    label_t = f"theory  ρ={rho}"
    label_m = f"MC  ρ={rho}"

    # Panel (a): signed error
    ax = axes[0]
    ax.plot(EPS_FINE * 100, theory_signed, color=color, lw=1.5, ls="--",
            label=label_t, zorder=3)
    ax.plot(NOISE_PCTS, mc_signed_mean, color=color, lw=2,
            marker=marker, ms=7, label=label_m, zorder=4)
    ax.fill_between(NOISE_PCTS,
                    mc_signed_mean - mc_signed_std,
                    mc_signed_mean + mc_signed_std,
                    color=color, alpha=RIVER_ALPHA, linewidth=0, zorder=1)

    # Panel (b): absolute error
    ax = axes[1]
    ax.plot(EPS_FINE * 100, theory_abs, color=color, lw=1.5, ls="--",
            label=label_t, zorder=3)
    ax.plot(NOISE_PCTS, mc_abs_mean, color=color, lw=2,
            marker=marker, ms=7, label=label_m, zorder=4)
    ax.fill_between(NOISE_PCTS,
                    mc_abs_mean - mc_abs_std,
                    mc_abs_mean + mc_abs_std,
                    color=color, alpha=RIVER_ALPHA, linewidth=0, zorder=1)

# Panel (a) formatting
ax = axes[0]
ax.axhline(0, ls="-", color="#555", lw=0.6)
ax.set_xlabel("Noise range (±%)")
ax.set_ylabel("Signed error  (cos_enc − cos_true)")
ax.set_title("(a)  Systematic Bias from Normalization")
ax.legend(fontsize=8, loc="lower left", framealpha=0.85, ncol=2)

# Panel (b) formatting
ax = axes[1]
ax.set_xlabel("Noise range (±%)")
ax.set_ylabel("|cos_enc − cos_true|")
ax.set_title("(b)  Mean Absolute Error")
ax.legend(fontsize=8, loc="upper left", framealpha=0.85, ncol=2)

fig.suptitle(
    f"Cosine Similarity Error vs. Uniform Noise Level  (d = {D}, {N_PAIRS:,} MC pairs)\n"
    "dashed = theory  ρ·(√((1−ε²)/(1+ε²/3)) − 1);  solid + band = Monte Carlo mean ± 1σ",
    fontsize=11, y=1.04)
fig.tight_layout()
save(fig, "fig10_cosine_error")

print("Done.")
