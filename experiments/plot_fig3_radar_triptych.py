#!/usr/bin/env python3
"""
Figure 3 — 1×3 radar triptych

  (a) DS and NI              — the two base privacy primitives
  (b) ROMM and ROME          — linearly-homomorphic encryption
  (c) CKKS                   — fully-homomorphic encryption
       + ghost outlines of DS+NI and ROME+NI for cross-panel context

Requires exp_A_algorithm_comparison.csv in the results directory.
Discovers datasets the same way plot_results.py does.

Usage:
    python experiments/plot_fig3_radar_triptych.py
    python experiments/plot_fig3_radar_triptych.py --dataset nfcorpus
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = Path("experiments/results")
FIGS_ROOT    = Path("experiments/figures")

DATASET_LABELS = {
    "nfcorpus": "NFCorpus",
    "nq":       "Natural Questions (NQ)",
    "medical":  "Medical KB",
}

# ── Style (matches plot_results.py) ──────────────────────────────────────────
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

ALGO_ORDER = [
    "Scrambling", "Noise", "Combined",
    "Rome", "RomeCombined", "Romm", "RommCombined",
    "Ckks", "CkksCombined",
]
SHORT = {
    "Scrambling": "DS",    "Noise": "NI",       "Combined": "DS+NI",
    "Rome": "ROME",        "RomeCombined": "ROME+NI",
    "Romm": "ROMM",        "RommCombined": "ROMM+NI",
    "Ckks": "CKKS",        "CkksCombined": "CKKS+NI",
}
MARKERS = {
    "DS": "o",     "NI": "s",      "DS+NI": "D",
    "ROME": "^",   "ROME+NI": "*",
    "ROMM": "v",   "ROMM+NI": "s",
    "CKKS": "P",   "CKKS+NI": "X",
}

# Panel-level palettes: base algo gets solid fill, +NI gets dashed/unfilled.
# Colors follow the MATLAB default order for the 7 algorithms with data.
PANEL_COLORS = {
    "DS":       "#0072BD",  "NI":       "#D95319",  "DS+NI":    "#EDB120",
    "ROME":     "#7E2F8E",  "ROME+NI":  "#77AC30",
    "ROMM":     "#808080",  "ROMM+NI":  "#808080",
    "CKKS":     "#4DBEEE",  "CKKS+NI":  "#A2142F",
}
NI_VARIANTS = {"DS+NI", "ROME+NI", "ROMM+NI", "CKKS+NI"}

GHOST_ALPHA = 0.35
GHOST_LW    = 1.0

AXES_LABELS = [
    "Security\n($\\log\\log C_{\\mathrm{attack}}$)",
    "Fidelity\n(Recall, $k{=}10$)",
    "Efficiency\n($1/\\tau_{\\mathrm{total}}$)",
]


# ── Security model ───────────────────────────────────────────────────────────
ATTACK_OPS = {
    "DS":      384**3,
    "NI":      384,
    "DS+NI":   384**3,
    "ROME":    384*384*512 + 512**3,
    "ROME+NI": 384*384*512 + 512**3,
    "ROMM":    384**3,
    "ROMM+NI": 384**3,
}

_ckks_log10 = 0.292 * 8192 * math.log10(2)
_ckks_dlog  = math.log10(_ckks_log10)

def _double_log(ops):
    return math.log10(math.log10(ops))

SECURITY_RAW = {a: _double_log(ATTACK_OPS[a]) for a in ATTACK_OPS}
SECURITY_RAW["CKKS"]    = _ckks_dlog
SECURITY_RAW["CKKS+NI"] = _ckks_dlog


# ── Normalisation helpers ────────────────────────────────────────────────────

def _minmax(d):
    vals = np.array(list(d.values()))
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def compute_normalised_metrics(df_a):
    """Return (sec_norm, eff_norm, rec_norm) dicts keyed by short label."""
    present = [a for a in ALGO_ORDER if a in df_a["algorithm"].unique()]
    agg = (
        df_a[df_a["algorithm"].isin(present)]
        .groupby("algorithm")
        .agg(total_us=("enc_total_us", "mean"), recall=("recall_at_k", "mean"))
    )

    efficiency_raw, recall_raw = {}, {}
    for algo_key in present:
        label = SHORT[algo_key]
        efficiency_raw[label] = 1.0 / agg.loc[algo_key, "total_us"]
        recall_raw[label]     = agg.loc[algo_key, "recall"]

    _sec_lin = _minmax(SECURITY_RAW)
    sec_norm = {k: v ** 0.4 for k, v in _sec_lin.items()}

    _eff_log = {k: math.log10(v) for k, v in efficiency_raw.items()}
    eff_norm = _minmax(_eff_log)

    _recall_floor = 0.97
    rec_norm = {k: max(0.0, (v - _recall_floor) / (1.0 - _recall_floor))
                for k, v in recall_raw.items()}

    return sec_norm, eff_norm, rec_norm


# ── Drawing helpers ──────────────────────────────────────────────────────────

LABEL_RADIUS = 1.22

def _radar_setup(ax):
    """Configure a single polar axis with our 3-metric layout."""
    N = len(AXES_LABELS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Suppress auto labels, place them manually for precise control
    ax.set_thetagrids(np.degrees(angles[:-1]), [""] * N)

    # Security (top): just inside the ring, below the title
    ax.text(angles[0], LABEL_RADIUS * 0.88, AXES_LABELS[0],
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            transform=ax.transData)
    # Fidelity (bottom-right)
    ax.text(angles[1] + 0.14, LABEL_RADIUS, AXES_LABELS[1],
            ha="left", va="top", fontsize=9, fontweight="bold",
            transform=ax.transData)
    # Efficiency (bottom-left)
    ax.text(angles[2] - 0.14, LABEL_RADIUS, AXES_LABELS[2],
            ha="right", va="top", fontsize=9, fontweight="bold",
            transform=ax.transData)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color="#ccc", linewidth=0.6)
    ax.xaxis.grid(True, color="#ccc", linewidth=0.6)

    return angles


def _plot_algo(ax, angles, label, sec, eff, rec, *,
               ghost=False, zorder=3):
    """Plot one algorithm on a radar axis."""
    vals = [sec, rec, eff] + [sec]
    color = PANEL_COLORS[label]
    is_ni = label in NI_VARIANTS

    if ghost:
        kw = dict(
            color=color, linewidth=1.4, linestyle=":",
            marker=MARKERS[label], markersize=9,
            label=label, zorder=zorder,
        )
        if is_ni:
            kw.update(markerfacecolor="white", markeredgecolor=color,
                      markeredgewidth=1.5)
        ax.plot(angles, vals, **kw)
        ax.fill(angles, vals, color=color, alpha=0.04, zorder=zorder - 1)
    else:
        kw = dict(
            color=color, linewidth=1.4, marker=MARKERS[label],
            markersize=9, label=label, zorder=zorder,
        )
        if is_ni:
            kw.update(linestyle="--", markerfacecolor="white",
                      markeredgecolor=color, markeredgewidth=1.5)
        ax.plot(angles, vals, **kw)
        ax.fill(angles, vals, color=color, alpha=0.06, zorder=zorder - 1)


# ── Main figure ──────────────────────────────────────────────────────────────

def plot_triptych(results_dir, figs_dir, ds_tag):
    path = results_dir / "exp_A_algorithm_comparison.csv"
    if not path.exists():
        print(f"  Fig 3 triptych: skipped (no {path.name})")
        return

    ds_label = DATASET_LABELS.get(ds_tag, ds_tag)
    print(f"  Fig 3 triptych: {ds_label} …")

    df_a = pd.read_csv(path)
    if "trial" not in df_a.columns:
        df_a["trial"] = 1

    sec, eff, rec = compute_normalised_metrics(df_a)

    PANEL_A = ["DS", "NI", "DS+NI"]
    PANEL_B = ["ROME", "ROME+NI", "ROMM", "ROMM+NI"]
    PANEL_C_SOLID = ["CKKS", "CKKS+NI"]

    def _area(label):
        return sec.get(label, 0) * eff.get(label, 0) * rec.get(label, 0)

    he_ni = [l for l in ["ROME+NI", "ROMM+NI"] if l in sec]
    best_he_ni = max(he_ni, key=_area) if he_ni else None
    PANEL_C_GHOSTS = ["DS+NI"]
    if best_he_ni:
        PANEL_C_GHOSTS.append(best_he_ni)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                             subplot_kw={"polar": True})

    titles = [
        "(a) DS and NI",
        "(b) ROMM and ROME",
        "(c) CKKS with DS+NI and ROME+NI",
    ]

    for ax_idx, (ax, panel_algos, title) in enumerate(zip(
        axes, [PANEL_A, PANEL_B, PANEL_C_SOLID], titles
    )):
        angles = _radar_setup(ax)

        available = [l for l in panel_algos if l in sec and l in eff and l in rec]
        for i, label in enumerate(available):
            _plot_algo(ax, angles, label, sec[label], eff[label], rec[label],
                       zorder=3 + i)

        if ax_idx == 2:
            for i, label in enumerate(PANEL_C_GHOSTS):
                if label in sec and label in eff and label in rec:
                    _plot_algo(ax, angles, label,
                               sec[label], eff[label], rec[label],
                               ghost=True, zorder=2)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=32)
        ax.legend(loc="upper left", bbox_to_anchor=(0.92, 1.08),
                  fontsize=7.5, framealpha=0.9, handlelength=1.8)

    fig.subplots_adjust(left=0.01, right=0.97, top=0.84, bottom=0.14, wspace=-0.12)
    save(fig, figs_dir, "fig3_security_cost_triptych")


def save(fig, figs_dir, name):
    figs_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    print(f"  saved {name}.png / .pdf  →  {figs_dir}/")
    plt.close(fig)


# ── Discovery (mirrors plot_results.py) ─────────────────────────────────────

def discover_datasets():
    datasets = []
    for subdir in sorted(RESULTS_ROOT.iterdir()):
        if subdir.is_dir() and any(subdir.glob("exp_*.csv")):
            datasets.append((subdir.name, subdir))
    # Also include CSVs sitting directly in the results root (legacy layout).
    if any(RESULTS_ROOT.glob("exp_*.csv")):
        datasets.append(("legacy", RESULTS_ROOT))
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Fig 3: radar triptych")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Limit to one dataset (e.g. nfcorpus)")
    args = parser.parse_args()

    datasets = discover_datasets()
    if not datasets:
        print("No experiment results found in", RESULTS_ROOT)
        sys.exit(1)

    if args.dataset:
        datasets = [(t, p) for t, p in datasets if t == args.dataset]
        if not datasets:
            print(f"Dataset '{args.dataset}' not found.")
            sys.exit(1)

    for ds_tag, results_dir in datasets:
        figs_dir = FIGS_ROOT if ds_tag == "legacy" else FIGS_ROOT / ds_tag
        plot_triptych(results_dir, figs_dir, ds_tag)

    print("Done.")


if __name__ == "__main__":
    main()
