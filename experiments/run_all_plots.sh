#!/usr/bin/env bash
# Run all figure-generation scripts. Assumes experiments have been run.
set -euo pipefail
cd "$(dirname "$0")/.."

# Main figures (fig1, fig2, fig3 radar, supp figs)
python3 experiments/plot_results.py

# Fig 3 triptych: 1×3 radar by cryptographic strength tier
python3 experiments/plot_fig3_radar_triptych.py

# Standalone cosine-error two-panel figure
python3 experiments/plot_fig9_cosine_error.py

# Standalone cosine-error (variant)
python3 experiments/plot_fig10_cosine_error.py

echo ""
echo "All plots generated."
