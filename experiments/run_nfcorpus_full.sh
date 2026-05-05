#!/usr/bin/env bash
# Full NFCorpus experiment pipeline: clean DB → load → run A/D/E/I sweep → plot.
# Designed to be run as a single command so you authorize once.
set -euo pipefail
cd "$(dirname "$0")/.."

API="http://127.0.0.1:3000"
PY=python3
RUN_ID="${RUN_ID:-$(date -u '+%Y%m%d_%H%M%S')}"

echo "══════════════════════════════════════════════════"
echo "  NFCorpus full pipeline  (run-id: $RUN_ID)"
echo "══════════════════════════════════════════════════"

# ── 1. Reset the database (preserves historical CSVs) ───────────────────────
echo ""
echo "── Step 1: Resetting database (historical results preserved) ──"

psql "postgresql://postgres:test@localhost/vpk_test" \
  -c "TRUNCATE documents, index_mapping CASCADE;" 2>&1 || true

# Reset id_map so load_nfcorpus re-uploads from scratch into the clean DB.
# Historical CSVs in experiments/results/nfcorpus/<run-id>/ are untouched.
rm -f experiments/results/nfcorpus_id_map.json
rm -f experiments/results/nfcorpus_ground_truth.json
# Keep the HuggingFace download cache to avoid re-downloading

echo "  ✓ PostgreSQL tables truncated"
echo "  ✓ id_map/ground_truth reset (historical CSVs preserved)"
echo "  ✓ This run will write to: experiments/results/nfcorpus/$RUN_ID/"

# Wait for backend to notice (it reads from DB on demand)
sleep 2

# Verify clean state
DOC_COUNT=$(curl -s "$API/api/health" | $PY -c "import sys,json; print(json.load(sys.stdin).get('documents',0))")
echo "  Backend reports $DOC_COUNT documents (should be 0)"
if [ "$DOC_COUNT" -ne 0 ]; then
  echo "  ✗ ERROR: Backend still has documents. Restart it and re-run."
  exit 1
fi

# ── 2. Corpus-size sweep: load incrementally, run exp I at each size ────────
echo ""
echo "── Step 2: Corpus-size sweep (exp I at each size) ──"

# Sizes for the sweep — last one is the full corpus (3633)
SWEEP_SIZES=(500 1000 2000 3633)

for SIZE in "${SWEEP_SIZES[@]}"; do
  echo ""
  echo "  ── Loading $SIZE docs ──"
  $PY experiments/load_nfcorpus.py --limit "$SIZE"

  echo "  ── Running exp I at corpus_size=$SIZE ──"
  $PY experiments/run_experiments.py --exp I --dataset nfcorpus --run-id "$RUN_ID"
done

# ── 3. Run remaining experiments (A, D, E) on full corpus ───────────────────
# Corpus is already at full 3633 from the last sweep step.
echo ""
echo "── Step 3: Running experiments A, D, E on full corpus ──"

$PY experiments/run_experiments.py --exp A --dataset nfcorpus --run-id "$RUN_ID"
$PY experiments/run_experiments.py --exp D --dataset nfcorpus --run-id "$RUN_ID"
$PY experiments/run_experiments.py --exp E --dataset nfcorpus --run-id "$RUN_ID"

# ── 4. Generate all figures ──────────────────────────────────────────────────
echo ""
echo "── Step 4: Generating figures ──"

$PY experiments/plot_results.py

# Additional plot scripts (triptych, cosine-error variants)
[ -f experiments/plot_fig3_radar_triptych.py ] && $PY experiments/plot_fig3_radar_triptych.py || true
[ -f experiments/plot_fig9_cosine_error.py ]   && $PY experiments/plot_fig9_cosine_error.py   || true
[ -f experiments/plot_fig10_cosine_error.py ]  && $PY experiments/plot_fig10_cosine_error.py  || true

echo ""
echo "══════════════════════════════════════════════════"
echo "  Done. Results: experiments/results/nfcorpus/$RUN_ID/"
echo "  Figures:       experiments/figures/nfcorpus/"
echo "══════════════════════════════════════════════════"
