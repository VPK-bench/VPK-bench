#!/usr/bin/env bash
# Overnight experiment runner — chains E+I/nfcorpus then A+D+E+I/nq then figures.
# Safe to run while exp D is still in progress; waits for it first.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/phe-overnight.log
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=== overnight runner started at $(ts) ==="

# Wait for any in-flight run_experiments.py (exp D) to finish
INFLIGHT=$(pgrep -f "run_experiments.py" || true)
if [[ -n "$INFLIGHT" ]]; then
  echo "[$(ts)] Waiting for existing run_experiments.py PIDs: $INFLIGHT"
  for pid in $INFLIGHT; do
    while kill -0 "$pid" 2>/dev/null; do sleep 10; done
  done
  echo "[$(ts)] All in-flight experiments finished."
fi

run() {
  echo ""
  echo ">>> [$(ts)] $*"
  python "$@"
  echo "<<< [$(ts)] done: $*"
}

run experiments/run_experiments.py --exp E --dataset nfcorpus
run experiments/run_experiments.py --exp I --dataset nfcorpus

run experiments/run_experiments.py --exp A --dataset nq
run experiments/run_experiments.py --exp D --dataset nq
run experiments/run_experiments.py --exp E --dataset nq
run experiments/run_experiments.py --exp I --dataset nq

run experiments/plot_results.py

echo ""
echo "=== ALL DONE at $(ts) ==="
echo "Figures: experiments/figures/"
echo "Results: experiments/results/"
