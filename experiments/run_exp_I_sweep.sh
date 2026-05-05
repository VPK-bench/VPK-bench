#!/usr/bin/env bash
# Run Experiment I at 7 corpus sizes (multiples of 519 = 3×173).
# NFCorpus has 3,633 docs and 3,633 / 7 = 519, so sizes are:
#   519, 1038, 1557, 2076, 2595, 3114, 3633
#
# The exp_I CSV is append-only: each run adds rows with its corpus_size,
# so the plotting code can group by corpus_size for panel (b).
#
# Prerequisites: docker containers (vpk-postgres, vpk-qdrant) and the
# embedding server must already be running.  This script manages the
# Rust backend lifecycle itself.
#
# Usage:
#   ./experiments/run_exp_I_sweep.sh            # run all 7 sizes
#   ./experiments/run_exp_I_sweep.sh 519 1038   # run only these sizes
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_BASE="http://127.0.0.1:3000"
DATASET="nfcorpus"
STEP=519                           # 3 × 173
ALL_SIZES=(519 1038 1557 2076 2595 3114 3633)

if [[ $# -gt 0 ]]; then
    SIZES=("$@")
else
    SIZES=("${ALL_SIZES[@]}")
fi

log()  { echo "[sweep] $*"; }
die()  { echo "[sweep] ERROR: $*" >&2; exit 1; }

wait_for_port() {
    local name=$1 port=$2 max=${3:-120}
    for i in $(seq 1 "$max"); do
        nc -z 127.0.0.1 "$port" 2>/dev/null && return 0
        sleep 1
    done
    die "$name did not come up on :$port after ${max}s"
}

kill_backend() {
    if [[ -f /tmp/phe-backend.pid ]]; then
        kill "$(cat /tmp/phe-backend.pid)" 2>/dev/null || true
    fi
    lsof -ti :3000 | xargs kill -9 2>/dev/null || true
    sleep 2
}

start_backend() {
    log "Starting Rust backend …"
    nohup cargo run > /tmp/phe-backend.log 2>&1 &
    echo $! > /tmp/phe-backend.pid
    wait_for_port "Backend" 3000 120
    log "Backend up (pid $(cat /tmp/phe-backend.pid))"
}

clear_database() {
    log "Truncating postgres tables …"
    PGPASSWORD=test psql -h localhost -U postgres -d vpk_test -q -c \
        "TRUNCATE documents, index_mapping CASCADE;"
    log "Removing cached ID map …"
    rm -f experiments/results/nfcorpus_id_map.json
}

# ── Preflight checks ─────────────────────────────────────────────────────────

nc -z 127.0.0.1 5432 2>/dev/null || die "postgres not running on :5432"
nc -z 127.0.0.1 6333 2>/dev/null || die "qdrant not running on :6333"
nc -z 127.0.0.1 5001 2>/dev/null || die "embedding server not running on :5001"

# Ensure the per-dataset results dir exists (run_experiments.py writes here)
mkdir -p "experiments/results/${DATASET}"

# If other experiment CSVs only exist in the legacy flat layout, copy them
# into the per-dataset dir so plot_results.py sees everything together.
for f in experiments/results/exp_*.csv; do
    base="$(basename "$f")"
    target="experiments/results/${DATASET}/${base}"
    if [[ -f "$f" && ! -f "$target" ]]; then
        cp "$f" "$target"
        log "Copied legacy $base → ${DATASET}/"
    fi
done

# Delete the old exp_I CSV so we start fresh
rm -f "experiments/results/${DATASET}/exp_I_corpus_size_noise.csv"

log "Corpus sizes to run: ${SIZES[*]}"
log ""

for SIZE in "${SIZES[@]}"; do
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  Corpus size: ${SIZE}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    kill_backend
    clear_database
    start_backend

    log "Uploading ${SIZE} NFCorpus docs …"
    python3 experiments/load_nfcorpus.py --limit "$SIZE"

    log "Running Experiment I (n=${SIZE}) …"
    python3 experiments/run_experiments.py --exp I --dataset "$DATASET"

    log "Done with n=${SIZE}"
    log ""
done

kill_backend
log "All sizes complete.  Results in experiments/results/${DATASET}/exp_I_corpus_size_noise.csv"
log "Run 'python experiments/plot_results.py' to regenerate figures."
