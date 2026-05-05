# VPKbench

**A benchmark suite for privacy-preserving vector search under partial homomorphic encryption.**

VPKbench evaluates a family of PHE-based schemes for encrypted similarity search — measuring fidelity (Recall@K, nDCG@K), latency (encrypt, search), and score entropy across two standard IR datasets. The accompanying paper (under review) presents results for nine algorithm configurations on NFCorpus and Natural Questions.

---

## Algorithms benchmarked

| Algorithm | Type | Similarity metric |
|-----------|------|-------------------|
| Scrambling (DS) | Dimensional scrambling | Cosine |
| Noise (NI) | Noise injection | Cosine |
| Combined (DS+NI) | DS + noise | Cosine |
| ROME | Orthogonal matrix rotation | Cosine |
| ROME+NI | ROME + noise | Cosine |
| DIEHARD | Asymmetric dual-matrix | **Dot product** |
| DIEHARD+NI | DIEHARD + noise | Dot product |
| CKKS | BFV/CKKS homomorphic | Cosine |
| CKKS+NI | CKKS + noise | Cosine |

DIEHARD uses dot product because its asymmetric construction (query matrix A, document matrix B, A^T B = I_n) preserves inner products exactly but not vector norms; cosine similarity on expanded vectors distorts rankings.

---

## Requirements

- Rust 1.75+ (`rustup` recommended)
- Python 3.9+
- Docker (for PostgreSQL and Qdrant — optional, in-memory mode available)

```bash
pip install -r requirements.txt
```

---

## Quick start (in-memory mode)

```bash
# 1. Start embedding server (loads all-MiniLM-L6-v2, ~10s)
python embedding_server.py &

# 2. Build and start the benchmark backend (in-memory vector store, no Docker needed)
cargo run --release

# 3. Load NFCorpus (downloads ~3,600 docs via BEIR, caches locally)
python experiments/load_nfcorpus.py --limit 2000

# 4. Run the main algorithm comparison experiment
python experiments/run_experiments.py --exp A --dataset nfcorpus

# 5. Generate figures
python experiments/plot_results.py
```

Figures are written to `experiments/figures/nfcorpus/`.

---

## Full reproducibility (all experiments)

Experiments A, D, E reproduce the paper's primary tables and figures. Each takes a `--n-trials N` argument (default 10) and writes append-only CSVs.

```bash
# NFCorpus
python experiments/run_experiments.py --exp A --dataset nfcorpus
python experiments/run_experiments.py --exp D --dataset nfcorpus
python experiments/run_experiments.py --exp E --dataset nfcorpus

# Natural Questions
python experiments/load_nq.py
python experiments/run_experiments.py --exp A --dataset nq
python experiments/run_experiments.py --exp D --dataset nq
python experiments/run_experiments.py --exp E --dataset nq

# Regenerate all figures
python experiments/plot_results.py
```

For the full unattended pipeline (clean DB → load corpus → all experiments → figures):

```bash
bash experiments/run_nfcorpus_full.sh
```

Expected wall-clock time at N=10 trials on a laptop (NFCorpus, 3,190 docs):

| Experiment | Time |
|------------|------|
| exp_A | ~3.5 h |
| exp_D | ~12 h |
| exp_E | ~17 h |
| Total | ~33 h |

CKKS dominates runtime (~36 s/query). Non-CKKS algorithms complete in seconds per trial.

---

## Pre-computed results

Results from N=10 trials are included in `experiments/results/`:

```
experiments/results/
  nfcorpus/
    exp_A_algorithm_comparison.csv
    exp_D_noise_tradeoff.csv
    exp_E_topk_sensitivity.csv
  nq/
    exp_A_algorithm_comparison.csv
    exp_D_noise_tradeoff.csv
    exp_E_topk_sensitivity.csv
```

Pre-generated figures are in `experiments/figures/{nfcorpus,nq}/`.

---

## Configuration

`experiments/config.yaml` controls all experiment parameters (algorithms, noise levels, K values, trial counts). The backend configuration is in `config_1shard.toml`.

---

## Augmenting results (add new algorithms without re-running all)

```bash
python experiments/run_experiments.py --exp A --dataset nfcorpus --only-algos Diehard,DiehardCombined
```

`--only-algos` appends rows to the existing CSV rather than overwriting it.

---

## Repository structure

```
vpkbench/
├── src/
│   ├── crypto/          # Encryption algorithms (DS, NI, ROME, DIEHARD, CKKS)
│   ├── vectordb/        # In-memory and remote vector storage backends
│   ├── shard/           # Multi-shard fan-out manager
│   ├── vpk/             # VPK orchestration layer (encrypt, search, re-key)
│   ├── embedding/       # Embedding server client
│   └── api/             # HTTP API (Axum)
├── experiments/
│   ├── config.yaml      # Experiment parameters
│   ├── run_experiments.py
│   ├── plot_results.py
│   ├── load_nfcorpus.py
│   ├── load_nq.py
│   ├── results/         # Pre-computed CSVs (N=10)
│   └── figures/         # Pre-generated PDFs
├── tests/               # Integration and homomorphism validation tests
├── benches/             # Criterion microbenchmarks
└── embedding_server.py  # Flask server for all-MiniLM-L6-v2
```

---

## Metrics

- **Recall@K**: fraction of true top-K documents recovered by the encrypted search
- **nDCG@K**: normalized discounted cumulative gain at K
- **Score entropy**: Shannon entropy over a 20-bin histogram of Eve's encrypted similarity scores — proxy for score indistinguishability from the adversary's view
- **Encrypt latency**: wall-clock time to encrypt one query vector (ms)
- **Search latency**: wall-clock time for one encrypted similarity search over the full corpus (ms)

---

## License

MIT
