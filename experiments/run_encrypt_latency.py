#!/usr/bin/env python3
"""
Lightweight experiment: encryption latency only (fig2 panel b).

Runs all 7 algorithms × n_trials, collecting only enc_encrypt_us from
a few queries per trial.  Skips baseline collection and recall computation.

Writes exp_A_algorithm_comparison.csv with dummy recall columns so the
existing fig2 plotter works unchanged.

Usage:
    python experiments/run_encrypt_latency.py                  # 5 trials, 5 queries
    python experiments/run_encrypt_latency.py --trials 10      # 10 trials
    python experiments/run_encrypt_latency.py --dataset nfcorpus
"""

import argparse
import csv
import time
from pathlib import Path

import requests

API_BASE = "http://127.0.0.1:3000"

ALGO_NAME_MAP = {
    "Scrambling":   "scrambling",
    "Noise":        "noise",
    "Combined":     "combined",
    "Rome":         "rome",
    "RomeCombined": "rome-combined",
    "Ckks":         "ckks",
    "CkksCombined": "ckks-combined",
}

ALGORITHMS = [
    {"name": "Scrambling",    "noise_min": 0.9, "noise_max": 1.1},
    {"name": "Noise",         "noise_min": 0.9, "noise_max": 1.1},
    {"name": "Combined",      "noise_min": 0.9, "noise_max": 1.1},
    {"name": "Rome",          "noise_min": 1.0, "noise_max": 1.0},
    {"name": "RomeCombined",  "noise_min": 0.9, "noise_max": 1.1},
    {"name": "Ckks",          "noise_min": 1.0, "noise_max": 1.0},
    {"name": "CkksCombined",  "noise_min": 0.9, "noise_max": 1.1},
]

QUERIES = [
    "What are the symptoms of type 2 diabetes?",
    "How is hypertension treated?",
    "What causes heart failure?",
    "Describe the symptoms of lupus",
    "What medications treat depression?",
]


def set_algorithm(api_base, algorithm, noise_min, noise_max):
    api_name = ALGO_NAME_MAP[algorithm]
    payload = {"algorithm": api_name, "noise_min": noise_min, "noise_max": noise_max}
    requests.post(f"{api_base}/api/config", json=payload, timeout=120).raise_for_status()
    r = requests.post(f"{api_base}/api/reencrypt", timeout=3600)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(f"Re-encrypt failed: {data.get('message')}")
    return data


def query_batch(api_base, queries, top_k):
    r = requests.post(f"{api_base}/api/experiment",
                      json={"queries": queries, "top_k": top_k,
                            "include_plaintext_baseline": False},
                      timeout=3600)
    r.raise_for_status()
    return r.json()


def append_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Encryption latency experiment (fig2 panel b)")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--queries", type=int, default=5,
                        help="Queries per trial (just for timing samples)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dataset", default="nfcorpus",
                        help="Dataset subdirectory for results")
    parser.add_argument("--api", default=API_BASE)
    args = parser.parse_args()

    api_base = args.api
    n_trials = args.trials
    queries = QUERIES[:args.queries]
    top_k = args.top_k

    # Health check
    try:
        h = requests.get(f"{api_base}/api/health", timeout=30).json()
        n_docs = h.get("documents", 0)
        print(f"Backend OK: {n_docs} documents, algorithm={h.get('algorithm')}")
    except Exception as e:
        raise SystemExit(f"Backend not reachable at {api_base}: {e}")

    results_dir = Path("experiments/results") / args.dataset
    out = results_dir / "exp_A_algorithm_comparison.csv"
    out.unlink(missing_ok=True)

    n_algos = len(ALGORITHMS)
    total_reencrypts = n_trials * n_algos
    print(f"\nPlan: {n_trials} trials × {n_algos} algorithms = {total_reencrypts} re-encrypts")
    print(f"      {len(queries)} queries per config (timing samples only)\n")

    t_start = time.time()

    for trial in range(1, n_trials + 1):
        print(f"── Trial {trial}/{n_trials} ──")

        for algo_cfg in ALGORITHMS:
            name = algo_cfg["name"]
            noise_min, noise_max = algo_cfg["noise_min"], algo_cfg["noise_max"]

            t0 = time.time()
            reenc = set_algorithm(api_base, name, noise_min, noise_max)
            reenc_s = time.time() - t0

            reenc_per_doc_us = reenc["encrypt_us"] / max(reenc["documents_processed"], 1)

            time.sleep(0.2)
            result = query_batch(api_base, queries, top_k)

            rows = []
            for qm in result["results"]:
                rows.append({
                    "trial": trial,
                    "exp": "A",
                    "algorithm": name,
                    "noise_min": noise_min,
                    "noise_max": noise_max,
                    "top_k": top_k,
                    "query": qm["query"],
                    "recall_at_k": 1.0,
                    "ndcg_at_k": 1.0,
                    "kendall_tau": 1.0,
                    "enc_embed_us": qm["enc_embed_us"],
                    "enc_encrypt_us": qm["enc_encrypt_us"],
                    "enc_search_us": qm["enc_search_us"],
                    "enc_total_us": qm["enc_total_us"],
                    "score_entropy": 0.0,
                    "score_variance": 0.0,
                    "score_distinct_count": 0,
                })
            append_rows(out, rows)

            mean_enc = sum(r["enc_encrypt_us"] for r in rows) / len(rows)
            print(f"  {name:14s}  reenc={reenc_s:5.1f}s  query_enc={mean_enc/1000:7.2f}ms  "
                  f"doc_enc={reenc_per_doc_us/1000:7.2f}ms")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. Results: {out}")


if __name__ == "__main__":
    main()
