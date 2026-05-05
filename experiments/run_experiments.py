#!/usr/bin/env python3
"""
NeurIPS experiment runner for PHE/VPK.

Each configuration is run n_trials times.  On every trial set_algorithm() is
called anew so the backend generates fresh random encryption matrices — giving
genuine statistical independence across trials.

Supports multiple datasets (--dataset): results and figures are written to
per-dataset subdirectories so both can coexist.

Usage:
    python experiments/run_experiments.py --exp A --dataset nfcorpus
    python experiments/run_experiments.py --exp D --dataset nq
    python experiments/run_experiments.py --exp all --dataset all
    python experiments/run_experiments.py --exp A --dry-run
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import requests
import yaml

CONFIG_PATH = Path("experiments/config.yaml")
RESULTS_DIR = Path("experiments/results")

# ── Dataset definitions ──────────────────────────────────────────────────────

MEDICAL_QUERIES = [
    "What are the symptoms of type 2 diabetes?",
    "How is hypertension treated?",
    "What causes heart failure?",
    "Describe the symptoms of lupus",
    "What medications treat depression?",
    "How does Alzheimer's disease progress?",
    "What are risk factors for stroke?",
    "How is asthma managed long-term?",
    "What are signs of kidney disease?",
    "Describe rheumatoid arthritis treatment",
    "What causes Parkinson's disease?",
    "How is multiple sclerosis diagnosed?",
    "What are symptoms of anemia?",
    "How is hypothyroidism treated?",
    "What causes chronic pain?",
    "Describe COPD management",
    "What are signs of liver disease?",
    "How is epilepsy treated?",
    "What causes osteoporosis?",
    "Describe symptoms of anxiety disorder",
    "What medications are used for HIV/AIDS?",
    "How is pneumonia diagnosed?",
    "What causes inflammatory bowel disease?",
    "Describe treatment for chronic kidney disease",
    "What are symptoms of schizophrenia?",
    "How is bipolar disorder managed?",
    "What causes psoriasis?",
    "Describe sepsis treatment",
    "What are risk factors for cancer?",
    "How is ALS diagnosed?",
]

DATASETS = {
    "nfcorpus": {
        "label": "NFCorpus",
        "cache_dir": Path("experiments/.nfcorpus_cache"),
        "queries_file": "nfcorpus_queries.json",
    },
    "nq": {
        "label": "Natural Questions",
        "cache_dir": Path("experiments/.nq_cache"),
        "queries_file": "nq_queries.json",
    },
}


def load_dataset_queries(dataset_tag: str, n: int) -> list[str]:
    """Load query texts for a dataset, falling back to MEDICAL_QUERIES."""
    if dataset_tag not in DATASETS:
        print(f"  [warn] Unknown dataset '{dataset_tag}', using MEDICAL_QUERIES")
        return MEDICAL_QUERIES[:n]

    ds = DATASETS[dataset_tag]
    queries_path = ds["cache_dir"] / ds["queries_file"]
    if not queries_path.exists():
        print(f"  [warn] {queries_path} not found — using MEDICAL_QUERIES "
              f"(run load_{dataset_tag}.py first to cache dataset queries)")
        return MEDICAL_QUERIES[:n]

    raw = json.loads(queries_path.read_text())
    texts = [q["text"] for q in raw if q.get("text", "").strip()]
    print(f"  [{dataset_tag}] Loaded {len(texts)} queries from cache, using first {min(n, len(texts))}")
    return texts[:n]


def available_datasets() -> list[str]:
    """Return dataset tags that have cached query data."""
    avail = []
    for tag, ds in DATASETS.items():
        if (ds["cache_dir"] / ds["queries_file"]).exists():
            avail.append(tag)
    if not avail:
        print("  [warn] No cached datasets found; use 'medical' for built-in queries")
        avail = ["medical"]
    return avail


# ── API ───────────────────────────────────────────────────────────────────────

ALGO_NAME_MAP = {
    "Scrambling":   "scrambling",
    "Noise":        "noise",
    "Combined":     "combined",
    "Rome":         "rome",
    "RomeCombined": "rome-combined",
    "Romm":         "romm",
    "RommCombined": "romm-combined",
    "Diehard":         "diehard",
    "DiehardCombined": "diehard-combined",
    "Ckks":            "ckks",
    "CkksCombined":    "ckks-combined",
}


def set_algorithm(api_base, algorithm, noise_min=0.9, noise_max=1.1,
                  freeze_rotation=False):
    """Switch algorithm (or just noise when freeze_rotation=True) + re-encrypt.

    freeze_rotation=True: preserves DS / ROME / CKKS rotation matrix;
    only the noise layer is reseeded.  Used by Exp G to isolate noise effects.
    """
    api_name = ALGO_NAME_MAP.get(algorithm, algorithm.lower())
    payload = {"algorithm": api_name, "noise_min": noise_min, "noise_max": noise_max}
    if freeze_rotation:
        payload["freeze_rotation"] = True
    requests.post(f"{api_base}/api/config", json=payload, timeout=30).raise_for_status()
    r = requests.post(f"{api_base}/api/reencrypt", timeout=3600)
    r.raise_for_status()
    if not r.json().get("success"):
        raise RuntimeError(f"Re-encrypt failed: {r.json().get('message')}")


def query_batch(api_base, queries, top_k):
    r = requests.post(f"{api_base}/api/experiment",
                      json={"queries": queries, "top_k": top_k,
                            "include_plaintext_baseline": False},
                      timeout=3600)
    r.raise_for_status()
    return r.json()


def health_check(api_base):
    try:
        return requests.get(f"{api_base}/api/health", timeout=5).status_code == 200
    except Exception:
        return False


def corpus_size(api_base):
    """Return the number of documents currently in the corpus."""
    try:
        return requests.get(f"{api_base}/api/health", timeout=5).json().get("documents", 0)
    except Exception:
        return 0


# ── Metrics (Python, vs external baseline) ───────────────────────────────────

def recall_at_k(gt, enc, k):
    k = min(k, len(gt), len(enc))
    return sum(1 for x in enc[:k] if x in set(gt[:k])) / k if k else 1.0

def ndcg_at_k(gt_ids, gt_scores, enc, k):
    k = min(k, len(enc))
    rel = {i: max(s, 0.0) for i, s in zip(gt_ids, gt_scores)}
    dcg  = sum(rel.get(enc[i], 0.0) / math.log2(i+2) for i in range(k))
    idcg = sum(s / math.log2(i+2) for i, s in enumerate(sorted(rel.values(), reverse=True)[:k]))
    return dcg / idcg if idcg else 1.0

def kendall_tau(gt, enc):
    pos_g = {v: i for i, v in enumerate(gt)}
    pos_e = {v: i for i, v in enumerate(enc)}
    common = [v for v in gt if v in pos_e]
    n = len(common)
    if n < 2: return 1.0
    c = d = 0
    for i in range(n):
        for j in range(i+1, n):
            if (pos_g[common[i]] < pos_g[common[j]]) == (pos_e[common[i]] < pos_e[common[j]]):
                c += 1
            else:
                d += 1
    return (c - d) / (c + d) if (c + d) else 1.0

def score_entropy(scores, n_bins=20):
    if len(scores) < 2: return 0.0
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12: return 0.0
    w = (hi - lo) / n_bins
    bins = [0] * n_bins
    for s in scores:
        bins[min(int((s-lo)/w), n_bins-1)] += 1
    n = len(scores)
    return -sum((c/n)*math.log(c/n) for c in bins if c)

def score_variance(scores):
    return float(np.var(scores)) if len(scores) > 1 else 0.0

def score_distinct(scores):
    return len({round(s, 4) for s in scores})


# ── Baseline ─────────────────────────────────────────────────────────────────

def collect_baseline(api_base, queries, top_k):
    """Scrambling baseline: new keys, re-encrypt, then query once."""
    set_algorithm(api_base, "Scrambling", 1.0, 1.0)
    time.sleep(0.3)
    result = query_batch(api_base, queries, top_k)
    return {qm["query"]: {"doc_ids": qm["enc_doc_ids"], "scores": qm["enc_scores"]}
            for qm in result["results"]}


# ── CSV ───────────────────────────────────────────────────────────────────────

def append_rows(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        w.writerows(rows)


# ── Experiments ───────────────────────────────────────────────────────────────

def exp_A(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment A: Algorithm Comparison ===")
    ecfg    = cfg["exp_A"]
    top_k   = ecfg["top_k"]
    n_trials = cfg["n_trials"]
    out      = results_dir / "exp_A_algorithm_comparison.csv"
    algo_list = [ac for ac in ecfg["algorithms"] if only_algos is None or ac["name"] in only_algos]
    if not only_algos:
        out.unlink(missing_ok=True)

    if dry_run:
        for ac in algo_list:
            print(f"  [dry-run] {ac['name']} × {n_trials} trials")
        return

    for trial in range(1, n_trials + 1):
        print(f"\n  ── Trial {trial}/{n_trials} ──")
        baseline = collect_baseline(api_base, queries, top_k)

        for algo_cfg in algo_list:
            name      = algo_cfg["name"]
            noise_min = algo_cfg["noise_min"]
            noise_max = algo_cfg["noise_max"]
            set_algorithm(api_base, name, noise_min, noise_max)
            time.sleep(0.3)
            result = query_batch(api_base, queries, top_k)
            rows = []
            for qm in result["results"]:
                gt = baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"], "scores": qm["enc_scores"]})
                rows.append({
                    "trial": trial, "exp": "A",
                    "algorithm": name, "noise_min": noise_min, "noise_max": noise_max, "top_k": top_k,
                    "query": qm["query"],
                    "recall_at_k":          round(recall_at_k(gt["doc_ids"], qm["enc_doc_ids"], top_k), 4),
                    "ndcg_at_k":            round(ndcg_at_k(gt["doc_ids"], gt["scores"], qm["enc_doc_ids"], top_k), 4),
                    "kendall_tau":          round(kendall_tau(gt["doc_ids"], qm["enc_doc_ids"]), 4),
                    "enc_embed_us":         qm["enc_embed_us"],
                    "enc_encrypt_us":       qm["enc_encrypt_us"],
                    "enc_search_us":        qm["enc_search_us"],
                    "enc_total_us":         qm["enc_total_us"],
                    "score_entropy":        round(score_entropy(qm["enc_scores"]), 4),
                    "score_variance":       round(score_variance(qm["enc_scores"]), 6),
                    "score_distinct_count": score_distinct(qm["enc_scores"]),
                })
            append_rows(out, rows)
            mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
            print(f"    {name:12s}  recall={mean_r:.3f}")


def exp_D(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment D: Noise–Security Tradeoff ===")
    ecfg     = cfg["exp_D"]
    top_k    = ecfg["top_k"]
    n_trials = cfg["n_trials"]
    out      = results_dir / "exp_D_noise_tradeoff.csv"
    algo_list = [a for a in ecfg["algorithms"] if only_algos is None or a in only_algos]
    if not only_algos:
        out.unlink(missing_ok=True)

    if dry_run:
        for a in algo_list:
            for nc in ecfg["noise_levels"]:
                print(f"  [dry-run] {a} {nc['label']} × {n_trials} trials")
        return

    for trial in range(1, n_trials + 1):
        print(f"\n  ── Trial {trial}/{n_trials} ──")
        baseline = collect_baseline(api_base, queries, top_k)

        for algo_name in algo_list:
            for noise_cfg in ecfg["noise_levels"]:
                noise_min, noise_max = noise_cfg["noise_min"], noise_cfg["noise_max"]
                label = noise_cfg["label"]
                set_algorithm(api_base, algo_name, noise_min, noise_max)
                time.sleep(0.3)
                result = query_batch(api_base, queries, top_k)
                rows = []
                for qm in result["results"]:
                    gt = baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"], "scores": qm["enc_scores"]})
                    rows.append({
                        "trial": trial, "exp": "D",
                        "algorithm": algo_name,
                        "noise_label": label, "noise_min": noise_min, "noise_max": noise_max,
                        "noise_range_pct": round((noise_max - 1.0) * 100, 1),
                        "top_k": top_k, "query": qm["query"],
                        "recall_at_k":          round(recall_at_k(gt["doc_ids"], qm["enc_doc_ids"], top_k), 4),
                        "ndcg_at_k":            round(ndcg_at_k(gt["doc_ids"], gt["scores"], qm["enc_doc_ids"], top_k), 4),
                        "kendall_tau":          round(kendall_tau(gt["doc_ids"], qm["enc_doc_ids"]), 4),
                        "score_entropy":        round(score_entropy(qm["enc_scores"]), 4),
                        "score_variance":       round(score_variance(qm["enc_scores"]), 6),
                        "score_distinct_count": score_distinct(qm["enc_scores"]),
                        "enc_total_us":         qm["enc_total_us"],
                    })
                append_rows(out, rows)
                mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
                print(f"    {algo_name:12s} {label:4s}  recall={mean_r:.3f}")


def _scaling_exp(cfg_key, cfg, api_base, dry_run, results_dir, out_name, loop_fn, augment=False):
    ecfg     = cfg[cfg_key]
    n_trials = cfg["n_trials"]
    if dry_run:
        print(f"  [dry-run] {n_trials} trials"); return
    out = results_dir / out_name
    if not augment:
        out.unlink(missing_ok=True)
    for trial in range(1, n_trials + 1):
        print(f"  ── Trial {trial}/{n_trials} ──")
        loop_fn(ecfg, api_base, out, trial)


def exp_B(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment B: Repo Size Scaling ===")
    q10 = queries[:10]

    def loop(ecfg, api_base, out, trial):
        baseline = collect_baseline(api_base, q10, ecfg["top_k"])
        for algo in ecfg["algorithms"]:
            for repo_size in ecfg["repo_sizes"]:
                print(f"    {algo} repo={repo_size}")
                set_algorithm(api_base, algo)
                time.sleep(0.2)
                result = query_batch(api_base, q10, ecfg["top_k"])
                rows = [{"trial": trial, "exp": "B", "algorithm": algo,
                         "repo_size": repo_size, "top_k": ecfg["top_k"], "query": qm["query"],
                         "recall_at_k": round(recall_at_k(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                             qm["enc_doc_ids"], ecfg["top_k"]), 4),
                         "enc_embed_us": qm["enc_embed_us"],
                         "enc_encrypt_us": qm["enc_encrypt_us"],
                         "enc_search_us": qm["enc_search_us"],
                         "enc_total_us": qm["enc_total_us"]}
                        for qm in result["results"]]
                append_rows(out, rows)

    _scaling_exp("exp_B", cfg, api_base, dry_run, results_dir, "exp_B_repo_scaling.csv", loop)


def exp_C(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment C: Shard Scaling ===")
    q10 = queries[:10]

    def loop(ecfg, api_base, out, trial):
        baseline = collect_baseline(api_base, q10, ecfg["top_k"])
        for algo in ecfg["algorithms"]:
            for n_shards in ecfg["shard_counts"]:
                print(f"    {algo} shards={n_shards}")
                set_algorithm(api_base, algo)
                time.sleep(0.2)
                result = query_batch(api_base, q10, ecfg["top_k"])
                rows = [{"trial": trial, "exp": "C", "algorithm": algo,
                         "n_shards": n_shards, "top_k": ecfg["top_k"], "query": qm["query"],
                         "recall_at_k": round(recall_at_k(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                             qm["enc_doc_ids"], ecfg["top_k"]), 4),
                         "enc_search_us": qm["enc_search_us"],
                         "enc_total_us": qm["enc_total_us"]}
                        for qm in result["results"]]
                append_rows(out, rows)

    _scaling_exp("exp_C", cfg, api_base, dry_run, results_dir, "exp_C_shard_scaling.csv", loop)


def exp_E(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment E: TopK Sensitivity ===")
    q10 = queries[:10]

    def loop(ecfg, api_base, out, trial):
        max_k    = max(ecfg["k_values"])
        baseline = collect_baseline(api_base, q10, max_k)
        algo_list = [a for a in ecfg["algorithms"] if only_algos is None or a in only_algos]
        for algo in algo_list:
            set_algorithm(api_base, algo)
            time.sleep(0.2)
            for k in ecfg["k_values"]:
                result = query_batch(api_base, q10, k)
                rows = [{"trial": trial, "exp": "E", "algorithm": algo,
                         "top_k": k, "query": qm["query"],
                         "recall_at_k": round(recall_at_k(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                             qm["enc_doc_ids"], k), 4),
                         "ndcg_at_k": round(ndcg_at_k(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"], "scores": qm["enc_scores"]})["doc_ids"],
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"], "scores": qm["enc_scores"]})["scores"],
                             qm["enc_doc_ids"], k), 4),
                         "kendall_tau": round(kendall_tau(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                             qm["enc_doc_ids"]), 4)}
                        for qm in result["results"]]
                append_rows(out, rows)
            print(f"    {algo} done")

    _scaling_exp("exp_E", cfg, api_base, dry_run, results_dir, "exp_E_topk_sensitivity.csv", loop,
                 augment=only_algos is not None)


def exp_F(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment F: Padding Ratio ===")
    q10 = queries[:10]

    def loop(ecfg, api_base, out, trial):
        baseline = collect_baseline(api_base, q10, ecfg["top_k"])
        for algo in ecfg["algorithms"]:
            for ratio in ecfg["padding_ratios"]:
                set_algorithm(api_base, algo)
                time.sleep(0.2)
                result = query_batch(api_base, q10, ecfg["top_k"])
                rows = [{"trial": trial, "exp": "F", "algorithm": algo,
                         "padding_ratio": ratio, "top_k": ecfg["top_k"], "query": qm["query"],
                         "recall_at_k": round(recall_at_k(
                             baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                             qm["enc_doc_ids"], ecfg["top_k"]), 4),
                         "enc_encrypt_us": qm["enc_encrypt_us"],
                         "enc_total_us": qm["enc_total_us"]}
                        for qm in result["results"]]
                append_rows(out, rows)
                print(f"    {algo} padding={ratio}x")

    _scaling_exp("exp_F", cfg, api_base, dry_run, results_dir, "exp_F_padding_ratio.csv", loop)


def exp_G(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment G: Noise Sweep — Frozen DS Matrix ===")
    ecfg     = cfg["exp_G"]
    top_k    = ecfg["top_k"]
    n_trials = cfg["n_trials"]
    out      = results_dir / "exp_G_frozen_ds_noise.csv"
    out.unlink(missing_ok=True)

    if dry_run:
        for nc in ecfg["noise_levels"]:
            print(f"  [dry-run] Combined frozen-DS {nc['label']} × {n_trials} trials")
        return

    for trial in range(1, n_trials + 1):
        print(f"\n  ── Trial {trial}/{n_trials} ──")
        baseline = collect_baseline(api_base, queries, top_k)

        for i, noise_cfg in enumerate(ecfg["noise_levels"]):
            noise_min, noise_max = noise_cfg["noise_min"], noise_cfg["noise_max"]
            label = noise_cfg["label"]

            freeze = (i > 0)
            set_algorithm(api_base, "Combined", noise_min, noise_max,
                          freeze_rotation=freeze)
            time.sleep(0.3)
            result = query_batch(api_base, queries, top_k)
            rows = []
            for qm in result["results"]:
                gt = baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"],
                                                "scores": qm["enc_scores"]})
                rows.append({
                    "trial": trial, "exp": "G",
                    "algorithm": "Combined",
                    "noise_label": label, "noise_min": noise_min, "noise_max": noise_max,
                    "noise_range_pct": round((noise_max - 1.0) * 100, 1),
                    "top_k": top_k, "query": qm["query"],
                    "recall_at_k":    round(recall_at_k(gt["doc_ids"],
                                                        qm["enc_doc_ids"], top_k), 4),
                    "ndcg_at_k":      round(ndcg_at_k(gt["doc_ids"], gt["scores"],
                                                       qm["enc_doc_ids"], top_k), 4),
                    "score_entropy":  round(score_entropy(qm["enc_scores"]), 4),
                    "score_variance": round(score_variance(qm["enc_scores"]), 6),
                    "enc_total_us":   qm["enc_total_us"],
                })
            append_rows(out, rows)
            mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
            print(f"    Combined {label:4s} (freeze={freeze})  recall={mean_r:.3f}")


def exp_H(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment H: DS Divergence Covariate ===")
    ecfg     = cfg["exp_H"]
    top_k    = ecfg["top_k"]
    n_trials = cfg["n_trials"]
    out      = results_dir / "exp_H_ds_divergence.csv"
    out.unlink(missing_ok=True)

    if dry_run:
        print(f"  [dry-run] {n_trials} trials × {len(ecfg['noise_levels'])} noise levels")
        return

    for trial in range(1, n_trials + 1):
        print(f"\n  ── Trial {trial}/{n_trials} ──")
        baseline = collect_baseline(api_base, queries, top_k)

        set_algorithm(api_base, "Combined", 1.0, 1.0)
        time.sleep(0.3)
        ds_result = query_batch(api_base, queries, top_k)
        ds_recall_scores = [
            recall_at_k(
                baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"]})["doc_ids"],
                qm["enc_doc_ids"], top_k)
            for qm in ds_result["results"]
        ]
        ds_quality = round(sum(ds_recall_scores) / len(ds_recall_scores), 4)
        print(f"    DS quality (noise=0%): {ds_quality:.3f}")

        for i, noise_cfg in enumerate(ecfg["noise_levels"]):
            noise_min, noise_max = noise_cfg["noise_min"], noise_cfg["noise_max"]
            label = noise_cfg["label"]

            if noise_min == 1.0 and noise_max == 1.0:
                result = ds_result
            else:
                set_algorithm(api_base, "Combined", noise_min, noise_max,
                              freeze_rotation=True)
                time.sleep(0.3)
                result = query_batch(api_base, queries, top_k)

            rows = []
            for qm in result["results"]:
                gt = baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"],
                                                "scores": qm["enc_scores"]})
                rows.append({
                    "trial": trial, "exp": "H",
                    "algorithm": "Combined",
                    "ds_quality": ds_quality,
                    "noise_label": label, "noise_min": noise_min, "noise_max": noise_max,
                    "noise_range_pct": round((noise_max - 1.0) * 100, 1),
                    "top_k": top_k, "query": qm["query"],
                    "recall_at_k":    round(recall_at_k(gt["doc_ids"],
                                                        qm["enc_doc_ids"], top_k), 4),
                    "ndcg_at_k":      round(ndcg_at_k(gt["doc_ids"], gt["scores"],
                                                       qm["enc_doc_ids"], top_k), 4),
                    "score_entropy":  round(score_entropy(qm["enc_scores"]), 4),
                    "score_variance": round(score_variance(qm["enc_scores"]), 6),
                    "enc_total_us":   qm["enc_total_us"],
                })
            append_rows(out, rows)
            mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
            print(f"    {label:4s}  recall={mean_r:.3f}")


def exp_I(cfg, api_base, dry_run, results_dir, queries, only_algos=None):
    print("\n=== Experiment I: Corpus Size × Noise Tradeoff ===")
    ecfg     = cfg["exp_I"]
    top_k    = ecfg["top_k"]
    n_trials = cfg["n_trials"]
    out      = results_dir / "exp_I_corpus_size_noise.csv"

    n_docs = corpus_size(api_base)
    print(f"  Corpus size: {n_docs} documents")

    if dry_run:
        print(f"  [dry-run] {n_trials} trials × {len(ecfg['noise_levels'])} noise levels")
        return

    for trial in range(1, n_trials + 1):
        print(f"\n  ── Trial {trial}/{n_trials} ──")
        baseline = collect_baseline(api_base, queries, top_k)

        for i, noise_cfg in enumerate(ecfg["noise_levels"]):
            noise_min, noise_max = noise_cfg["noise_min"], noise_cfg["noise_max"]
            label = noise_cfg["label"]
            set_algorithm(api_base, "Combined", noise_min, noise_max,
                          freeze_rotation=(i > 0))
            time.sleep(0.3)
            result = query_batch(api_base, queries, top_k)
            rows = []
            for qm in result["results"]:
                gt = baseline.get(qm["query"], {"doc_ids": qm["enc_doc_ids"],
                                                "scores": qm["enc_scores"]})
                rows.append({
                    "trial": trial, "exp": "I",
                    "algorithm": "Combined",
                    "corpus_size": n_docs,
                    "noise_label": label, "noise_min": noise_min, "noise_max": noise_max,
                    "noise_range_pct": round((noise_max - 1.0) * 100, 1),
                    "top_k": top_k, "query": qm["query"],
                    "recall_at_k":    round(recall_at_k(gt["doc_ids"],
                                                        qm["enc_doc_ids"], top_k), 4),
                    "ndcg_at_k":      round(ndcg_at_k(gt["doc_ids"], gt["scores"],
                                                       qm["enc_doc_ids"], top_k), 4),
                    "score_entropy":  round(score_entropy(qm["enc_scores"]), 4),
                    "score_variance": round(score_variance(qm["enc_scores"]), 6),
                    "enc_total_us":   qm["enc_total_us"],
                })
            append_rows(out, rows)
            mean_r = sum(r["recall_at_k"] for r in rows) / len(rows)
            print(f"    n_docs={n_docs} {label:4s}  recall={mean_r:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

EXP_MAP = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "F": exp_F,
           "G": exp_G, "H": exp_H, "I": exp_I}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="A")
    parser.add_argument("--dataset", default="all",
                        help="Dataset to run against: nfcorpus, nq, medical, or all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--run-id", default=None,
                        help="Run identifier appended as a subdirectory under results/<dataset>/. "
                             "Defaults to a UTC timestamp (YYYYMMDD_HHMMSS). Pass an explicit "
                             "value to group related experiments (e.g. 'full_3633_v2').")
    parser.add_argument("--only-algos", default=None,
                        help="Comma-separated algorithm names to run (e.g. 'Diehard,DiehardCombined'). "
                             "Appends rows to existing CSVs without deleting them. "
                             "Results are written to the flat results/<dataset>/ directory.")
    args = parser.parse_args()

    import datetime
    only_algos = set(args.only_algos.split(",")) if args.only_algos else None
    augment = only_algos is not None
    run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cfg      = yaml.safe_load(Path(args.config).read_text())
    api_base = cfg["api_base"]
    n_queries = cfg["trials_per_config"]

    if not args.dry_run and not health_check(api_base):
        raise SystemExit(f"Backend not reachable at {api_base}. Run ./start_demo.sh first.")

    to_run = list(EXP_MAP.keys()) if args.exp == "all" else [args.exp.upper()]

    # Resolve which datasets to iterate over
    if args.dataset == "all":
        ds_tags = available_datasets()
    else:
        ds_tags = [d.strip() for d in args.dataset.split(",")]

    for ds_tag in ds_tags:
        ds_label = DATASETS.get(ds_tag, {}).get("label", ds_tag)
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_label}  ({ds_tag})")
        print(f"{'='*60}")

        if ds_tag == "medical":
            queries = MEDICAL_QUERIES[:n_queries]
        else:
            queries = load_dataset_queries(ds_tag, n_queries)

        if augment:
            # Augment mode: write directly to the flat dataset directory so rows
            # are appended to the existing CSVs (no new run_id subdir, no symlink update).
            results_dir = RESULTS_DIR / ds_tag
        else:
            results_dir = RESULTS_DIR / ds_tag / run_id
            results_dir.mkdir(parents=True, exist_ok=True)
            # Write a symlink "latest" → this run so plot_results.py and callers
            # that don't pass --run-id always find the most recent data.
            latest = RESULTS_DIR / ds_tag / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_id)

        for exp_id in to_run:
            if exp_id not in EXP_MAP:
                print(f"Unknown experiment: {exp_id}"); continue
            EXP_MAP[exp_id](cfg, api_base, dry_run=args.dry_run,
                            results_dir=results_dir, queries=queries,
                            only_algos=only_algos)

    print("\nDone. Results in experiments/results/<dataset>/")


if __name__ == "__main__":
    main()
