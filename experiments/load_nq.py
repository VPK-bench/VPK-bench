#!/usr/bin/env python3
"""
Download a Natural Questions (NQ) subset and upload it to the VPK backend.

Loads from a local BeIR-format NQ directory (corpus.jsonl + queries.jsonl +
qrels/test.tsv) if available, otherwise downloads from HuggingFace BeIR/nq.

Default subset size matches NFCorpus: 3,633 docs and 323 queries, so
results are directly comparable across the two benchmarks.

Usage:
    python experiments/load_nq.py [--api http://127.0.0.1:3000] [--limit N]
    python experiments/load_nq.py --local /path/to/nq   # use local BeIR copy
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import requests

NUM_DOCS = 3633
NUM_QUERIES = 323
SEED = 42


# ── Load from local BeIR-format directory ────────────────────────────────────

def load_local_nq(nq_dir: Path, num_docs: int, num_queries: int):
    """Read corpus.jsonl / queries.jsonl / qrels/test.tsv from a local dir."""
    corpus_file = nq_dir / "corpus.jsonl"
    queries_file = nq_dir / "queries.jsonl"
    qrels_file = nq_dir / "qrels" / "test.tsv"

    if not corpus_file.exists():
        return None

    print(f"[nq] Loading local data from {nq_dir} ...")

    # Qrels — need these first so we can prefer docs/queries that have judgments
    qrels: dict[str, dict[str, int]] = {}
    if qrels_file.exists():
        with open(qrels_file) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                q_id = row["query-id"]
                d_id = row["corpus-id"]
                score = int(row["score"])
                qrels.setdefault(q_id, {})[d_id] = score
        print(f"[nq]   {len(qrels)} query relevance sets loaded")

    # Collect doc/query IDs that appear in qrels
    judged_doc_ids = {d for rels in qrels.values() for d in rels}
    judged_query_ids = set(qrels.keys())

    # Corpus — prioritize judged docs, then fill with random ones
    rng = random.Random(SEED)
    all_docs = []
    with open(corpus_file) as f:
        for line in f:
            all_docs.append(json.loads(line))

    judged_docs = [d for d in all_docs if d["_id"] in judged_doc_ids]
    other_docs = [d for d in all_docs if d["_id"] not in judged_doc_ids]
    rng.shuffle(other_docs)

    corpus = judged_docs[:num_docs]
    if len(corpus) < num_docs:
        corpus.extend(other_docs[: num_docs - len(corpus)])
    corpus = [
        {"id": d["_id"], "text": (d.get("title") or "") + " " + (d.get("text") or "")}
        for d in corpus
        if (d.get("text") or "").strip()
    ]

    # Queries — prioritize ones with judgments
    all_queries = []
    with open(queries_file) as f:
        for line in f:
            all_queries.append(json.loads(line))

    judged_qs = [q for q in all_queries if q["_id"] in judged_query_ids]
    other_qs = [q for q in all_queries if q["_id"] not in judged_query_ids]
    rng.shuffle(other_qs)

    queries_raw = judged_qs[:num_queries]
    if len(queries_raw) < num_queries:
        queries_raw.extend(other_qs[: num_queries - len(queries_raw)])
    queries = [{"id": q["_id"], "text": q["text"]} for q in queries_raw if q["text"].strip()]

    print(f"[nq]   {len(corpus)} docs, {len(queries)} queries selected")
    return corpus, queries, qrels


# ── Download from HuggingFace ────────────────────────────────────────────────

def download_nq(cache_dir: Path, num_docs: int, num_queries: int):
    """Download NQ via HuggingFace datasets and cache locally."""
    corpus_path = cache_dir / "nq_corpus.json"
    queries_path = cache_dir / "nq_queries.json"
    qrels_path = cache_dir / "nq_qrels.json"

    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        print("[nq] Using cached data from", cache_dir)
        return (
            json.loads(corpus_path.read_text()),
            json.loads(queries_path.read_text()),
            json.loads(qrels_path.read_text()),
        )

    print("[nq] Downloading from HuggingFace (BeIR/nq) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please install: pip install datasets")

    ds = load_dataset("BeIR/nq", "corpus")
    all_corpus = [
        {"id": row["_id"], "text": row["title"] + " " + row["text"]}
        for row in ds["corpus"]
        if row["text"].strip()
    ]

    qs = load_dataset("BeIR/nq", "queries")
    all_queries = [
        {"id": row["_id"], "text": row["text"]}
        for row in qs["queries"]
        if row["text"].strip()
    ]

    # Relevance judgments
    qrels: dict[str, dict[str, int]] = {}
    try:
        qrel_ds = load_dataset("BeIR/nq-qrels")
        split = "test" if "test" in qrel_ds else list(qrel_ds.keys())[0]
        for row in qrel_ds[split]:
            q_id = str(row.get("query-id", row.get("qid", "")))
            d_id = str(row.get("corpus-id", row.get("pid", "")))
            score = int(row.get("score", row.get("label", 1)))
            qrels.setdefault(q_id, {})[d_id] = score
        print(f"[nq] Loaded {len(qrels)} qrel sets")
    except Exception as e:
        print(f"[nq] Warning: could not load qrels ({e}); continuing without them")

    # Subset to match NFCorpus size, preferring docs/queries with judgments
    rng = random.Random(SEED)
    judged_doc_ids = {d for rels in qrels.values() for d in rels}
    judged_query_ids = set(qrels.keys())

    judged_docs = [d for d in all_corpus if d["id"] in judged_doc_ids]
    other_docs = [d for d in all_corpus if d["id"] not in judged_doc_ids]
    rng.shuffle(other_docs)
    corpus = judged_docs[:num_docs]
    if len(corpus) < num_docs:
        corpus.extend(other_docs[: num_docs - len(corpus)])

    judged_qs = [q for q in all_queries if q["id"] in judged_query_ids]
    other_qs = [q for q in all_queries if q["id"] not in judged_query_ids]
    rng.shuffle(other_qs)
    queries = judged_qs[:num_queries]
    if len(queries) < num_queries:
        queries.extend(other_qs[: num_queries - len(queries)])

    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus))
    queries_path.write_text(json.dumps(queries))
    qrels_path.write_text(json.dumps(qrels))
    print(f"[nq] Saved {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrel sets")
    return corpus, queries, qrels


# ── Upload / ground-truth (shared with load_nfcorpus.py) ─────────────────────

def upload_corpus(api_base: str, corpus: list[dict], batch_size: int = 50) -> dict[str, int]:
    """Upload corpus to VPK. Returns mapping {nq_id -> vpk_doc_id}."""
    id_map_path = Path("experiments/results/nq_id_map.json")
    if id_map_path.exists():
        print("[nq] Using existing ID map —", id_map_path)
        return json.loads(id_map_path.read_text())

    print(f"[nq] Uploading {len(corpus)} docs in batches of {batch_size} ...")
    id_map = {}

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        texts = [doc["text"][:1000] for doc in batch]
        resp = requests.post(
            f"{api_base}/api/upload",
            json={"documents": texts},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        inserted_ids = data.get("visualization", {}).get("doc_ids", [])
        for j, doc in enumerate(batch):
            if j < len(inserted_ids):
                id_map[doc["id"]] = inserted_ids[j]

        progress = min(i + batch_size, len(corpus))
        print(f"  {progress}/{len(corpus)} uploaded, errors: {len(data.get('errors', []))}")
        time.sleep(0.2)

    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.write_text(json.dumps(id_map))
    print(f"[nq] Upload complete. ID map saved to {id_map_path}")
    return id_map


def save_ground_truth(queries: list[dict], qrels: dict, id_map: dict[str, int]):
    """Save query ground truth (VPK doc IDs) for use in experiments."""
    gt = {}
    for q in queries:
        q_id = q["id"]
        if q_id not in qrels:
            continue
        relevant = {
            id_map[d_id]: score
            for d_id, score in qrels[q_id].items()
            if d_id in id_map
        }
        if relevant:
            gt[q_id] = {
                "text": q["text"],
                "relevant": relevant,
            }

    out = Path("experiments/results/nq_ground_truth.json")
    out.write_text(json.dumps(gt, indent=2))
    print(f"[nq] Ground truth saved: {len(gt)} queries → {out}")
    return gt


def main():
    parser = argparse.ArgumentParser(description="Load NQ subset into VPK")
    parser.add_argument("--api", default="http://127.0.0.1:3000")
    parser.add_argument("--limit", type=int, default=None, help="Override corpus size limit")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--cache", default="experiments/.nq_cache")
    parser.add_argument(
        "--local", default=None,
        help="Path to local BeIR-format NQ directory (corpus.jsonl + queries.jsonl + qrels/)",
    )
    args = parser.parse_args()

    num_docs = args.limit or NUM_DOCS
    num_queries = args.num_queries

    # Try local copy first if provided
    result = None
    if args.local:
        result = load_local_nq(Path(args.local), num_docs, num_queries)
        if result is None:
            print(f"[nq] Local path {args.local} not found, falling back to HuggingFace")

    if result is None:
        # Try the sibling workshop repo location
        workshop_nq = Path(__file__).resolve().parent.parent.parent / \
            "homomorphic-encryption-mpi-workshop" / "Workshop-Code" / \
            "jupyter_notebooks" / "datasets" / "nq"
        if workshop_nq.exists():
            print(f"[nq] Found workshop NQ data at {workshop_nq}")
            result = load_local_nq(workshop_nq, num_docs, num_queries)

    if result is None:
        result = download_nq(Path(args.cache), num_docs, num_queries)

    corpus, queries, qrels = result
    id_map = upload_corpus(args.api, corpus)
    save_ground_truth(queries, qrels, id_map)
    print("[nq] Done.")


if __name__ == "__main__":
    main()
