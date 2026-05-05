#!/usr/bin/env python3
"""
Embed an NQ subset with all-MiniLM-L6-v2 and save as a pickle for offline
benchmarking (no VPK backend needed).

Matches NFCorpus scale: 3,633 docs × 323 queries × 384-dim embeddings.

Sources (tried in order):
  1. --local path to BeIR-format NQ directory
  2. Sibling workshop repo at ../homomorphic-encryption-mpi-workshop/
  3. HuggingFace BeIR/nq download

Usage:
    python experiments/embed_nq_subset.py
    python experiments/embed_nq_subset.py --local /path/to/nq
    python experiments/embed_nq_subset.py --num-docs 500 --num-queries 50  # smaller for testing
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np

SEED = 42
NUM_DOCS = 3633
NUM_QUERIES = 323
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUT = os.path.join(os.path.dirname(__file__), "results", "nq_subset.pkl")


def load_jsonl(path, n=None):
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
            if n and len(items) >= n:
                break
    return items


def load_nq_texts(nq_dir: Path, num_docs: int, num_queries: int):
    """Load raw texts from a local BeIR-format NQ directory."""
    corpus_raw = load_jsonl(nq_dir / "corpus.jsonl")
    queries_raw = load_jsonl(nq_dir / "queries.jsonl")

    rng = random.Random(SEED)

    # If we have more docs than needed, randomly subsample
    if len(corpus_raw) > num_docs:
        rng.shuffle(corpus_raw)
        corpus_raw = corpus_raw[:num_docs]

    if len(queries_raw) > num_queries:
        rng.shuffle(queries_raw)
        queries_raw = queries_raw[:num_queries]

    doc_ids = [d["_id"] for d in corpus_raw]
    doc_texts = [(d.get("title") or "") + " " + (d.get("text") or "") for d in corpus_raw]
    query_ids = [q["_id"] for q in queries_raw]
    query_texts = [q["text"] for q in queries_raw]

    return doc_ids, doc_texts, query_ids, query_texts


def find_nq_dir(local_override: str | None) -> Path | None:
    """Try to find a local NQ directory."""
    if local_override:
        p = Path(local_override)
        if (p / "corpus.jsonl").exists():
            return p

    workshop_nq = (
        Path(__file__).resolve().parent.parent.parent
        / "homomorphic-encryption-mpi-workshop"
        / "Workshop-Code"
        / "jupyter_notebooks"
        / "datasets"
        / "nq"
    )
    if (workshop_nq / "corpus.jsonl").exists():
        return workshop_nq

    return None


def download_and_extract(num_docs: int, num_queries: int):
    """Fall back to HuggingFace if no local copy."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("No local NQ data found. Install 'datasets' to download: pip install datasets")

    print("[nq] Downloading from HuggingFace (BeIR/nq) ...")
    ds = load_dataset("BeIR/nq", "corpus")
    qs = load_dataset("BeIR/nq", "queries")

    rng = random.Random(SEED)

    corpus_raw = [row for row in ds["corpus"] if row["text"].strip()]
    if len(corpus_raw) > num_docs:
        rng.shuffle(corpus_raw)
        corpus_raw = corpus_raw[:num_docs]

    queries_raw = [row for row in qs["queries"] if row["text"].strip()]
    if len(queries_raw) > num_queries:
        rng.shuffle(queries_raw)
        queries_raw = queries_raw[:num_queries]

    doc_ids = [d["_id"] for d in corpus_raw]
    doc_texts = [d["title"] + " " + d["text"] for d in corpus_raw]
    query_ids = [q["_id"] for q in queries_raw]
    query_texts = [q["text"] for q in queries_raw]

    return doc_ids, doc_texts, query_ids, query_texts


def main():
    parser = argparse.ArgumentParser(description="Embed NQ subset for offline benchmarking")
    parser.add_argument("--local", default=None, help="Path to local BeIR-format NQ directory")
    parser.add_argument("--num-docs", type=int, default=NUM_DOCS)
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    nq_dir = find_nq_dir(args.local)
    if nq_dir:
        print(f"[nq] Loading from {nq_dir}")
        doc_ids, doc_texts, query_ids, query_texts = load_nq_texts(
            nq_dir, args.num_docs, args.num_queries
        )
    else:
        doc_ids, doc_texts, query_ids, query_texts = download_and_extract(
            args.num_docs, args.num_queries
        )

    print(f"[nq] {len(doc_texts)} docs, {len(query_texts)} queries")

    from sentence_transformers import SentenceTransformer

    print(f"[nq] Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("[nq] Encoding documents ...")
    doc_emb = model.encode(doc_texts, show_progress_bar=True, normalize_embeddings=False)
    print("[nq] Encoding queries ...")
    query_emb = model.encode(query_texts, show_progress_bar=True, normalize_embeddings=False)

    print(f"[nq] doc_emb: {doc_emb.shape}, query_emb: {query_emb.shape}")

    result = {
        "doc_ids": doc_ids,
        "doc_texts": doc_texts,
        "doc_emb": doc_emb,
        "query_ids": query_ids,
        "query_texts": query_texts,
        "query_emb": query_emb,
        "model": MODEL_NAME,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[nq] Saved to {out_path}")


if __name__ == "__main__":
    main()
