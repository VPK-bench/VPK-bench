#!/usr/bin/env python3
"""
Download NFCorpus and upload it to the VPK backend.

NFCorpus: 3,633 PubMed medical abstracts with 323 queries and graded
relevance judgments. Ideal for NeurIPS evaluation — same medical domain
as the synthetic knowledge base, and publicly reproducible.

Usage:
    python experiments/load_nfcorpus.py [--api http://127.0.0.1:3000] [--limit N]
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests


def download_nfcorpus(cache_dir: Path):
    """Download NFCorpus via HuggingFace datasets and cache locally."""
    corpus_path = cache_dir / "nfcorpus_corpus.json"
    queries_path = cache_dir / "nfcorpus_queries.json"
    qrels_path = cache_dir / "nfcorpus_qrels.json"

    if corpus_path.exists() and queries_path.exists() and qrels_path.exists():
        print("[nfcorpus] Using cached data from", cache_dir)
        return (
            json.loads(corpus_path.read_text()),
            json.loads(queries_path.read_text()),
            json.loads(qrels_path.read_text()),
        )

    print("[nfcorpus] Downloading from HuggingFace (BeIR/nfcorpus) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please install: pip install datasets")

    ds = load_dataset("BeIR/nfcorpus", "corpus")
    corpus = [
        {"id": row["_id"], "text": row["title"] + " " + row["text"]}
        for row in ds["corpus"]
        if row["text"].strip()
    ]

    qs = load_dataset("BeIR/nfcorpus", "queries")
    queries = [
        {"id": row["_id"], "text": row["text"]}
        for row in qs["queries"]
        if row["text"].strip()
    ]

    # Relevance judgments — stored as a separate dataset in newer BeIR format
    qrels: dict[str, dict[str, int]] = {}
    try:
        from datasets import load_dataset as _ld
        qrel_ds = _ld("BeIR/nfcorpus-qrels")
        split = "test" if "test" in qrel_ds else list(qrel_ds.keys())[0]
        for row in qrel_ds[split]:
            q_id = str(row.get("query-id", row.get("qid", "")))
            d_id = str(row.get("corpus-id", row.get("pid", "")))
            score = int(row.get("score", row.get("label", 1)))
            qrels.setdefault(q_id, {})[d_id] = score
        print(f"[nfcorpus] Loaded {len(qrels)} qrel sets")
    except Exception as e:
        print(f"[nfcorpus] Warning: could not load qrels ({e}); continuing without them")

    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus))
    queries_path.write_text(json.dumps(queries))
    qrels_path.write_text(json.dumps(qrels))
    print(f"[nfcorpus] Saved {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrel sets")
    return corpus, queries, qrels


def upload_corpus(api_base: str, corpus: list[dict], batch_size: int = 50) -> dict[str, int]:
    """Upload corpus to VPK. Returns mapping {nfcorpus_id -> vpk_doc_id}.

    Supports incremental loading: if an id_map already exists, only uploads
    docs whose IDs are not yet in the map.  This lets you call the script
    repeatedly with increasing --limit values for a corpus-size sweep.
    """
    id_map_path = Path("experiments/results/nfcorpus_id_map.json")
    id_map: dict[str, int] = {}
    if id_map_path.exists():
        id_map = json.loads(id_map_path.read_text())

    new_docs = [doc for doc in corpus if doc["id"] not in id_map]
    if not new_docs:
        print(f"[nfcorpus] All {len(corpus)} docs already uploaded (id_map has {len(id_map)} entries)")
        return id_map

    print(f"[nfcorpus] Uploading {len(new_docs)} new docs ({len(id_map)} already in map) ...")

    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i : i + batch_size]
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

        progress = min(i + batch_size, len(new_docs))
        print(f"  {progress}/{len(new_docs)} uploaded, errors: {len(data.get('errors', []))}")
        time.sleep(0.2)

    id_map_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path.write_text(json.dumps(id_map))
    print(f"[nfcorpus] Upload complete. ID map now has {len(id_map)} entries → {id_map_path}")
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
                "relevant": relevant,  # {vpk_doc_id: relevance_score}
            }

    out = Path("experiments/results/nfcorpus_ground_truth.json")
    out.write_text(json.dumps(gt, indent=2))
    print(f"[nfcorpus] Ground truth saved: {len(gt)} queries → {out}")
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:3000")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size (for testing)")
    parser.add_argument("--cache", default="experiments/.nfcorpus_cache")
    args = parser.parse_args()

    corpus, queries, qrels = download_nfcorpus(Path(args.cache))

    if args.limit:
        corpus = corpus[: args.limit]
        print(f"[nfcorpus] Limited to {len(corpus)} docs")

    id_map = upload_corpus(args.api, corpus)
    save_ground_truth(queries, qrels, id_map)
    print("[nfcorpus] Done.")


if __name__ == "__main__":
    main()
