# VPKbench Experiment Documentation

**Suite:** Privacy-Preserving Vector Search — PHE Algorithm Benchmarks  
**Paper:** Under review, NeurIPS 2026  
**Run ID (current):** 20260509_153345  
**Corpus:** NFCorpus (3,633 docs) + Natural Questions  
**Trials:** N = 10 independent key-generation repetitions  
**Queries per configuration:** 30  

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Encryption Algorithms](#2-encryption-algorithms)
3. [Datasets](#3-datasets)
4. [Metrics](#4-metrics)
5. [Experiment Designs](#5-experiment-designs)
6. [Configuration Reference](#6-configuration-reference)
7. [Execution Pipeline](#7-execution-pipeline)
8. [Results Interpretation](#8-results-interpretation)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Known Issues and Corrections](#10-known-issues-and-corrections)
11. [Reproducing Results](#11-reproducing-results)

---

## 1. System Architecture

VPKbench benchmarks a **Virtual Private Knowledge (VPK)** server — a privacy-preserving vector search system with the following components:

```
Bob (client)
    │  uploads plaintext docs / submits encrypted queries
    ▼
VPK Orchestrator (Rust / Axum, port 3000)
    │  embeds → encrypts → distributes
    ├── Eve-1 (in-memory shard, 1/3 of corpus)
    ├── Eve-2 (in-memory shard, 1/3 of corpus)
    └── Eve-3 (in-memory shard, 1/3 of corpus)
         │  search fan-out → merge → unscramble doc IDs
         ▼
    VPK returns ranked doc IDs to Bob
         ▼
Embedding server (Python/Flask, port 5001)
    all-MiniLM-L6-v2, dimension 384
```

**Key design principle:** Eve shards see only encrypted vectors and similarity scores — never plaintext. The VPK orchestrator holds the encryption keys and resolves scrambled doc IDs back to real IDs before returning results.

**Document lifecycle:**
1. Bob uploads text → VPK embeds (384-dim, L2-normalized) → encrypts → distributes across 3 shards, storing the scrambled vector ID mapping internally.
2. On query: Bob's text is embedded → encrypted → sent to all 3 shards in parallel → top-K results from each shard are merged → scrambled IDs resolved → ranked doc IDs returned.
3. Eve shards operate on encrypted vectors only; they cannot recover plaintext embeddings.

**Infrastructure:**
- Backend: Rust 1.75+, Axum 0.7, Tokio async runtime
- Vector storage: in-memory (default) or PostgreSQL + pgvector
- Embedding: `all-MiniLM-L6-v2` via HuggingFace sentence-transformers, served locally
- Homomorphic encryption: Microsoft SEAL via `sealy` 0.2 Rust bindings (CKKS path only)

---

## 2. Encryption Algorithms

All algorithms receive unit-normalized 384-dimensional embeddings from `all-MiniLM-L6-v2`. This normalization is a prerequisite for correctness of DS, ROME, and DIEHARD.

### 2.1 Dimensional Scrambling (DS)

**Type:** Permutation cipher  
**Similarity metric:** Cosine  
**Complexity:** O(n) encrypt, O(n) search  
**Dimension change:** None (384 → 384)

**Construction:**  
A uniform random permutation π ∈ S_n is drawn as the secret key. Encryption is:

```
E(v)[i] = v[π(i)]
```

**Homomorphism proof:**  
For unit vectors u, v with permutation π:
```
cos(E(u), E(v)) = Σ_i E(u)[i]·E(v)[i]
                = Σ_i u[π(i)]·v[π(i)]
                = Σ_j u[j]·v[j]      (reindex j = π(i))
                = cos(u, v)           ✓ exact
```

**Security:** The permutation is the sole secret. Without π, an adversary observing encrypted vectors cannot recover the ordering of dimensions. However, the encrypted vector lies on the same unit hypersphere as the plaintext — a passive observer knows ‖E(v)‖ = ‖v‖ = 1 and can compute pairwise distances between encrypted vectors.

---

### 2.2 Noise Injection (NI)

**Type:** Element-wise multiplicative perturbation  
**Similarity metric:** Cosine  
**Complexity:** O(n)  
**Dimension change:** None (384 → 384)

**Construction:**  
Secret noise vector n ∈ ℝⁿ with elements drawn uniformly from [noise_min, noise_max]:

```
E_doc(d)   = normalize(d ⊙ n)
E_query(q) = normalize(q ⊙ (1/n))
```

**Approximate homomorphism:**  
```
cos(E_query(q), E_doc(d)) ≈ cos(q, d)
```
The approximation is exact when all n_i = c (constant noise) — which collapses to a norm-preserving identity. At noise_min = noise_max = 1.0, it is the identity. The recall degradation grows with the variance of noise multipliers.

**Security:** The noise vector n is the secret key. Unlike DS, this scheme perturbs the embedding values, not just their ordering. Score entropy increases monotonically with noise range. Combined with DS (see §2.3), the two secrets are independent.

---

### 2.3 DS + NI (Combined)

**Type:** Composition of DS then NI  
**Similarity metric:** Cosine  
**Complexity:** O(n)  

```
E(v) = NI(DS(v)) = normalize(permute(v, π) ⊙ n)
```

The permutation and noise vector are drawn independently on each key-generation event. This provides two independent obfuscation layers: an adversary must defeat both to recover the original embedding structure.

---

### 2.4 ROME (Random Orthogonal Matrix Encryption)

**Type:** Orthogonal matrix transformation  
**Similarity metric:** Cosine  
**Complexity:** O(m²) where m = ⌈1.5 × 384⌉ = 576  
**Dimension change:** 384 → 576 (1.5× padding)

**Construction:**  
A random orthogonal matrix Q ∈ ℝᵐˣᵐ is the key (generated via QR decomposition of a random Gaussian matrix). Documents are zero-padded to dimension m:

```
pad(v) = [v₁, ..., v₃₈₄, 0, ..., 0] ∈ ℝ⁵⁷⁶
E(v)   = Q · pad(v)
```

**Exact homomorphism:**
```
⟨E(u), E(v)⟩ = (Q·pad(u))ᵀ(Q·pad(v))
              = pad(u)ᵀ Qᵀ Q pad(v)
              = pad(u)ᵀ pad(v)        (Q orthogonal: QᵀQ = I)
              = ⟨u, v⟩                ✓ exact
```
Since Q also preserves norms (‖Q·x‖ = ‖x‖), cosine similarity is preserved exactly.

**Security:** The random orthogonal key Q acts as a rotation/reflection of the padded vector space. An adversary sees Q·pad(v) ∈ ℝ⁵⁷⁶; without Q, recovering pad(v) requires solving a random linear system with 576 unknowns and at most 384 non-zero degrees of freedom.

**Key cost:** Q is a 576×576 matrix (≈ 2.6 MB per key). Encryption requires a dense matrix-vector multiply — O(m²) vs O(n) for DS/NI.

---

### 2.5 ROME + NI (RomeCombined)

ROME encryption followed by noise injection on the expanded 576-dimensional encrypted vector:

```
E(v) = normalize(ROME(v) ⊙ n)
```

The noise is applied in the expanded space. Because cosine similarity is scale-invariant, the noise injection provides additional score obfuscation without destroying the inner-product structure — but introduces the same approximate recall degradation as standalone NI.

---

### 2.6 DIEHARD (Dual Independent Encryption for Hardened Asymmetric Retrieval Defense)

**Type:** Asymmetric dual-matrix linear map  
**Similarity metric:** **Dot product** (not cosine — see §10.1)  
**Complexity:** O(m·n) encrypt  
**Dimension change:** 384 → 576

**Construction:**  
Two distinct matrices A, B ∈ ℝᵐˣⁿ are generated such that AᵀB = Iₙ:

1. Draw a random m×n Gaussian matrix; QR-decompose to get A with orthonormal columns.
2. Set B = A + P_⊥ · B̂ where P_⊥ = Iₘ − AA⁺ projects onto the null space of Aᵀ. This ensures AᵀB = AᵀA = Iₙ while B ≠ A.

```
E_query(q) = A · q    (A ∈ ℝ⁵⁷⁶ˣ³⁸⁴)
E_doc(d)   = B · d    (B ∈ ℝ⁵⁷⁶ˣ³⁸⁴)
```

**Exact inner-product homomorphism:**
```
⟨E_query(q), E_doc(d)⟩ = (Aq)ᵀ(Bd) = qᵀ AᵀB d = qᵀ d = ⟨q, d⟩   ✓ exact
```

**Why dot product, not cosine:**  
DIEHARD's dimensional expansion changes document norms: ‖Bd‖ ≠ ‖d‖ in general (since B is not orthogonal). Cosine similarity computes ⟨Aq, Bd⟩ / (‖Aq‖·‖Bd‖), which does NOT equal cos(q, d) — this is the root cause of the metric bug described in §10.1. Since `all-MiniLM-L6-v2` produces unit-normalized embeddings (‖q‖ = ‖d‖ = 1), ranking by ⟨q, d⟩ is equivalent to ranking by cos(q, d), so dot-product search is correct here.

**Asymmetric security advantage over ROME:**  
In ROME, query and document keys are the same matrix Q — a compromised document key immediately compromises queries. In DIEHARD, A ≠ B, so document-key compromise does not reveal A, and vice versa. This provides stronger forward secrecy for the query key.

---

### 2.7 DIEHARD + NI (DiehardCombined)

DIEHARD encryption followed by noise injection on the expanded vectors. **Note:** this combination has a known recall degradation (~0.845) that is not caused by the metric fix — it results from unpaired noise multipliers (see §10.2).

---

### 2.8 CKKS

**Type:** Fully homomorphic encryption (FHE)  
**Similarity metric:** Homomorphic dot product  
**Complexity:** O(n log n) encrypt; O(|corpus| × n log n) search  
**Dimension change:** 384 → 4096 CKKS slots (ciphertext)

**SEAL parameters:**

| Parameter           | Value                    |
|---------------------|--------------------------|
| Scheme              | CKKS                     |
| Poly modulus degree | 8192                     |
| Coeff modulus bits  | [60, 40, 40, 60]         |
| Scale               | 2⁴⁰                      |
| Security level      | TC128                    |
| Slot count          | 4096 = 8192/2            |

**Operation:**  
Query vector q and document vector d are encrypted as CKKS ciphertexts ct_q and ct_d. Similarity is computed entirely in the encrypted domain:

```
score = Dec(ct_q ⊙ ct_d)     (element-wise multiply then sum slots)
```

The CKKS path bypasses the standard vector-DB shard search and instead iterates over stored ciphertexts, performing a homomorphic inner product for each document.

**Correctness fix (§10.2):** The prior implementation used a log(N) homomorphic rotate-and-add loop (where N = poly_modulus_degree) to sum CKKS slots — accumulated noise at scale ≈ 2⁸⁰ caused rank corruption. The corrected implementation decrypts to plaintext first, then sums in floating point. This restores Recall@10 = 1.000 at zero noise.

**Performance:** ~69 ms per query encryption; ~36.8 s per encrypted search over 3,633 documents. CKKS dominates total experiment runtime.

---

### 2.9 CKKS + NI (CkksCombined)

Noise injection applied to the plaintext embedding before CKKS encryption:

```
ct = CKKS_encrypt(normalize(v ⊙ n))
```

The noise perturbs the plaintext embedding; the FHE layer then encrypts the perturbed vector. Search computes the homomorphic dot product of perturbed query and perturbed document ciphertexts.

---

### Algorithm Summary Table

| Name | Short | Encrypt key | Metric | Dim out | O(encrypt) | Exact? |
|------|-------|-------------|--------|---------|-----------|--------|
| Scrambling | DS | permutation π | Cosine | 384 | O(n) | ✓ |
| Noise | NI | noise vec n | Cosine | 384 | O(n) | ≈ |
| Combined | DS+NI | π, n | Cosine | 384 | O(n) | ≈ |
| Rome | ROME | Q ∈ O(576) | Cosine | 576 | O(m²) | ✓ |
| RomeCombined | ROME+NI | Q, n | Cosine | 576 | O(m²) | ≈ |
| Diehard | DIEHARD | A, B (asymmetric) | Dot product | 576 | O(mn) | ✓ |
| DiehardCombined | DIEHARD+NI | A, B, n | Dot product | 576 | O(mn) | ≈* |
| Ckks | CKKS | FHE keypair | Homomorphic | 4096 slots | O(n log n) | ✓ |
| CkksCombined | CKKS+NI | FHE keypair, n | Homomorphic | 4096 slots | O(n log n) | ≈ |

*DIEHARD+NI recall ~0.845 due to unpaired noise interaction (see §10.2).

---

## 3. Datasets

### 3.1 NFCorpus

- **Source:** BeIR/nfcorpus (HuggingFace), originally from PubMed
- **Domain:** Biomedical / clinical medicine
- **Corpus size:** 3,633 documents (PubMed abstracts with titles)
- **Queries:** 323 medical queries (test split)
- **Relevance judgments:** Graded (0–2), multi-level qrels
- **Used for experiments:** A, D, E, G, H, I
- **Cache location:** `experiments/.nfcorpus_cache/`

Each document is truncated to 1,000 characters before embedding (to fit within the embedding model's context window). The embedding model produces 384-dimensional unit-normalized vectors.

### 3.2 Natural Questions (NQ)

- **Source:** BeIR NQ subset
- **Domain:** Open-domain QA (Google Search questions)
- **Corpus size:** variable (see load script)
- **Queries:** 323 queries (first 323 from test split)
- **Used for experiments:** A, D, E
- **Cache location:** `experiments/.nq_cache/`

### 3.3 Query Selection

Each experiment configuration uses the first 30 queries from the dataset's query cache (`trials_per_config: 30` in `config.yaml`). This provides:
- 30 queries × 10 trials = 300 individual query-level observations per configuration
- Two-level aggregation: per-trial mean (30 queries → 1 scalar) then cross-trial statistics

---

## 4. Metrics

### 4.1 Recall@K

Fraction of ground-truth top-K documents that appear in the encrypted search's top-K results:

```
Recall@K = |{enc_top_K} ∩ {gt_top_K}| / K
```

**Ground truth:** Scrambling baseline results (fresh keys, no noise). This measures how well each PHE algorithm preserves the ranking order relative to exact cosine similarity on the unit-normalized embeddings. A value of 1.000 means perfect rank preservation.

**Why not dataset qrels?** The VPK framework measures preservation of embedding-space rankings, not dataset-level relevance. The dataset qrels are used only for the NQ/NFCorpus ground-truth files — within experiments, the Scrambling baseline serves as the reference ordering.

### 4.2 nDCG@K

Normalized Discounted Cumulative Gain at K, computed against the Scrambling baseline:

```
DCG@K  = Σ_{i=1}^{K} rel(enc[i]) / log₂(i+1)
IDCG@K = Σ_{i=1}^{K} rel(gt[i]) / log₂(i+1)
nDCG@K = DCG@K / IDCG@K
```

where `rel(d)` is the similarity score assigned to document d in the baseline. This weights position-sensitive agreement: returning the correct #1 result contributes more than returning the correct #10 result.

### 4.3 Kendall's τ

Rank correlation between the ground-truth ordering and the encrypted ordering over common documents:

```
τ = (concordant_pairs − discordant_pairs) / (concordant_pairs + discordant_pairs)
```

τ = 1 means identical ranking; τ = 0 means uncorrelated; τ = −1 means reversed. Reported in Exp A CSV but not used as primary metric in figures.

### 4.4 Score Entropy

Shannon entropy of the encrypted similarity score distribution, binned into 20 equal-width bins over [min_score, max_score]:

```
H = −Σ_i (c_i/n) log(c_i/n)
```

This measures how uniformly spread the Eve-visible similarity scores are. Higher entropy means more uniform scores → harder for Eve to distinguish relevant from irrelevant documents. Maximum entropy for n documents with 20 bins is log(20) ≈ 3.0.

**Interpretation:** A system with H near zero leaks near-perfect ranking information (Eve can re-rank documents from scores alone). A system with H near log(20) provides near-maximal score obfuscation.

### 4.5 Score Variance and Distinct Count

- **Score variance:** Var({scores}) — auxiliary measure of score spread
- **Score distinct count:** Number of unique scores (rounded to 4 decimal places) — measures whether the encryption collapses scores to a small discrete set

### 4.6 Latency Metrics

All latencies are measured wall-clock (microseconds) from API response:

| Field | Meaning |
|-------|---------|
| `enc_embed_us` | Time to embed the query text (embedding server round-trip) |
| `enc_encrypt_us` | Time to encrypt the 384-dim embedding vector |
| `enc_search_us` | Time for shard fan-out + merge (dominant for CKKS) |
| `enc_total_us` | End-to-end query latency |

---

## 5. Experiment Designs

### 5.1 Experiment A — Algorithm Comparison (Primary Table)

**Purpose:** Compare all 9 algorithm configurations at a fixed operating point.

**Fixed parameters:**
- Corpus: 3,633 NFCorpus docs (or NQ equivalent)
- Shards: 3
- K: 10
- Noise: minimum noise for each algorithm (DS/NI: ±10%; ROME/DIEHARD/CKKS: ±0%)
- Trials: N = 10 (fresh key generation per trial)

**Procedure (per trial):**
1. Generate fresh Scrambling baseline (new permutation, re-encrypt corpus, collect 30 query results)
2. For each algorithm in order: set_algorithm() → re-encrypt corpus → run 30 queries → compute metrics against baseline
3. Append rows to `exp_A_algorithm_comparison.csv`

**Output CSV fields:** `trial, exp, algorithm, noise_min, noise_max, top_k, query, recall_at_k, ndcg_at_k, kendall_tau, enc_embed_us, enc_encrypt_us, enc_search_us, enc_total_us, score_entropy, score_variance, score_distinct_count`

**Expected runtime:** ~3.5 hours (CKKS dominates: 30 queries × 10 trials × ~36 s/query × 2 CKKS variants ≈ 6 hours; non-CKKS algorithms complete in seconds per trial)

---

### 5.2 Experiment D — Noise–Security Tradeoff

**Purpose:** Map the Pareto frontier between rank preservation (Recall@K) and score obfuscation (score entropy) as noise level varies.

**Noise levels swept:** ±1%, ±5%, ±10%, ±20%, ±30%, ±50%, ±70%

**Algorithms:** Noise, Combined, RomeCombined, DiehardCombined, CkksCombined

**Design:** On each trial, a fresh (DS matrix, noise vector) pair is drawn for each noise level independently. This means the DS matrix changes with each noise level within a trial — confounding DS-initialisation quality and noise level effects. (Exp G isolates this confound.)

**Expected runtime:** ~12 hours

---

### 5.3 Experiment E — Top-K Sensitivity

**Purpose:** Measure how Recall@K and nDCG@K vary with K across all algorithms.

**K values:** 1, 5, 10, 20, 50

**All algorithms:** DS, NI, DS+NI, ROME, ROME+NI, DIEHARD, DIEHARD+NI, CKKS, CKKS+NI

**Design:** Per trial, baseline is collected at K = max(K_values) = 50. All K values then share the same baseline. Each algorithm's noise setting matches Exp A defaults.

**Expected runtime:** ~17 hours

---

### 5.4 Experiment G — Frozen DS Noise Sweep

**Purpose:** Isolate the effect of noise magnitude from DS matrix quality by holding the DS matrix constant across noise levels within each trial.

**Design:**
- First noise level (0%): fresh DS matrix generated
- Subsequent noise levels: DS matrix frozen (`freeze_rotation=True`); only NI reseeded
- Algorithm: Combined only
- Noise levels: 0%, ±1%, ±5%, ±10%, ±20%, ±30%, ±50%, ±70%

**Key finding:** Freezing DS collapses recall variance 5–9× compared to Exp D. The DS-initialization quality hypothesis is false (DS quality = 1.000 ± 0.000 in all Exp H trials); the true source of Exp D variance is the independent joint (DS, NI) draw creating variable rank-disruptiveness configurations.

---

### 5.5 Experiment H — DS Divergence as Covariate

**Purpose:** Measure per-trial DS quality (recall at 0% noise) as a potential covariate for explaining variance in the noise sweep.

**Design:**
1. Set Combined at 0% noise → record recall (DS quality score)
2. Freeze DS matrix, sweep noise levels (same as Exp G)
3. Record DS quality alongside noise-level recall for correlation analysis

**Output:** `ds_quality` column alongside standard metrics, enabling per-trial regression of noise-level recall against DS initialisation quality.

---

### 5.6 Experiment I — Corpus Size × Noise Tradeoff

**Purpose:** Understand how recall degrades with noise as a function of corpus size (more documents = harder to preserve top-K rankings).

**Design:** Run at each corpus size after incrementally loading documents:
- Corpus sizes: 500, 1,000, 2,000, 3,633 docs
- Noise levels: 0%, ±1%, ±10%, ±30%, ±50%, ±70%
- Algorithm: Combined
- Append-only CSV accumulates results across corpus-size steps

**Procedure (full pipeline):** `run_nfcorpus_full.sh` loads docs in steps and runs Exp I at each step before proceeding to A, D, E.

---

## 6. Configuration Reference

### 6.1 `experiments/config.yaml`

```yaml
api_base: "http://127.0.0.1:3000"
trials_per_config: 30        # queries per trial
top_k_default: 10
n_trials: 10                 # independent key-generation repetitions
```

The `n_trials` parameter controls statistical robustness. Each trial:
1. Calls `set_algorithm()` which generates fresh random encryption matrices (permutation, orthogonal matrix, noise vector, or FHE keypair)
2. Calls `/api/reencrypt` which re-encrypts the entire corpus with the new keys
3. Collects `trials_per_config` query measurements

This gives genuine statistical independence: results are not repeated measurements under the same key, but different runs with different random encryption parameters.

### 6.2 Algorithm Parameters

| Algorithm | noise_min | noise_max | padding_ratio | Notes |
|-----------|-----------|-----------|---------------|-------|
| Scrambling | 0.9 | 1.1 | 1.0 | NI inactive (unused by DS) |
| Noise | 0.9 | 1.1 | 1.0 | ±10% element-wise noise |
| Combined | 0.9 | 1.1 | 1.0 | DS + NI, ±10% |
| Rome | 1.0 | 1.0 | 1.5 | No noise; 576-dim output |
| RomeCombined | 0.9 | 1.1 | 1.5 | ROME + NI, ±10% |
| Diehard | 1.0 | 1.0 | 1.5 | No noise; dot-product metric |
| DiehardCombined | 0.9 | 1.1 | 1.5 | DIEHARD + NI, ±10% |
| Ckks | 1.0 | 1.0 | 1.5 | No noise; FHE |
| CkksCombined | 0.9 | 1.1 | 1.5 | CKKS + NI, ±10% |

### 6.3 Backend (`config_1shard.toml` / defaults)

| Parameter | Value |
|-----------|-------|
| `vector_dim` | 384 |
| `padding_dim` | 128 (→ 576 total for ROME/DIEHARD/CKKS) |
| Port | 3000 |
| Shard count | 3 (default_test) |
| CKKS poly_modulus_degree | 8192 |
| CKKS coeff_modulus | [60, 40, 40, 60] bits |
| CKKS scale | 2⁴⁰ |

---

## 7. Execution Pipeline

### 7.1 Prerequisites

```bash
# 1. Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable

# 2. Python dependencies
pip install -r requirements.txt

# 3. Build release binary
cargo build --release
```

### 7.2 Service Startup

```bash
# Terminal A: embedding server (port 5001)
python embedding_server.py &

# Terminal B: VPK backend (port 3000)
./target/release/vpkbench
```

### 7.3 Corpus Loading

```bash
# NFCorpus (full: 3,633 docs, ~5 min)
python experiments/load_nfcorpus.py

# Natural Questions
python experiments/load_nq.py
```

Both scripts support `--limit N` for a reduced corpus (useful for development). They cache the HuggingFace dataset locally and support incremental upload via `id_map`.

### 7.4 Running Experiments

```bash
# Single experiment, single dataset
python experiments/run_experiments.py --exp A --dataset nfcorpus

# Full suite (append-only, both datasets)
python experiments/run_experiments.py --exp all --dataset all

# Specific algorithms only (augment existing CSV, no overwrite)
python experiments/run_experiments.py --exp A --dataset nfcorpus \
    --only-algos Diehard,DiehardCombined

# Named run (writes to experiments/results/<dataset>/<run-id>/)
python experiments/run_experiments.py --exp A --dataset nfcorpus \
    --run-id my_run_20260509
```

Results are written to `experiments/results/<dataset>/<run-id>/` with a `latest` symlink updated to point to the most recent run.

### 7.5 Figure Generation

```bash
python experiments/plot_results.py
# Additional plots:
python experiments/plot_fig3_radar_triptych.py
python experiments/plot_fig9_cosine_error.py
python experiments/plot_fig10_cosine_error.py
```

Figures are written to `experiments/figures/<dataset>/` as both PDF and PNG.

### 7.6 Analysis Notebook

```bash
cd experiments
jupyter notebook analysis.ipynb
```

The notebook performs two-level aggregation (30 queries → trial mean, then 10 trial means → statistics), generates paper-quality figures, and runs Wilcoxon signed-rank tests vs. the DS baseline.

### 7.7 Full Unattended Pipeline

```bash
# NFCorpus only (corpus size sweep → A/D/E/I → figures)
RUN_ID=20260509_full bash experiments/run_nfcorpus_full.sh

# Overnight (E/I NFCorpus → A/D/E/I NQ → figures)
bash experiments/run_overnight.sh
```

---

## 8. Results Interpretation

### 8.1 Latency Tiers

Three performance tiers emerge from Experiment A:

**Tier 1 — O(n) algorithms (DS, NI, DS+NI):**
- Encrypt: < 0.1 ms
- Search: ~5.5 ms over 3,633 docs
- Practical for real-time RAG pipelines

**Tier 2 — Matrix algorithms (ROME, DIEHARD):**
- Encrypt: 5–6 ms (dense 576×384 matrix-vector multiply)
- Search: 4–7 ms (cosine/dot-product over 576-dim vectors)
- DIEHARD is ~44% faster than ROME on search because dot product omits per-document norm computation

**Tier 3 — FHE (CKKS):**
- Encrypt: ~69 ms
- Search: ~36,800 ms over 3,633 docs (~10 ms/doc)
- Suitable only for offline or small-corpus (<100 docs) scenarios

### 8.2 Recall–Security Tradeoff (Exp D)

At ±0% noise, all algorithms except DIEHARD+NI achieve Recall@10 = 1.000. Score entropy is near zero (Eve can rank documents from scores).

As noise increases to ±70%:
- Recall degrades to ~0.932–0.938 for NI, DS+NI, ROME+NI, CKKS+NI
- Score entropy increases to maximum (~log(20) ≈ 3.0)
- DIEHARD+NI recall is flat at ~0.845 across all noise levels (see §10.2)

The key operating point for practical deployment is ±10%–±30% noise, where recall remains above 0.95 and entropy is meaningfully elevated.

### 8.3 Top-K Sensitivity (Exp E)

All algorithms maintain recall > 0.985 at K=10. As K increases to 50, recall improves slightly (more chances for correct hits). The exception is DIEHARD+NI, which is flat near 0.845 regardless of K.

### 8.4 Algorithm Selection Guide

| Use case | Algorithm | Rationale |
|----------|-----------|-----------|
| Maximum throughput, accept weak obfuscation | DS | 0.038 ms encrypt, 5.6 ms search, Recall=1.000 |
| Throughput + score obfuscation | DS+NI | Same latency, Recall≈0.983, meaningful entropy |
| Strong geometric security, perfect fidelity | DIEHARD | 5.1 ms encrypt, 3.9 ms search, Recall=1.000 |
| Strong geometric security + obfuscation | ROME+NI | 6.2 ms encrypt, 7.0 ms search, Recall≈0.987 |
| Information-theoretic security (small corpus) | CKKS | 69 ms encrypt, 36.8 s search, Recall=1.000 |

---

## 9. Statistical Analysis

### 9.1 Aggregation Procedure

All reported statistics use two-level aggregation to separate query-level variation from key-generation-level variation:

1. **Level 1 (within-trial):** Mean over 30 queries → one scalar per trial per configuration
2. **Level 2 (across-trials):** Mean ± standard deviation over N=10 trial-level scalars

This prevents the 30 queries within a trial from inflating the effective sample size. The N=10 trial means are the unit of statistical inference.

### 9.2 Uncertainty Visualization

Figures show ±1σ river bands (shaded regions) around the mean curve. The band alpha is 0.18, allowing overlapping rivers to blend transparently. This differs from error bars: river bands convey the full shape of uncertainty across the x-axis rather than per-point vertical intervals.

### 9.3 Significance Tests

Recall@10 distributions are tested pairwise against the DS baseline using the **Wilcoxon signed-rank test** (non-parametric, appropriate for N=10 paired samples). Each algorithm contributes one per-trial mean recall value; the 10-element vectors are tested against the DS baseline vector.

Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant.

### 9.4 Key Statistical Properties

- DS (Scrambling): always Recall@10 = 1.000 ± 0.000 (exact by construction)
- ROME: always Recall@10 = 1.000 ± 0.000 (exact by construction)
- DIEHARD (no noise): always Recall@10 = 1.000 ± 0.000 (dot-product fix; see §10.1)
- CKKS (no noise): always Recall@10 = 1.000 ± 0.000 (plaintext-sum fix; see §10.2)

Algorithms with noise injection show σ > 0 across trials, reflecting genuine variation in how different noise vectors and permutations interact with specific query-document pairs.

---

## 10. Known Issues and Corrections

### 10.1 DIEHARD Metric Bug (Fixed)

**Original implementation:** DIEHARD-encrypted vectors were stored in Qdrant with cosine similarity, consistent with all other algorithms.

**Problem:** DIEHARD's 384 → 576 dimensional expansion changes document L2 norms: ‖B·d‖ ≠ ‖d‖ in general. Cosine similarity normalises by vector norms, so:

```
cos(Aq, Bd) = ⟨Aq, Bd⟩ / (‖Aq‖ · ‖Bd‖) ≠ cos(q, d)
```

This produced Recall@10 ≈ 0.846 (vs. the theoretical 1.000).

**Fix:** Switch similarity metric to raw dot product for DIEHARD. Since all-MiniLM-L6-v2 produces unit-normalized embeddings:
- ‖q‖ = ‖d‖ = 1
- ⟨q, d⟩ = cos(q, d)

Ranking by ⟨Aq, Bd⟩ = ⟨q, d⟩ is equivalent to ranking by cos(q, d). This restores Recall@10 = 1.000 and reduces search latency from 7.3 ms to 3.9 ms (dot product omits per-document norm computation).

**Residual DIEHARD+NI degradation:** The combined DIEHARD+NI variant still shows Recall@10 ≈ 0.845 — not from the metric bug, but from **unpaired noise multipliers**. Query noise is sampled at query time; document noise is sampled at upload time. Under a dot-product metric (scale-sensitive), independently sampled noise vectors distort ⟨A(q ⊙ n_q), B(d ⊙ n_d)⟩ ≠ ⟨q, d⟩. A paired construction (n_query = 1/n_document element-wise) would restore inner-product preservation; this is left as future work.

### 10.2 CKKS Rank Corruption Bug (Fixed)

**Original implementation:** Used a log(N) homomorphic rotate-and-add loop to sum CKKS slots, where N = poly_modulus_degree = 8192.

**Problem:** The rotation loop accumulated homomorphic noise at scale ≈ 2⁸⁰ after 13 rotations (log₂(8192) = 13). This noise corrupted the final sum, producing incorrect similarity scores and rank corruption.

**Fix:** After element-wise homomorphic multiplication (ct_q ⊙ ct_d), decrypt to plaintext and sum in floating-point:

```
score = Σ_i Dec(ct_q ⊙ ct_d)[i]
```

This is semantically equivalent to the homomorphic sum but avoids noise accumulation. It does require decryption before summation, which means Eve cannot rank documents without the decryption key — the security model is unchanged (Eve stores ciphertexts; VPK performs decryption and ranking).

**Result:** CKKS Recall@10 = 1.000 across all K values and both datasets after the fix.

---

## 11. Reproducing Results

### 11.1 Current Run

The experiments in the `20260509_153345` run are executing against:
- Full NFCorpus: 3,633 documents (IDs 1,191–4,823 in PostgreSQL)
- Backend: release build (`./target/release/vpkbench`)
- Embedding server: `all-MiniLM-L6-v2` on CPU at port 5001
- N = 10 trials, 30 queries per configuration

Monitor progress:
```bash
tail -f /tmp/vpkbench_run.log
```

Results accumulate in:
```
experiments/results/nfcorpus/20260509_153345/
experiments/results/nq/20260509_153345/
```

A `latest` symlink in each dataset directory always points to the most recent run.

### 11.2 Pre-Computed Results

The repository includes pre-computed N=10 results in `experiments/results/{nfcorpus,nq}/` (flat layout, from the initial anonymized release). These can be used directly with `plot_results.py` and `analysis.ipynb` without re-running the suite.

### 11.3 Expected Runtime

| Experiment | NFCorpus (3,633 docs) | Notes |
|------------|----------------------|-------|
| Exp A | ~3.5 h | CKKS: ~36 s/query × 60 queries × 10 trials |
| Exp D | ~12 h | 7 noise levels × 5 algorithms × CKKS |
| Exp E | ~17 h | 5 K values × 11 algorithms × CKKS |
| Exp I | < 1 h | Combined only, no CKKS |
| **Total** | **~33 h** | |

NQ adds a comparable additional runtime for Exp A and D. Exp G, H run in minutes (Combined only, no CKKS).

### 11.4 Partial Runs

To add a single algorithm without re-running all:
```bash
python experiments/run_experiments.py --exp A --dataset nfcorpus \
    --only-algos Ckks,CkksCombined
```

`--only-algos` appends rows to existing CSVs without deleting them, and writes to the flat `results/<dataset>/` directory (not a run-id subdirectory).

---

*Documentation generated 2026-05-09. Experiment run 20260509_153345 in progress.*
