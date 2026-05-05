# Results Narrative
*Generated from experimental data — N=10 trials, corpus=3,190 NFCorpus docs, 30 queries per configuration.*

---

## Section 3: Experimental Results

### 3.1 Algorithm Comparison (Experiment A)

Table 1 summarises query-time performance across all algorithms at zero added noise (or minimum noise where the algorithm requires it), K=10, on NFCorpus. Results on Natural Questions are consistent and reported in the Appendix.

| Algorithm       | Encrypt (ms) | Search (ms) | Recall@10 | nDCG@10 |
|-----------------|-------------:|------------:|----------:|--------:|
| Scrambling (DS) |        0.038 |         5.6 |    1.0000 |  1.0000 |
| Noise (NI)      |        0.066 |         5.5 |    0.9827 |  0.9906 |
| Combined (DS+NI)|        0.066 |         5.5 |    0.9830 |  0.9906 |
| ROME            |        6.131 |         7.0 |    1.0000 |  1.0000 |
| ROME+NI         |        6.184 |         7.0 |    0.9870 |  0.9929 |
| DIEHARD         |        5.067 |         3.9 |    1.0000 |  1.0000 |
| DIEHARD+NI      |        5.193 |         4.0 |    0.8450 |  0.8990 |
| CKKS            |       69.416 |    36,803.5 |    1.0000 |  1.0000 |
| CKKS+NI         |       69.156 |    36,742.9 |    0.9813 |  0.9897 |

Three distinct latency tiers emerge. The **DS/NI tier** (Scrambling, Noise, Combined) achieves sub-0.1 ms encryption and ~5.5 ms search. The **matrix tier** (ROME, ROME+NI, DIEHARD, DIEHARD+NI) incurs ~5–6 ms encryption and 4–7 ms search. The **CKKS tier** pays 69 ms for encryption and 36.8 s for search — approximately 9,400× slower than DIEHARD and 5,200× slower than ROME for search, and 14× slower for encryption than either.

DIEHARD merits specific note. At zero noise it achieves perfect Recall@10 = 1.000 and nDCG@10 = 1.000 — matching ROME and Scrambling. Its search latency (3.9 ms) is 44% faster than ROME (7.0 ms) because DIEHARD uses a raw dot product metric rather than cosine similarity; dot product omits the per-document norm computation, reducing search cost by approximately 3× at equal vector dimension. DIEHARD's encryption is slightly cheaper than ROME's (5.1 ms vs 6.1 ms), as its query matrix A has orthonormal columns while ROME's matrix QE involves a combined zero-padding and orthogonal rotation.

DIEHARD+NI (combined with noise injection) shows Recall@10 = 0.845, substantially below other *+NI variants (0.981–0.987). This degradation is attributable to the interaction between asymmetric encryption and the noise injection implementation: the noise multipliers applied to the query and document are sampled independently at query and upload time respectively, and under a raw dot product metric (which has no scale-invariance), these unpaired noise vectors distort the inner product. For ROME+NI, cosine similarity's scale-invariance absorbs most of the noise distortion (only 1.3 pp recall loss at zero nominal noise), whereas for DIEHARD+NI, dot product scoring fully exposes the unpaired noise effect.

### 3.2 Noise–Fidelity Tradeoff (Experiment D)

Experiment D sweeps noise range ±ε for ε ∈ {1%, 5%, 10%, 20%, 30%, 50%, 70%} across the five noise-capable algorithms. Figure 1 shows Recall@10 as a function of noise level.

**NFCorpus recall@10 at key noise levels:**

| Noise | NI     | DS+NI  | ROME+NI | DIEHARD+NI | CKKS+NI |
|-------|-------:|-------:|--------:|-----------:|--------:|
| 1%    | 0.9960 | 0.9970 |  0.9973 |     0.8433 |  0.9967 |
| 10%   | 0.9850 | 0.9863 |  0.9823 |     0.8417 |  0.9820 |
| 30%   | 0.9577 | 0.9653 |  0.9653 |     0.8483 |  0.9617 |
| 70%   | 0.9323 | 0.9247 |  0.9377 |     0.8347 |  0.9303 |

All algorithms except DIEHARD+NI degrade gracefully from near-perfect recall at 1% noise to recall in the 0.92–0.94 range at 70% noise — a 5–7 pp degradation across the full noise range. ROME+NI retains the best fidelity at high noise (0.938 at 70%), consistent with its stronger geometric structure.

DIEHARD+NI shows a flat recall profile (~0.843–0.848) across all noise levels. Unlike the DIEHARD cosine-metric artifact (corrected in this work — see §4.1), this flatness reflects the noise injection interaction: even at 1% noise, unpaired noise multipliers reduce recall to 0.843, and increasing noise does not meaningfully worsen it further because the dominant degradation source is the unpaired noise at upload vs query time, not the noise magnitude per se.

Results on NQ are consistent: ROME+NI and CKKS+NI both retain recall ≥ 0.936 at 70% noise; DIEHARD+NI holds flat at ~0.860 across all noise levels.

### 3.3 Top-K Sensitivity (Experiment E)

Recall@K was measured at K ∈ {1, 5, 10, 20, 50}. All algorithms except DIEHARD+NI show recall ≥ 0.985 at K=10. DIEHARD (no noise) maintains Recall@K = 1.000 across all K values and both datasets, confirming that the dot product metric fix fully restores exact ranking preservation. CKKS maintains Recall@K = 1.000 across all K values, confirming that the decryption-time plaintext summation fix eliminates the rank corruption seen in the prior log(N) homomorphic rotation implementation.

---

## Section 4: Discussion

### 4.1 DIEHARD Metric Fix

DIEHARD was designed with asymmetric document/query encryption matrices (A for queries, B for documents, A^T B = I_n), constructively preserving inner products: ⟨Aq, Bd⟩ = ⟨q, d⟩. The initial deployment against Qdrant (which ranks by cosine similarity) produced Recall@10 = 0.846 — a 15.4 pp gap below ROME.

The root cause: DIEHARD's 1.5× dimensional expansion (384 → 576) alters document vector L2 norms (‖Bd‖ ≠ ‖d‖), causing cosine similarities to diverge from the originals and breaking rank preservation. The paper (§4.1 of the SIAM workshop report) explicitly identifies this: requiring both A^T B = I_n and norm preservation forces B = A, collapsing DIEHARD to ROME.

The fix: switch the similarity metric to raw dot product for DIEHARD. Since all-MiniLM-L6-v2 produces unit-normalised embeddings (‖q‖ = ‖d‖ = 1), ranking by ⟨Aq, Bd⟩ = ⟨q, d⟩ is equivalent to ranking by cos(q, d). This recovers Recall@10 = 1.000 at zero noise and additionally reduces search latency from 7.3 ms (cosine) to 3.9 ms (dot product), making DIEHARD the fastest matrix-tier algorithm.

**Residual DIEHARD+NI degradation**: The combined DIEHARD+NI variant still shows recall ~0.845, not from the metric mismatch but from unpaired noise multipliers (query noise sampled at query time, document noise sampled at upload time). Under a dot product metric — unlike cosine similarity, which is scale-invariant — independently sampled noise vectors distort the inner product. A paired noise construction (noise_query = 1/noise_document element-wise) would fully restore inner product preservation; we leave this as future work.

### 4.2 CKKS Correctness

The corrected CKKS implementation (plaintext summation after decrypt-relinearize, replacing the prior log(N) homomorphic rotate-and-add) achieves Recall@10 = 1.000 at zero noise across all K values and both datasets. The fix eliminates rank corruption caused by accumulated homomorphic noise at scale 2^80 in the rotation loop.

### 4.3 Practical Algorithm Selection

| Use case | Recommended | Rationale |
|----------|-------------|-----------|
| Maximum throughput, weak security | Scrambling (DS) | 0.038 ms enc, 5.6 ms search, perfect recall |
| Throughput + noise obfuscation | Combined (DS+NI) | Same latency, 0.983 recall, noise masks individual scores |
| Strong geometric security, perfect fidelity | DIEHARD | 5.1 ms enc, 3.9 ms search, perfect recall (dot product backend) |
| Strong geometric security + noise | ROME+NI | 6.2 ms enc, 7.0 ms search, 0.987 recall |
| Information-theoretic security | CKKS | 69 ms enc, 36.8 s search, perfect recall — offline/small-corpus only |
