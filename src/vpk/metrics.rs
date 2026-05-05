//! Evaluation metrics for NeurIPS experiments
//!
//! All rank-preservation metrics compare an *encrypted* ranking against a
//! *plaintext* baseline produced by the same VPK pipeline with no encryption.

use rand::Rng;
use std::collections::HashSet;

// ── Rank Preservation ────────────────────────────────────────────────────────

/// Fraction of plaintext top-K doc IDs that appear anywhere in the encrypted top-K.
///
/// Returns 1.0 if the two sets are identical, 0.0 if completely disjoint.
pub fn recall_at_k(ground_truth: &[i64], encrypted: &[i64], k: usize) -> f64 {
    let k = k.min(ground_truth.len()).min(encrypted.len());
    if k == 0 {
        return 1.0;
    }
    let gt_set: HashSet<i64> = ground_truth.iter().take(k).copied().collect();
    let hits = encrypted.iter().take(k).filter(|id| gt_set.contains(id)).count();
    hits as f64 / k as f64
}

/// Normalized Discounted Cumulative Gain at K.
///
/// Uses the plaintext similarity scores as graded relevance labels.
/// `gt_scores` is a slice of `(doc_id, plaintext_score)` in any order.
/// `enc_ranking` is the encrypted result list in ranked order (best first).
pub fn ndcg_at_k(gt_scores: &[(i64, f64)], enc_ranking: &[i64], k: usize) -> f64 {
    let k = k.min(enc_ranking.len());
    if k == 0 {
        return 1.0;
    }

    // Build relevance map from ground-truth scores.
    // Scores are clipped to [0, ∞); negative cosine similarities count as 0.
    let rel_map: std::collections::HashMap<i64, f64> = gt_scores
        .iter()
        .map(|&(id, s)| (id, s.max(0.0)))
        .collect();

    // DCG of encrypted ranking
    let dcg: f64 = enc_ranking
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = rel_map.get(id).copied().unwrap_or(0.0);
            rel / (i as f64 + 2.0).log2()
        })
        .sum();

    // Ideal DCG: sort ground-truth by score descending
    let mut ideal: Vec<f64> = gt_scores.iter().map(|&(_, s)| s.max(0.0)).collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, &rel)| rel / (i as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        return 1.0;
    }
    dcg / idcg
}

/// Kendall τ rank correlation between two orderings of the same doc IDs.
///
/// Returns a value in [-1, 1]: 1.0 = identical order, -1.0 = fully reversed,
/// 0.0 = uncorrelated. Only considers IDs present in both lists.
pub fn kendall_tau(ranking_a: &[i64], ranking_b: &[i64]) -> f64 {
    // Build position maps for IDs present in both lists
    let pos_a: std::collections::HashMap<i64, usize> =
        ranking_a.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let pos_b: std::collections::HashMap<i64, usize> =
        ranking_b.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let common: Vec<i64> = ranking_a
        .iter()
        .filter(|id| pos_b.contains_key(id))
        .copied()
        .collect();

    let n = common.len();
    if n < 2 {
        return 1.0;
    }

    let mut concordant = 0i64;
    let mut discordant = 0i64;

    for i in 0..n {
        for j in (i + 1)..n {
            let ai = pos_a[&common[i]];
            let aj = pos_a[&common[j]];
            let bi = pos_b[&common[i]];
            let bj = pos_b[&common[j]];

            let agree = (ai < aj) == (bi < bj);
            if agree {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let total = concordant + discordant;
    if total == 0 {
        return 1.0;
    }
    (concordant - discordant) as f64 / total as f64
}

/// True if the top-K sets are identical (order-insensitive).
pub fn exact_match_at_k(ground_truth: &[i64], encrypted: &[i64], k: usize) -> bool {
    let k = k.min(ground_truth.len()).min(encrypted.len());
    let gt_set: HashSet<i64> = ground_truth.iter().take(k).copied().collect();
    let enc_set: HashSet<i64> = encrypted.iter().take(k).copied().collect();
    gt_set == enc_set
}

// ── Encryption Strength Heuristics ───────────────────────────────────────────

/// Shannon entropy of a similarity score distribution (nats).
///
/// Scores are binned into `n_bins` equal-width buckets over [min, max].
/// Higher entropy → scores are more uniformly distributed → harder to exploit.
pub fn score_entropy(scores: &[f64], n_bins: usize) -> f64 {
    if scores.len() < 2 || n_bins == 0 {
        return 0.0;
    }
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return 0.0;
    }
    let width = (max - min) / n_bins as f64;
    let mut bins = vec![0usize; n_bins];
    for &s in scores {
        let idx = ((s - min) / width) as usize;
        bins[idx.min(n_bins - 1)] += 1;
    }
    let n = scores.len() as f64;
    bins.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / n;
            -p * p.ln()
        })
        .sum()
}

/// Variance of a score distribution.
///
/// Noise injection reduces variance — a lower value indicates scores are
/// harder to distinguish and membership inference is less reliable.
pub fn score_variance(scores: &[f64]) -> f64 {
    let n = scores.len();
    if n < 2 {
        return 0.0;
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64
}

/// Number of distinct score values in a result set.
///
/// Scores are rounded to 4 decimal places before counting.
/// Noise injection should collapse many scores to the same value.
pub fn score_distinct_count(scores: &[f64]) -> usize {
    let rounded: HashSet<u64> = scores
        .iter()
        .map(|&s| (s * 10_000.0).round() as u64)
        .collect();
    rounded.len()
}

/// Mean cosine distance between each encrypted vector and a random unit vector.
///
/// A value approaching 0.0 means the encrypted vectors are indistinguishable
/// from random — ideal for security. A value of 1.0 would mean perfectly
/// aligned with random (also distinguishable but in the opposite way).
/// For truly random unit vectors, expected value ≈ 0.5 * sqrt(π/2) ≈ 0.627.
pub fn cosine_distance_to_random<R: Rng>(encrypted_vecs: &[Vec<f64>], rng: &mut R) -> f64 {
    if encrypted_vecs.is_empty() {
        return 0.0;
    }
    let dim = encrypted_vecs[0].len();
    let distances: Vec<f64> = encrypted_vecs
        .iter()
        .map(|vec| {
            // Sample a random unit vector
            let raw: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            let norm: f64 = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                return 0.5;
            }
            let rand_unit: Vec<f64> = raw.iter().map(|x| x / norm).collect();

            // Cosine similarity, then distance = 1 - similarity
            let dot: f64 = vec.iter().zip(rand_unit.iter()).map(|(a, b)| a * b).sum();
            let vec_norm: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
            if vec_norm < 1e-12 {
                return 0.5;
            }
            1.0 - (dot / vec_norm).abs()
        })
        .collect();

    distances.iter().sum::<f64>() / distances.len() as f64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect_match() {
        let gt = vec![1, 2, 3, 4, 5];
        assert_eq!(recall_at_k(&gt, &gt, 5), 1.0);
    }

    #[test]
    fn recall_no_match() {
        let gt = vec![1, 2, 3];
        let enc = vec![4, 5, 6];
        assert_eq!(recall_at_k(&gt, &enc, 3), 0.0);
    }

    #[test]
    fn recall_partial() {
        let gt = vec![1, 2, 3, 4];
        let enc = vec![1, 2, 5, 6];
        assert!((recall_at_k(&gt, &enc, 4) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn ndcg_perfect() {
        let gt_scores = vec![(1, 1.0), (2, 0.8), (3, 0.5)];
        let enc = vec![1, 2, 3];
        assert!((ndcg_at_k(&gt_scores, &enc, 3) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_reversed() {
        let gt_scores = vec![(1, 1.0), (2, 0.8), (3, 0.5)];
        let enc = vec![3, 2, 1];
        let score = ndcg_at_k(&gt_scores, &enc, 3);
        assert!(score < 1.0, "Reversed order should have nDCG < 1.0");
        assert!(score > 0.0, "Even reversed order should have nDCG > 0 with overlapping items");
    }

    #[test]
    fn kendall_tau_identical() {
        let ranking = vec![1, 2, 3, 4];
        assert!((kendall_tau(&ranking, &ranking) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_tau_reversed() {
        let a = vec![1, 2, 3, 4];
        let b = vec![4, 3, 2, 1];
        assert!((kendall_tau(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn exact_match_same() {
        let gt = vec![1, 2, 3];
        assert!(exact_match_at_k(&gt, &gt, 3));
    }

    #[test]
    fn exact_match_reordered() {
        let gt = vec![1, 2, 3];
        let enc = vec![3, 1, 2];
        assert!(exact_match_at_k(&gt, &enc, 3));
    }

    #[test]
    fn score_entropy_uniform() {
        let scores: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let entropy = score_entropy(&scores, 10);
        assert!(entropy > 0.0);
    }

    #[test]
    fn score_entropy_constant() {
        let scores = vec![0.5f64; 50];
        assert_eq!(score_entropy(&scores, 10), 0.0);
    }

    #[test]
    fn score_variance_constant() {
        let scores = vec![0.5f64; 10];
        assert_eq!(score_variance(&scores), 0.0);
    }
}
