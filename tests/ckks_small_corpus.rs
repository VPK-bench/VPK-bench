//! CKKS small-corpus integration test
//!
//! Validates the full CKKS pipeline (encrypt → homomorphic dot-product → decrypt)
//! on a realistic small corpus of synthetic unit-length embeddings.
//!
//! Run:  cargo test --test ckks_small_corpus -- --nocapture

use ndarray::Array1;
use phe::crypto::Ckks;
use std::time::Instant;

const DIM: usize = 64;

fn normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        v.clone()
    } else {
        v / norm
    }
}

fn make_embedding(seed: usize, dim: usize) -> Array1<f64> {
    normalize(&Array1::from_vec(
        (0..dim)
            .map(|i| ((seed * 17 + i * 31) as f64).sin() + ((seed * 7 + i * 13) as f64).cos())
            .collect(),
    ))
}

fn plain_dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// 20-doc corpus, verify top-K ranking matches plaintext for several queries.
#[test]
fn test_ckks_corpus_ranking() {
    let corpus_size = 20;
    let top_k = 5;

    println!("\n=== CKKS small-corpus ranking test ===");
    println!("  corpus_size = {}, dim = {}, top_k = {}", corpus_size, DIM, top_k);

    let t0 = Instant::now();
    let ckks = Ckks::new(DIM, DIM + 32).expect("Ckks::new failed");
    println!("  SEAL context created in {:?}", t0.elapsed());

    let docs: Vec<Array1<f64>> = (0..corpus_size).map(|i| make_embedding(i, DIM)).collect();

    let t_enc = Instant::now();
    let ct_docs: Vec<Vec<u8>> = docs.iter().map(|d| ckks.encrypt(d).unwrap()).collect();
    let enc_elapsed = t_enc.elapsed();
    println!(
        "  Encrypted {} docs in {:?} ({:.1} ms/doc)",
        corpus_size,
        enc_elapsed,
        enc_elapsed.as_secs_f64() * 1000.0 / corpus_size as f64
    );

    let queries = vec![
        ("near doc 0", make_embedding(0, DIM)),
        ("near doc 10", make_embedding(10, DIM)),
        ("orthogonal", normalize(&Array1::from_vec((0..DIM).map(|i| (i as f64 * 0.7).cos()).collect()))),
    ];

    for (label, query) in &queries {
        println!("\n  Query: {}", label);

        let mut plain_scores: Vec<(usize, f64)> = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (i, plain_dot(query, d)))
            .collect();
        plain_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ct_query = ckks.encrypt(query).unwrap();

        let t_search = Instant::now();
        let mut enc_scores: Vec<(usize, f64)> = ct_docs
            .iter()
            .enumerate()
            .map(|(i, ct_d)| {
                let s = ckks.dot_product_decrypt(&ct_query, ct_d).unwrap();
                (i, s)
            })
            .collect();
        enc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("    search time: {:?}", t_search.elapsed());

        let plain_top: Vec<usize> = plain_scores.iter().take(top_k).map(|s| s.0).collect();
        let enc_top: Vec<usize> = enc_scores.iter().take(top_k).map(|s| s.0).collect();

        println!("    plain top-{}: {:?}", top_k, plain_top);
        println!("    enc   top-{}: {:?}", top_k, enc_top);

        for (p, e) in plain_scores.iter().take(top_k).zip(enc_scores.iter().take(top_k)) {
            println!(
                "      doc {:>2}: plain={:+.6}  enc={:+.6}  Δ={:.6}",
                p.0,
                p.1,
                e.1,
                (p.1 - e.1).abs()
            );
        }

        assert_eq!(
            plain_top[0], enc_top[0],
            "Top-1 must match for query '{}'",
            label
        );

        let overlap: usize = plain_top.iter().filter(|id| enc_top.contains(id)).count();
        let recall = overlap as f64 / top_k as f64;
        println!("    recall@{}: {:.0}%", top_k, recall * 100.0);
        assert!(
            recall >= 0.8,
            "recall@{} = {:.0}% (expected ≥ 80%) for query '{}'",
            top_k,
            recall * 100.0,
            label
        );
    }

    println!("\n=== PASSED ===\n");
}

/// Encrypt → decrypt round-trip fidelity on corpus vectors.
#[test]
fn test_ckks_corpus_roundtrip_fidelity() {
    let corpus_size = 20;

    println!("\n=== CKKS round-trip fidelity ===");

    let ckks = Ckks::new(DIM, DIM + 32).unwrap();
    let docs: Vec<Array1<f64>> = (0..corpus_size).map(|i| make_embedding(i, DIM)).collect();

    let mut max_err = 0.0_f64;
    let mut sum_err = 0.0_f64;

    for (i, doc) in docs.iter().enumerate() {
        let ct = ckks.encrypt(doc).unwrap();
        let dec = ckks.decrypt(&ct).unwrap();

        let err: f64 = doc
            .iter()
            .zip(dec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        max_err = max_err.max(err);
        sum_err += err;

        if err > 1e-3 {
            println!("  doc {}: max element error = {:.2e} (WARNING)", i, err);
        }
    }

    let avg_err = sum_err / corpus_size as f64;
    println!(
        "  max element error across corpus: {:.2e}",
        max_err
    );
    println!("  avg max-element error: {:.2e}", avg_err);

    assert!(
        max_err < 1e-2,
        "Round-trip fidelity too low: max_err = {:.2e}",
        max_err
    );

    println!("=== PASSED ===\n");
}

/// Dot-product accuracy: compare encrypted vs plaintext dot products.
#[test]
fn test_ckks_corpus_dot_product_accuracy() {
    let corpus_size = 20;

    println!("\n=== CKKS dot-product accuracy ===");

    let ckks = Ckks::new(DIM, DIM + 32).unwrap();
    let docs: Vec<Array1<f64>> = (0..corpus_size).map(|i| make_embedding(i, DIM)).collect();
    let query = make_embedding(999, DIM);

    let ct_query = ckks.encrypt(&query).unwrap();
    let ct_docs: Vec<Vec<u8>> = docs.iter().map(|d| ckks.encrypt(d).unwrap()).collect();

    let mut max_delta = 0.0_f64;
    for (i, (doc, ct_doc)) in docs.iter().zip(ct_docs.iter()).enumerate() {
        let p = plain_dot(&query, doc);
        let e = ckks.dot_product_decrypt(&ct_query, ct_doc).unwrap();
        let delta = (p - e).abs();
        max_delta = max_delta.max(delta);
        println!("  doc {:>2}: plain={:+.6}  enc={:+.6}  Δ={:.6}", i, p, e, delta);
    }

    println!("  max |Δ| = {:.6}", max_delta);

    assert!(
        max_delta < 0.1,
        "Dot-product deviation too large: {:.6}",
        max_delta
    );

    println!("=== PASSED ===\n");
}

/// Full-ranking Kendall-τ correlation between plaintext and encrypted orderings.
#[test]
fn test_ckks_corpus_kendall_tau() {
    let corpus_size = 20;

    println!("\n=== CKKS Kendall-τ ranking correlation ===");

    let ckks = Ckks::new(DIM, DIM + 32).unwrap();
    let docs: Vec<Array1<f64>> = (0..corpus_size).map(|i| make_embedding(i, DIM)).collect();
    let query = make_embedding(42, DIM);

    let ct_query = ckks.encrypt(&query).unwrap();
    let ct_docs: Vec<Vec<u8>> = docs.iter().map(|d| ckks.encrypt(d).unwrap()).collect();

    let mut plain_scores: Vec<(usize, f64)> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| (i, plain_dot(&query, d)))
        .collect();
    plain_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut enc_scores: Vec<(usize, f64)> = ct_docs
        .iter()
        .enumerate()
        .map(|(i, ct)| (i, ckks.dot_product_decrypt(&ct_query, ct).unwrap()))
        .collect();
    enc_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let plain_rank: Vec<usize> = plain_scores.iter().map(|s| s.0).collect();
    let enc_rank: Vec<usize> = enc_scores.iter().map(|s| s.0).collect();

    let n = corpus_size;
    let mut concordant = 0_i64;
    let mut discordant = 0_i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let pi = plain_rank.iter().position(|&x| x == i).unwrap() as i64;
            let pj = plain_rank.iter().position(|&x| x == j).unwrap() as i64;
            let ei = enc_rank.iter().position(|&x| x == i).unwrap() as i64;
            let ej = enc_rank.iter().position(|&x| x == j).unwrap() as i64;
            if (pi - pj).signum() == (ei - ej).signum() {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let tau = (concordant - discordant) as f64 / (concordant + discordant) as f64;

    println!("  plain ranking: {:?}", plain_rank);
    println!("  enc   ranking: {:?}", enc_rank);
    println!(
        "  concordant={}, discordant={}, τ={:.4}",
        concordant, discordant, tau
    );

    assert!(
        tau > 0.85,
        "Kendall τ = {:.4} too low (expected > 0.85)",
        tau
    );

    println!("=== PASSED ===\n");
}

/// Ciphertext size sanity check.
#[test]
fn test_ckks_ciphertext_size() {
    println!("\n=== CKKS ciphertext size ===");

    let ckks = Ckks::new(DIM, DIM + 32).unwrap();
    let v = make_embedding(0, DIM);
    let ct = ckks.encrypt(&v).unwrap();

    println!("  input dim:       {}", DIM);
    println!("  slot count:      {}", ckks.slot_count());
    println!("  ciphertext size: {} bytes ({:.1} KB)", ct.len(), ct.len() as f64 / 1024.0);
    println!(
        "  expansion ratio: {:.1}×",
        ct.len() as f64 / (DIM * 8) as f64
    );

    assert!(
        ct.len() > 0 && ct.len() < 10_000_000,
        "Ciphertext size {} looks wrong",
        ct.len()
    );

    println!("=== PASSED ===\n");
}
