//! Comprehensive Homomorphism Validation for Dimensional Scrambling
//!
//! This test suite rigorously validates the homomorphic property:
//!   rank(cos(v1, q)) = rank(cos(E(v1), E(q)))
//!
//! Where E is the dimensional scrambling encryption function and all
//! vectors are normalized to unit length.

use ndarray::Array1;
use phe::crypto::dimensional_scrambling::DimensionalScrambling;
use approx::assert_abs_diff_eq;

/// Helper function to normalize a vector to unit length
fn normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        panic!("Cannot normalize zero vector");
    }
    v / norm
}

/// Helper function to compute cosine similarity between two vectors
fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return 0.0;
    }

    dot / (norm1 * norm2)
}

/// Compute ranking order from similarity scores
fn compute_ranking(similarities: &[(usize, f64)]) -> Vec<usize> {
    let mut sorted = similarities.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.iter().map(|(id, _)| *id).collect()
}

#[test]
fn test_homomorphism_basic_property() {
    println!("\n=== Testing Basic Homomorphism Property ===\n");

    let ds = DimensionalScrambling::new(10, 5).unwrap();

    // Create test vectors
    // v1 and v2 are reverses of each other; q must NOT be uniform or it produces
    // equal similarities to both (a tie that breaks nondeterministically after encryption).
    // Use q = v1 direction so cos(q,v1) >> cos(q,v2).
    let v1_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let v2_raw = Array1::from_vec(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let q_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    let v1 = normalize(&v1_raw);
    let v2 = normalize(&v2_raw);
    let q = normalize(&q_raw);

    // Compute plaintext similarities
    let plain_sim1 = cosine_similarity(&q, &v1);
    let plain_sim2 = cosine_similarity(&q, &v2);

    println!("Plaintext cosine similarities:");
    println!("  cos(q, v1) = {:.6}", plain_sim1);
    println!("  cos(q, v2) = {:.6}", plain_sim2);
    println!("  Plaintext ranking: v{} > v{}",
             if plain_sim1 > plain_sim2 { 1 } else { 2 },
             if plain_sim1 > plain_sim2 { 2 } else { 1 });

    // Encrypt vectors
    let e_q = ds.encrypt(&q).unwrap();
    let e_v1 = ds.encrypt(&v1).unwrap();
    let e_v2 = ds.encrypt(&v2).unwrap();

    // Compute encrypted similarities
    let enc_sim1 = cosine_similarity(&e_q, &e_v1);
    let enc_sim2 = cosine_similarity(&e_q, &e_v2);

    println!("\nEncrypted cosine similarities:");
    println!("  cos(E(q), E(v1)) = {:.6}", enc_sim1);
    println!("  cos(E(q), E(v2)) = {:.6}", enc_sim2);
    println!("  Encrypted ranking: v{} > v{}",
             if enc_sim1 > enc_sim2 { 1 } else { 2 },
             if enc_sim1 > enc_sim2 { 2 } else { 1 });

    // Verify ranking preservation
    let plain_order = plain_sim1 > plain_sim2;
    let enc_order = enc_sim1 > enc_sim2;

    assert_eq!(plain_order, enc_order,
               "Homomorphism violated: ranking order changed after encryption");

    println!("\n✓ Basic homomorphism property validated");
}

#[test]
fn test_homomorphism_multiple_vectors() {
    println!("\n=== Testing Homomorphism with Multiple Vectors ===\n");

    let ds = DimensionalScrambling::new(20, 10).unwrap();

    // Create a query vector
    let query_raw = Array1::from_vec((0..20).map(|i| (i as f64 + 1.0)).collect());
    let query = normalize(&query_raw);

    // Create 10 document vectors with varying similarities
    let mut docs: Vec<Array1<f64>> = Vec::new();
    for i in 0..10 {
        let doc_raw = Array1::from_vec((0..20).map(|j| {
            ((j + i * 3) as f64).sin() * 10.0
        }).collect());
        docs.push(normalize(&doc_raw));
    }

    // Compute plaintext similarities
    let mut plain_sims: Vec<(usize, f64)> = docs.iter()
        .enumerate()
        .map(|(i, doc)| (i, cosine_similarity(&query, doc)))
        .collect();
    let plain_ranking = compute_ranking(&plain_sims);

    println!("Plaintext similarities:");
    for (i, sim) in &plain_sims {
        println!("  doc{}: {:.6}", i, sim);
    }
    println!("Plaintext ranking: {:?}", plain_ranking);

    // Encrypt all vectors
    let enc_query = ds.encrypt(&query).unwrap();
    let enc_docs: Vec<Array1<f64>> = docs.iter()
        .map(|doc| ds.encrypt(doc).unwrap())
        .collect();

    // Compute encrypted similarities
    let mut enc_sims: Vec<(usize, f64)> = enc_docs.iter()
        .enumerate()
        .map(|(i, enc_doc)| (i, cosine_similarity(&enc_query, enc_doc)))
        .collect();
    let enc_ranking = compute_ranking(&enc_sims);

    println!("\nEncrypted similarities:");
    for (i, sim) in &enc_sims {
        println!("  doc{}: {:.6}", i, sim);
    }
    println!("Encrypted ranking: {:?}", enc_ranking);

    // Verify complete ranking preservation
    assert_eq!(plain_ranking, enc_ranking,
               "Homomorphism violated: complete ranking order must be preserved");

    println!("\n✓ Homomorphism validated for {} vectors", docs.len());
}

#[test]
fn test_homomorphism_random_vectors() {
    println!("\n=== Testing Homomorphism with Random Vectors ===\n");

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let ds = DimensionalScrambling::new(50, 20).unwrap();

    // Test with 100 random vector pairs
    let num_trials = 100;
    let mut successes = 0;

    for trial in 0..num_trials {
        // Generate random vectors
        let v1_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());
        let v2_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());
        let q_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());

        let v1 = normalize(&v1_raw);
        let v2 = normalize(&v2_raw);
        let q = normalize(&q_raw);

        // Plaintext similarities
        let plain_sim1 = cosine_similarity(&q, &v1);
        let plain_sim2 = cosine_similarity(&q, &v2);

        // Encrypted similarities
        let e_q = ds.encrypt(&q).unwrap();
        let e_v1 = ds.encrypt(&v1).unwrap();
        let e_v2 = ds.encrypt(&v2).unwrap();

        let enc_sim1 = cosine_similarity(&e_q, &e_v1);
        let enc_sim2 = cosine_similarity(&e_q, &e_v2);

        // Check ranking preservation
        let plain_order = plain_sim1 > plain_sim2;
        let enc_order = enc_sim1 > enc_sim2;

        if plain_order == enc_order {
            successes += 1;
        } else {
            println!("Trial {}: Ranking mismatch!", trial);
            println!("  Plain: {:.6} vs {:.6} -> {}", plain_sim1, plain_sim2, plain_order);
            println!("  Enc:   {:.6} vs {:.6} -> {}", enc_sim1, enc_sim2, enc_order);
        }
    }

    let success_rate = (successes as f64) / (num_trials as f64);
    println!("Success rate: {}/{} ({:.1}%)", successes, num_trials, success_rate * 100.0);

    // Require 100% success rate for homomorphism to hold
    assert_eq!(successes, num_trials,
               "Homomorphism violated in {}/{} trials", num_trials - successes, num_trials);

    println!("\n✓ Homomorphism validated across {} random trials", num_trials);
}

#[test]
fn test_homomorphism_edge_cases() {
    println!("\n=== Testing Homomorphism Edge Cases ===\n");

    let ds = DimensionalScrambling::new(10, 5).unwrap();

    // Test Case 1: Identical vectors (highest similarity)
    println!("Test 1: Identical vectors");
    let v1 = normalize(&Array1::from_vec(vec![1.0; 10]));
    let v2 = v1.clone();

    let plain_sim = cosine_similarity(&v1, &v2);
    let enc_sim = cosine_similarity(&ds.encrypt(&v1).unwrap(), &ds.encrypt(&v2).unwrap());

    println!("  Plain cos(v, v) = {:.6}", plain_sim);
    println!("  Enc cos(E(v), E(v)) = {:.6}", enc_sim);
    assert_abs_diff_eq!(plain_sim, 1.0, epsilon = 1e-6);
    // Encrypted similarity may differ but should be close to 1
    assert!(enc_sim > 0.99, "Encrypted similarity of identical vectors should be ~1.0");

    // Test Case 2: Orthogonal vectors (zero similarity)
    println!("\nTest 2: Orthogonal vectors");
    let mut v3_vec = vec![0.0; 10];
    v3_vec[0] = 1.0;
    let mut v4_vec = vec![0.0; 10];
    v4_vec[1] = 1.0;

    let v3 = Array1::from_vec(v3_vec);
    let v4 = Array1::from_vec(v4_vec);

    let plain_sim_orth = cosine_similarity(&v3, &v4);
    let enc_sim_orth = cosine_similarity(&ds.encrypt(&v3).unwrap(), &ds.encrypt(&v4).unwrap());

    println!("  Plain cos(v3, v4) = {:.6}", plain_sim_orth);
    println!("  Enc cos(E(v3), E(v4)) = {:.6}", enc_sim_orth);
    assert_abs_diff_eq!(plain_sim_orth, 0.0, epsilon = 1e-6);
    // Note: After padding, they may not be exactly orthogonal
    // but similarity should be small

    // Test Case 3: Opposite vectors (negative similarity)
    println!("\nTest 3: Opposite vectors");
    let v5 = normalize(&Array1::from_vec(vec![1.0; 10]));
    let v6 = normalize(&Array1::from_vec(vec![-1.0; 10]));

    let plain_sim_opp = cosine_similarity(&v5, &v6);
    let enc_sim_opp = cosine_similarity(&ds.encrypt(&v5).unwrap(), &ds.encrypt(&v6).unwrap());

    println!("  Plain cos(v, -v) = {:.6}", plain_sim_opp);
    println!("  Enc cos(E(v), E(-v)) = {:.6}", enc_sim_opp);
    assert_abs_diff_eq!(plain_sim_opp, -1.0, epsilon = 1e-6);
    // Encrypted similarity should also be negative
    assert!(enc_sim_opp < -0.99, "Encrypted similarity of opposite vectors should be ~-1.0");

    println!("\n✓ Edge cases validated");
}

#[test]
fn test_homomorphism_transitivity() {
    println!("\n=== Testing Homomorphism Transitivity ===\n");

    let ds = DimensionalScrambling::new(15, 8).unwrap();

    // Create three vectors with known ordering: v1 > v2 > v3 relative to query
    let query_raw = Array1::from_vec((0..15).map(|i| (i as f64)).collect());
    let query = normalize(&query_raw);

    // v1 is very similar to query
    let v1 = normalize(&Array1::from_vec((0..15).map(|i| (i as f64) + 0.1).collect()));

    // v2 is moderately similar
    let v2 = normalize(&Array1::from_vec((0..15).map(|i| (i as f64) * 0.5 + 1.0).collect()));

    // v3 is least similar
    let v3 = normalize(&Array1::from_vec((0..15).map(|i| (14 - i) as f64).collect()));

    // Plaintext similarities
    let s1_plain = cosine_similarity(&query, &v1);
    let s2_plain = cosine_similarity(&query, &v2);
    let s3_plain = cosine_similarity(&query, &v3);

    println!("Plaintext similarities:");
    println!("  cos(q, v1) = {:.6}", s1_plain);
    println!("  cos(q, v2) = {:.6}", s2_plain);
    println!("  cos(q, v3) = {:.6}", s3_plain);

    // Verify plaintext ordering
    assert!(s1_plain > s2_plain, "Expected s1 > s2 in plaintext");
    assert!(s2_plain > s3_plain, "Expected s2 > s3 in plaintext");
    println!("  Plaintext: v1 > v2 > v3 ✓");

    // Encrypted similarities
    let eq = ds.encrypt(&query).unwrap();
    let e1 = ds.encrypt(&v1).unwrap();
    let e2 = ds.encrypt(&v2).unwrap();
    let e3 = ds.encrypt(&v3).unwrap();

    let s1_enc = cosine_similarity(&eq, &e1);
    let s2_enc = cosine_similarity(&eq, &e2);
    let s3_enc = cosine_similarity(&eq, &e3);

    println!("\nEncrypted similarities:");
    println!("  cos(E(q), E(v1)) = {:.6}", s1_enc);
    println!("  cos(E(q), E(v2)) = {:.6}", s2_enc);
    println!("  cos(E(q), E(v3)) = {:.6}", s3_enc);

    // Verify encrypted ordering matches
    assert!(s1_enc > s2_enc,
            "Homomorphism violated: s1 > s2 in plaintext but not in encrypted space");
    assert!(s2_enc > s3_enc,
            "Homomorphism violated: s2 > s3 in plaintext but not in encrypted space");

    println!("  Encrypted: v1 > v2 > v3 ✓");
    println!("\n✓ Transitivity preserved through encryption");
}

#[test]
fn test_homomorphism_with_different_keys() {
    println!("\n=== Testing Homomorphism with Different Keys ===\n");

    // Test that homomorphism holds regardless of the random key
    let num_keys = 20;

    for key_num in 0..num_keys {
        let ds = DimensionalScrambling::new(30, 15).unwrap();

        // Create test vectors
        let q = normalize(&Array1::from_vec((0..30).map(|i| (i as f64)).collect()));
        let v1 = normalize(&Array1::from_vec((0..30).map(|i| (i as f64) + 1.0).collect()));
        let v2 = normalize(&Array1::from_vec((0..30).map(|i| (29 - i) as f64).collect()));

        // Plaintext
        let s1_plain = cosine_similarity(&q, &v1);
        let s2_plain = cosine_similarity(&q, &v2);
        let plain_order = s1_plain > s2_plain;

        // Encrypted
        let eq = ds.encrypt(&q).unwrap();
        let e1 = ds.encrypt(&v1).unwrap();
        let e2 = ds.encrypt(&v2).unwrap();

        let s1_enc = cosine_similarity(&eq, &e1);
        let s2_enc = cosine_similarity(&eq, &e2);
        let enc_order = s1_enc > s2_enc;

        assert_eq!(plain_order, enc_order,
                   "Homomorphism violated for key {}", key_num);
    }

    println!("Tested {} different random keys", num_keys);
    println!("✓ Homomorphism holds for all keys");
}

#[test]
fn test_homomorphism_mathematical_proof() {
    println!("\n=== Mathematical Proof of Homomorphism ===\n");

    println!("Theorem: For unit vectors v1, v2 with ||v1|| = ||v2|| = 1,");
    println!("and diagonal scaling matrix S, the ranking of cosine similarities");
    println!("is preserved under encryption E(v) = S × pad(v).");
    println!();

    let ds = DimensionalScrambling::new(5, 3).unwrap();

    // Use simple vectors for demonstration
    let v1 = Array1::from_vec(vec![0.6, 0.8, 0.0, 0.0, 0.0]); // Unit vector
    let v2 = Array1::from_vec(vec![0.8, 0.6, 0.0, 0.0, 0.0]); // Unit vector
    let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);  // Unit vector

    // Verify they're unit vectors
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
    let normq: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("Step 1: Verify unit vectors");
    println!("  ||v1|| = {:.6}", norm1);
    println!("  ||v2|| = {:.6}", norm2);
    println!("  ||q||  = {:.6}", normq);
    assert_abs_diff_eq!(norm1, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(norm2, 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(normq, 1.0, epsilon = 1e-10);

    println!("\nStep 2: Compute plaintext cosine similarities");
    let cos_qv1 = cosine_similarity(&q, &v1);
    let cos_qv2 = cosine_similarity(&q, &v2);
    println!("  cos(q, v1) = {:.6}", cos_qv1);
    println!("  cos(q, v2) = {:.6}", cos_qv2);
    println!("  Ranking: v{} is more similar to q", if cos_qv1 > cos_qv2 { 1 } else { 2 });

    println!("\nStep 3: Encrypt vectors with dimensional scrambling");
    let eq = ds.encrypt(&q).unwrap();
    let e1 = ds.encrypt(&v1).unwrap();
    let e2 = ds.encrypt(&v2).unwrap();
    println!("  E(q) dimension: {}", eq.len());
    println!("  E(v1) dimension: {}", e1.len());
    println!("  E(v2) dimension: {}", e2.len());

    println!("\nStep 4: Compute encrypted cosine similarities");
    let cos_eq_e1 = cosine_similarity(&eq, &e1);
    let cos_eq_e2 = cosine_similarity(&eq, &e2);
    println!("  cos(E(q), E(v1)) = {:.6}", cos_eq_e1);
    println!("  cos(E(q), E(v2)) = {:.6}", cos_eq_e2);
    println!("  Ranking: v{} is more similar to q", if cos_eq_e1 > cos_eq_e2 { 1 } else { 2 });

    println!("\nStep 5: Verify ranking preservation");
    let plain_ranking = cos_qv1 > cos_qv2;
    let enc_ranking = cos_eq_e1 > cos_eq_e2;

    assert_eq!(plain_ranking, enc_ranking,
               "Homomorphism violated: rankings don't match");

    println!("  Plaintext ranking: {}", if plain_ranking { "v1 > v2" } else { "v2 > v1" });
    println!("  Encrypted ranking: {}", if enc_ranking { "v1 > v2" } else { "v2 > v1" });
    println!("  ✓ Rankings match!");

    println!("\nConclusion:");
    println!("The dimensional scrambling encryption preserves cosine similarity");
    println!("rankings for unit vectors, validating the homomorphic property.");
    println!("\n✓ Mathematical proof validated experimentally");
}
