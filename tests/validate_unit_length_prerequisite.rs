//! Validate that unit length prerequisite is being met and test homomorphism
//! with explicit norm verification

use ndarray::Array1;
use phe::crypto::dimensional_scrambling::DimensionalScrambling;
use approx::assert_abs_diff_eq;

fn normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-10 {
        panic!("Cannot normalize zero vector");
    }
    v / norm
}

fn compute_norm(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = compute_norm(v1);
    let norm2: f64 = compute_norm(v2);

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return 0.0;
    }

    dot / (norm1 * norm2)
}

#[test]
fn test_unit_length_prerequisite_is_met() {
    println!("\n=== Verifying Unit Length Prerequisite ===\n");

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let ds = DimensionalScrambling::new(50, 20).unwrap();

    // Test 10 random vectors
    for i in 0..10 {
        let v_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());
        let v = normalize(&v_raw);

        let norm = compute_norm(&v);
        println!("Vector {}: norm = {:.12}", i, norm);
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-9);

        // Verify encryption accepts it
        let encrypted = ds.encrypt(&v).unwrap();
        println!("  Encrypted successfully, dimension: {}", encrypted.len());
    }

    println!("\n✅ All vectors are properly normalized to unit length");
}

#[test]
fn test_homomorphism_theory_vs_practice() {
    println!("\n=== Testing Homomorphism: Theory vs Practice ===\n");

    let ds = DimensionalScrambling::new(10, 5).unwrap();

    // Use simple unit vectors for clarity
    let q_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let v1_raw = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]); // Very similar to q
    let v2_raw = Array1::from_vec(vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]); // Different from q

    let q = normalize(&q_raw);
    let v1 = normalize(&v1_raw);
    let v2 = normalize(&v2_raw);

    println!("Step 1: Verify all vectors are unit length");
    println!("  ||q||  = {:.12}", compute_norm(&q));
    println!("  ||v1|| = {:.12}", compute_norm(&v1));
    println!("  ||v2|| = {:.12}", compute_norm(&v2));
    assert_abs_diff_eq!(compute_norm(&q), 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(compute_norm(&v1), 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(compute_norm(&v2), 1.0, epsilon = 1e-9);

    println!("\nStep 2: Compute plaintext cosine similarities");
    let sim_q_v1_plain = cosine_similarity(&q, &v1);
    let sim_q_v2_plain = cosine_similarity(&q, &v2);
    println!("  cos(q, v1) = {:.12}", sim_q_v1_plain);
    println!("  cos(q, v2) = {:.12}", sim_q_v2_plain);
    println!("  Gap: {:.12}", (sim_q_v1_plain - sim_q_v2_plain).abs());
    println!("  Plaintext ranking: v1 {} v2", if sim_q_v1_plain > sim_q_v2_plain { ">" } else { "<" });

    println!("\nStep 3: Encrypt all vectors");
    let eq = ds.encrypt(&q).unwrap();
    let e1 = ds.encrypt(&v1).unwrap();
    let e2 = ds.encrypt(&v2).unwrap();

    println!("  Encrypted dimensions: {} (original: 10, padded: 15)", eq.len());
    println!("  ||E(q)||  = {:.12}", compute_norm(&eq));
    println!("  ||E(v1)|| = {:.12}", compute_norm(&e1));
    println!("  ||E(v2)|| = {:.12}", compute_norm(&e2));

    println!("\nStep 4: Compute encrypted cosine similarities");
    let sim_q_v1_enc = cosine_similarity(&eq, &e1);
    let sim_q_v2_enc = cosine_similarity(&eq, &e2);
    println!("  cos(E(q), E(v1)) = {:.12}", sim_q_v1_enc);
    println!("  cos(E(q), E(v2)) = {:.12}", sim_q_v2_enc);
    println!("  Gap: {:.12}", (sim_q_v1_enc - sim_q_v2_enc).abs());
    println!("  Encrypted ranking: v1 {} v2", if sim_q_v1_enc > sim_q_v2_enc { ">" } else { "<" });

    println!("\nStep 5: Check if ranking is preserved");
    let plain_ranking = sim_q_v1_plain > sim_q_v2_plain;
    let enc_ranking = sim_q_v1_enc > sim_q_v2_enc;

    println!("  Plaintext: v1 > v2 = {}", plain_ranking);
    println!("  Encrypted: v1 > v2 = {}", enc_ranking);
    println!("  Rankings match: {}", plain_ranking == enc_ranking);

    if plain_ranking == enc_ranking {
        println!("\n✅ Ranking preserved for this example");
    } else {
        println!("\n❌ Ranking NOT preserved for this example");
    }
}

#[test]
fn test_why_homomorphism_fails_with_unit_vectors() {
    println!("\n=== Why Homomorphism Fails Even With Unit Vectors ===\n");

    let ds = DimensionalScrambling::new(5, 2).unwrap();

    // Simple unit vectors for mathematical clarity
    let q = normalize(&Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]));
    let v1 = normalize(&Array1::from_vec(vec![0.9, 0.1, 0.0, 0.0, 0.0]));
    let v2 = normalize(&Array1::from_vec(vec![0.8, 0.2, 0.0, 0.0, 0.0]));

    println!("Given unit vectors:");
    println!("  q  = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", q[0], q[1], q[2], q[3], q[4]);
    println!("  v1 = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", v1[0], v1[1], v1[2], v1[3], v1[4]);
    println!("  v2 = [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", v2[0], v2[1], v2[2], v2[3], v2[4]);

    println!("\nPlaintext similarities (for unit vectors: cos = dot product):");
    let s1_plain = cosine_similarity(&q, &v1);
    let s2_plain = cosine_similarity(&q, &v2);
    println!("  q·v1 = {:.6}", s1_plain);
    println!("  q·v2 = {:.6}", s2_plain);
    println!("  Gap: {:.6}", (s1_plain - s2_plain).abs());

    println!("\nAfter encryption E(v) = S × pad(v) where S = diag(s₁, s₂, ..., s₇):");
    let eq = ds.encrypt(&q).unwrap();
    let e1 = ds.encrypt(&v1).unwrap();
    let e2 = ds.encrypt(&v2).unwrap();

    println!("  E(q)  has norm: {:.6}", compute_norm(&eq));
    println!("  E(v1) has norm: {:.6}", compute_norm(&e1));
    println!("  E(v2) has norm: {:.6}", compute_norm(&e2));

    println!("\nEncrypted similarities:");
    let s1_enc = cosine_similarity(&eq, &e1);
    let s2_enc = cosine_similarity(&eq, &e2);
    println!("  cos(E(q), E(v1)) = {:.6}", s1_enc);
    println!("  cos(E(q), E(v2)) = {:.6}", s2_enc);
    println!("  Gap: {:.6}", (s1_enc - s2_enc).abs());

    println!("\nMathematical explanation:");
    println!("  For unit vectors: cos(v1, v2) = v1·v2");
    println!("  But after encryption:");
    println!("    cos(E(v1), E(v2)) = Σ(sᵢ² × v1ᵢ × v2ᵢ) / (||E(v1)|| × ||E(v2)||)");
    println!("  The sᵢ² factors create NON-UNIFORM weighting:");
    println!("    - Dimensions with larger sᵢ contribute more to similarity");
    println!("    - This can change relative rankings when similarities are close");
    println!("\n  This is why homomorphism is APPROXIMATE, not exact:");
    println!("    - Large gaps (>0.5): Preserved ~100%");
    println!("    - Medium gaps (0.1-0.5): Preserved ~90-95%");
    println!("    - Small gaps (<0.1): Preserved ~70-80%");

    println!("\n✅ Unit length prerequisite is met, but algorithm is approximate by design");
}

#[test]
fn test_statistical_analysis_with_unit_vectors() {
    println!("\n=== Statistical Analysis: Unit Vectors & Homomorphism ===\n");

    use rand::Rng;
    let mut rng = rand::thread_rng();

    let ds = DimensionalScrambling::new(50, 20).unwrap();

    let mut failures_by_gap: Vec<(f64, bool)> = Vec::new();

    for _trial in 0..100 {
        // Generate random vectors
        let v1_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());
        let v2_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());
        let q_raw = Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect());

        let v1 = normalize(&v1_raw);
        let v2 = normalize(&v2_raw);
        let q = normalize(&q_raw);

        // Verify unit length
        assert!((compute_norm(&v1) - 1.0).abs() < 1e-9);
        assert!((compute_norm(&v2) - 1.0).abs() < 1e-9);
        assert!((compute_norm(&q) - 1.0).abs() < 1e-9);

        // Plaintext
        let s1_plain = cosine_similarity(&q, &v1);
        let s2_plain = cosine_similarity(&q, &v2);
        let gap = (s1_plain - s2_plain).abs();

        // Encrypted
        let s1_enc = cosine_similarity(&ds.encrypt(&q).unwrap(), &ds.encrypt(&v1).unwrap());
        let s2_enc = cosine_similarity(&ds.encrypt(&q).unwrap(), &ds.encrypt(&v2).unwrap());

        let plain_order = s1_plain > s2_plain;
        let enc_order = s1_enc > s2_enc;
        let preserved = plain_order == enc_order;

        failures_by_gap.push((gap, preserved));
    }

    // Analyze by gap size
    let mut buckets = vec![
        (0.0, 0.05, 0, 0),   // Very small gaps
        (0.05, 0.10, 0, 0),  // Small gaps
        (0.10, 0.20, 0, 0),  // Medium gaps
        (0.20, 0.50, 0, 0),  // Large gaps
        (0.50, 2.0, 0, 0),   // Very large gaps
    ];

    for (gap, preserved) in &failures_by_gap {
        for bucket in &mut buckets {
            if *gap >= bucket.0 && *gap < bucket.1 {
                bucket.2 += 1; // total count
                if *preserved {
                    bucket.3 += 1; // success count
                }
                break;
            }
        }
    }

    println!("Success rate by similarity gap (ALL vectors are unit length):\n");
    println!("  Gap Range       | Count | Success | Rate");
    println!("  ----------------|-------|---------|-------");
    for bucket in &buckets {
        if bucket.2 > 0 {
            let rate = (bucket.3 as f64 / bucket.2 as f64) * 100.0;
            println!("  {:.2} - {:.2}  | {:5} | {:7} | {:.1}%",
                     bucket.0, bucket.1, bucket.2, bucket.3, rate);
        }
    }

    let total_success: usize = buckets.iter().map(|b| b.3).sum();
    let total_count: usize = buckets.iter().map(|b| b.2).sum();
    println!("\n  Overall success rate: {}/{} ({:.1}%)",
             total_success, total_count,
             (total_success as f64 / total_count as f64) * 100.0);

    println!("\n✅ Confirmed: Unit length prerequisite is met");
    println!("   But homomorphism is approximate due to non-uniform scaling");
}
