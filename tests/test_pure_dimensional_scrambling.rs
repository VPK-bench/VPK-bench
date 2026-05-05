//! Test pure dimensional scrambling (permutation only, no padding/scaling)
//! to validate that it provides EXACT homomorphism preservation

use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use approx::assert_abs_diff_eq;

fn normalize(v: &Array1<f64>) -> Array1<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v / norm
}

fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> f64 {
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (norm1 * norm2)
}

/// Pure dimensional scrambling: just permute the elements
fn pure_scramble(v: &Array1<f64>, permutation: &[usize]) -> Array1<f64> {
    let mut scrambled = Array1::zeros(v.len());
    for (i, &perm_idx) in permutation.iter().enumerate() {
        scrambled[i] = v[perm_idx];
    }
    scrambled
}

#[test]
fn test_pure_scrambling_preserves_norm() {
    println!("\n=== Pure Scrambling Preserves Norm ===\n");

    let v = normalize(&Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let permutation = vec![4, 2, 0, 3, 1]; // arbitrary permutation

    let scrambled = pure_scramble(&v, &permutation);

    let norm_before = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_after = scrambled.iter().map(|x| x * x).sum::<f64>().sqrt();

    println!("Original vector: {:?}", v.as_slice().unwrap());
    println!("Scrambled vector: {:?}", scrambled.as_slice().unwrap());
    println!("Norm before: {:.12}", norm_before);
    println!("Norm after:  {:.12}", norm_after);

    assert_abs_diff_eq!(norm_before, 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(norm_after, 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(norm_before, norm_after, epsilon = 1e-12);

    println!("\n✅ Permutation preserves norm exactly");
}

#[test]
fn test_pure_scrambling_preserves_cosine_similarity_exactly() {
    println!("\n=== Pure Scrambling Preserves Cosine Similarity EXACTLY ===\n");

    let q = normalize(&Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let v1 = normalize(&Array1::from_vec(vec![1.1, 2.1, 3.1, 4.1, 5.1]));
    let v2 = normalize(&Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]));

    // Use same permutation for all vectors (this is the key!)
    let permutation = vec![4, 2, 0, 3, 1];

    // Plaintext similarities
    let sim_q_v1_plain = cosine_similarity(&q, &v1);
    let sim_q_v2_plain = cosine_similarity(&q, &v2);

    println!("Plaintext similarities:");
    println!("  cos(q, v1) = {:.12}", sim_q_v1_plain);
    println!("  cos(q, v2) = {:.12}", sim_q_v2_plain);
    println!("  Gap: {:.12}", (sim_q_v1_plain - sim_q_v2_plain).abs());

    // Scramble all vectors with SAME permutation
    let sq = pure_scramble(&q, &permutation);
    let sv1 = pure_scramble(&v1, &permutation);
    let sv2 = pure_scramble(&v2, &permutation);

    // Scrambled similarities
    let sim_q_v1_scrambled = cosine_similarity(&sq, &sv1);
    let sim_q_v2_scrambled = cosine_similarity(&sq, &sv2);

    println!("\nScrambled similarities:");
    println!("  cos(S(q), S(v1)) = {:.12}", sim_q_v1_scrambled);
    println!("  cos(S(q), S(v2)) = {:.12}", sim_q_v2_scrambled);
    println!("  Gap: {:.12}", (sim_q_v1_scrambled - sim_q_v2_scrambled).abs());

    println!("\nDifferences:");
    println!("  |cos(q,v1) - cos(S(q),S(v1))| = {:.15}", (sim_q_v1_plain - sim_q_v1_scrambled).abs());
    println!("  |cos(q,v2) - cos(S(q),S(v2))| = {:.15}", (sim_q_v2_plain - sim_q_v2_scrambled).abs());

    // Should be EXACTLY equal (within floating point precision)
    assert_abs_diff_eq!(sim_q_v1_plain, sim_q_v1_scrambled, epsilon = 1e-12);
    assert_abs_diff_eq!(sim_q_v2_plain, sim_q_v2_scrambled, epsilon = 1e-12);

    println!("\n✅ Pure permutation preserves cosine similarity EXACTLY");
}

#[test]
fn test_pure_scrambling_100_percent_ranking_preservation() {
    println!("\n=== Pure Scrambling: 100% Ranking Preservation ===\n");

    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate a random permutation
    let mut permutation: Vec<usize> = (0..50).collect();
    permutation.shuffle(&mut rng);

    let num_trials = 1000;
    let mut successes = 0;

    for _trial in 0..num_trials {
        // Generate random unit vectors
        let v1 = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));
        let v2 = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));
        let q = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));

        // Plaintext similarities
        let s1_plain = cosine_similarity(&q, &v1);
        let s2_plain = cosine_similarity(&q, &v2);

        // Scrambled similarities (same permutation for all)
        let sq = pure_scramble(&q, &permutation);
        let sv1 = pure_scramble(&v1, &permutation);
        let sv2 = pure_scramble(&v2, &permutation);

        let s1_scrambled = cosine_similarity(&sq, &sv1);
        let s2_scrambled = cosine_similarity(&sq, &sv2);

        // Check ranking preservation
        let plain_order = s1_plain > s2_plain;
        let scrambled_order = s1_scrambled > s2_scrambled;

        if plain_order == scrambled_order {
            successes += 1;
        }

        // Also verify exact equality (within floating point precision)
        assert!((s1_plain - s1_scrambled).abs() < 1e-10,
                "Similarity should be exactly preserved, got diff: {:.15}",
                (s1_plain - s1_scrambled).abs());
    }

    let success_rate = (successes as f64) / (num_trials as f64);
    println!("Success rate: {}/{} ({:.1}%)", successes, num_trials, success_rate * 100.0);

    assert_eq!(successes, num_trials, "Pure permutation should give 100% ranking preservation");

    println!("\n✅ Pure dimensional scrambling (permutation only) gives 100% ranking preservation");
}

#[test]
fn test_mathematical_proof_permutation_preserves_dot_product() {
    println!("\n=== Mathematical Proof: Permutation Preserves Dot Product ===\n");

    let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let v2 = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let permutation = vec![4, 2, 0, 3, 1];

    println!("Original vectors:");
    println!("  v1 = {:?}", v1.as_slice().unwrap());
    println!("  v2 = {:?}", v2.as_slice().unwrap());
    println!("  Permutation π = {:?}", permutation);

    let dot_original = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<f64>();
    println!("\nOriginal dot product:");
    println!("  v1·v2 = Σ v1[i] × v2[i] = {:.6}", dot_original);

    let sv1 = pure_scramble(&v1, &permutation);
    let sv2 = pure_scramble(&v2, &permutation);

    println!("\nScrambled vectors:");
    println!("  S(v1) = {:?}", sv1.as_slice().unwrap());
    println!("  S(v2) = {:?}", sv2.as_slice().unwrap());

    let dot_scrambled = sv1.iter().zip(sv2.iter()).map(|(a, b)| a * b).sum::<f64>();
    println!("\nScrambled dot product:");
    println!("  S(v1)·S(v2) = Σ v1[π(i)] × v2[π(i)] = {:.6}", dot_scrambled);

    println!("\nExplanation:");
    println!("  When we use the SAME permutation π for both vectors:");
    println!("    S(v1)·S(v2) = Σ_i v1[π(i)] × v2[π(i)]");
    println!("                = Σ_j v1[j] × v2[j]  (just reordering the sum)");
    println!("                = v1·v2");
    println!("\n  Since dot product is a sum, reordering terms doesn't change the result!");

    assert_abs_diff_eq!(dot_original, dot_scrambled, epsilon = 1e-12);

    println!("\n✅ Permutation preserves dot product exactly (reordered sum = same sum)");
}

#[test]
fn test_comparison_current_vs_pure_scrambling() {
    println!("\n=== Comparison: Current Implementation vs Pure Scrambling ===\n");

    use phe::crypto::dimensional_scrambling::DimensionalScrambling;
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Test current implementation
    let ds_current = DimensionalScrambling::new(50, 20).unwrap();
    let mut current_successes = 0;
    let num_trials = 100;

    // Test pure scrambling
    let mut permutation: Vec<usize> = (0..50).collect();
    permutation.shuffle(&mut rng);
    let mut pure_successes = 0;

    for _trial in 0..num_trials {
        let v1 = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));
        let v2 = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));
        let q = normalize(&Array1::from_vec((0..50).map(|_| rng.gen_range(-10.0..10.0)).collect()));

        let s1_plain = cosine_similarity(&q, &v1);
        let s2_plain = cosine_similarity(&q, &v2);
        let plain_order = s1_plain > s2_plain;

        // Test current implementation (scaling + padding)
        let eq_current = ds_current.encrypt(&q).unwrap();
        let ev1_current = ds_current.encrypt(&v1).unwrap();
        let ev2_current = ds_current.encrypt(&v2).unwrap();
        let s1_current = cosine_similarity(&eq_current, &ev1_current);
        let s2_current = cosine_similarity(&eq_current, &ev2_current);
        if plain_order == (s1_current > s2_current) {
            current_successes += 1;
        }

        // Test pure scrambling
        let sq_pure = pure_scramble(&q, &permutation);
        let sv1_pure = pure_scramble(&v1, &permutation);
        let sv2_pure = pure_scramble(&v2, &permutation);
        let s1_pure = cosine_similarity(&sq_pure, &sv1_pure);
        let s2_pure = cosine_similarity(&sq_pure, &sv2_pure);
        if plain_order == (s1_pure > s2_pure) {
            pure_successes += 1;
        }
    }

    println!("Current implementation (scaling + padding):");
    println!("  Success rate: {}/{} ({:.1}%)", current_successes, num_trials,
             (current_successes as f64 / num_trials as f64) * 100.0);

    println!("\nPure dimensional scrambling (permutation only):");
    println!("  Success rate: {}/{} ({:.1}%)", pure_successes, num_trials,
             (pure_successes as f64 / num_trials as f64) * 100.0);

    assert_eq!(pure_successes, num_trials);

    println!("\n✅ Pure scrambling: 100% preservation");
    println!("❌ Current implementation: ~{}% preservation",
             (current_successes as f64 / num_trials as f64) * 100.0);
}
