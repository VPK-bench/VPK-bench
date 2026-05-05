//! Vector operations utilities

use crate::error::{VPKError, VPKResult};
use ndarray::Array1;

/// Normalize a vector to unit length (L2 normalization)
///
/// # Arguments
/// * `vector` - The vector to normalize
///
/// # Returns
/// A new vector with L2 norm equal to 1.0
///
/// # Example
/// ```ignore
/// let v = Array1::from_vec(vec![3.0, 4.0]);
/// let normalized = normalize(&v).unwrap();
/// assert!((l2_norm(&normalized).unwrap() - 1.0).abs() < 1e-10);
/// ```
pub fn normalize(vector: &Array1<f64>) -> VPKResult<Array1<f64>> {
    let norm = l2_norm(vector)?;

    if norm < 1e-10 {
        return Err(VPKError::MatrixError(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    Ok(vector / norm)
}

/// Calculate the L2 (Euclidean) norm of a vector
///
/// # Arguments
/// * `vector` - The vector to calculate the norm for
///
/// # Returns
/// The L2 norm (sqrt of sum of squares)
pub fn l2_norm(vector: &Array1<f64>) -> VPKResult<f64> {
    let sum_squares: f64 = vector.iter().map(|x| x * x).sum();
    Ok(sum_squares.sqrt())
}

/// Calculate cosine similarity between two vectors
///
/// Cosine similarity = (v1 · v2) / (||v1|| * ||v2||)
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
/// Cosine similarity in range [-1.0, 1.0]
pub fn cosine_similarity(v1: &Array1<f64>, v2: &Array1<f64>) -> VPKResult<f64> {
    if v1.len() != v2.len() {
        return Err(VPKError::InvalidDimension {
            expected: v1.len(),
            actual: v2.len(),
        });
    }

    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = l2_norm(v1)?;
    let norm2 = l2_norm(v2)?;

    if norm1 < 1e-10 || norm2 < 1e-10 {
        return Err(VPKError::MatrixError(
            "Cannot compute cosine similarity with zero vector".to_string(),
        ));
    }

    Ok(dot_product / (norm1 * norm2))
}

/// Add zero padding to a vector
///
/// # Arguments
/// * `vector` - The original vector
/// * `padding_positions` - Positions where zeros should be inserted
/// * `total_dim` - Total dimension of the padded vector
///
/// # Returns
/// A new vector with zeros inserted at specified positions
pub fn zero_pad(
    vector: &Array1<f64>,
    padding_positions: &[usize],
    total_dim: usize,
) -> VPKResult<Array1<f64>> {
    if vector.len() + padding_positions.len() != total_dim {
        return Err(VPKError::PaddingError(format!(
            "Invalid padding: vector size {} + padding size {} != total dim {}",
            vector.len(),
            padding_positions.len(),
            total_dim
        )));
    }

    // Create result vector filled with zeros
    let mut result = Array1::zeros(total_dim);

    // Track position in original vector
    let mut orig_idx = 0;

    // Fill in values, skipping padding positions
    for i in 0..total_dim {
        if padding_positions.contains(&i) {
            // Leave as zero (already initialized)
            continue;
        }

        if orig_idx < vector.len() {
            result[i] = vector[orig_idx];
            orig_idx += 1;
        }
    }

    Ok(result)
}

/// Remove zero padding from a vector
///
/// # Arguments
/// * `padded` - The padded vector
/// * `padding_positions` - Positions where zeros were inserted
///
/// # Returns
/// The original vector with padding removed
pub fn remove_padding(padded: &Array1<f64>, padding_positions: &[usize]) -> VPKResult<Array1<f64>> {
    let orig_size = padded.len() - padding_positions.len();
    let mut result = Array1::zeros(orig_size);

    let mut result_idx = 0;
    for i in 0..padded.len() {
        if !padding_positions.contains(&i) {
            if result_idx < orig_size {
                result[result_idx] = padded[i];
                result_idx += 1;
            }
        }
    }

    Ok(result)
}

/// Calculate the dot product (inner product) of two vectors
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
///
/// # Returns
/// The dot product v1 · v2
pub fn dot_product(v1: &Array1<f64>, v2: &Array1<f64>) -> VPKResult<f64> {
    if v1.len() != v2.len() {
        return Err(VPKError::InvalidDimension {
            expected: v1.len(),
            actual: v2.len(),
        });
    }

    Ok(v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_l2_norm() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        let norm = l2_norm(&v).unwrap();
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        let normalized = normalize(&v).unwrap();
        let norm = l2_norm(&normalized).unwrap();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1e-10);

        // Check values
        assert_abs_diff_eq!(normalized[0], 0.6, epsilon = 1e-10);
        assert_abs_diff_eq!(normalized[1], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = Array1::from_vec(vec![0.0, 0.0]);
        let result = normalize(&v);
        assert!(result.is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        // Same vector should have similarity 1.0
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&v1, &v1).unwrap();
        assert_abs_diff_eq!(sim, 1.0, epsilon = 1e-10);

        // Orthogonal vectors should have similarity 0.0
        let v2 = Array1::from_vec(vec![1.0, 0.0]);
        let v3 = Array1::from_vec(vec![0.0, 1.0]);
        let sim = cosine_similarity(&v2, &v3).unwrap();
        assert_abs_diff_eq!(sim, 0.0, epsilon = 1e-10);

        // Opposite vectors should have similarity -1.0
        let v4 = Array1::from_vec(vec![1.0, 2.0]);
        let v5 = Array1::from_vec(vec![-1.0, -2.0]);
        let sim = cosine_similarity(&v4, &v5).unwrap();
        assert_abs_diff_eq!(sim, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let v1 = Array1::from_vec(vec![1.0, 2.0]);
        let v2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = cosine_similarity(&v1, &v2);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_pad() {
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let padding_positions = vec![1, 4]; // Insert zeros at positions 1 and 4
        let padded = zero_pad(&v, &padding_positions, 5).unwrap();

        // Expected: [1.0, 0.0, 2.0, 3.0, 0.0]
        assert_eq!(padded.len(), 5);
        assert_abs_diff_eq!(padded[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[2], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[3], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(padded[4], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_remove_padding() {
        let padded = Array1::from_vec(vec![1.0, 0.0, 2.0, 3.0, 0.0]);
        let padding_positions = vec![1, 4];
        let original = remove_padding(&padded, &padding_positions).unwrap();

        assert_eq!(original.len(), 3);
        assert_abs_diff_eq!(original[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(original[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(original[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_padding_roundtrip() {
        let original = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let padding_positions = vec![1, 3, 6];
        let total_dim = original.len() + padding_positions.len();

        let padded = zero_pad(&original, &padding_positions, total_dim).unwrap();
        let recovered = remove_padding(&padded, &padding_positions).unwrap();

        assert_eq!(recovered.len(), original.len());
        for i in 0..original.len() {
            assert_abs_diff_eq!(recovered[i], original[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dot_product() {
        let v1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let v2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let dot = dot_product(&v1, &v2).unwrap();

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_abs_diff_eq!(dot, 32.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_product_dimension_mismatch() {
        let v1 = Array1::from_vec(vec![1.0, 2.0]);
        let v2 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = dot_product(&v1, &v2);
        assert!(result.is_err());
    }
}
