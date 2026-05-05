//! Number Theoretic Transform for polynomial arithmetic in Z_q[X]/(X^N + 1)
//!
//! Provides negacyclic NTT-based polynomial multiplication used by the true
//! CKKS FHE implementation.  All coefficient arithmetic is exact integer
//! arithmetic mod Q; no floating-point is used here.
//!
//! # Parameters
//!
//! - N = 1024  (ring dimension; demo security ~80-bit; production: 4096+)
//! - Q = 998_244_353  (NTT prime: 119 × 2^23 + 1)
//! - G = 3  (primitive root of Q, well-established)
//!
//! Primitive 2N-th root of unity: ψ = G^((Q-1)/(2N)) mod Q
//! Since Q-1 = 119 × 2^23 and 2N = 2^11, (Q-1)/(2N) = 119 × 2^12 is exact.

/// Ring dimension for CkksFhe (demo).  Production should be ≥ 4096.
pub const N: usize = 1024;
/// NTT prime: 119 × 2^23 + 1.  Primitive root G=3 is well-known.
pub const Q: i64 = 998_244_353;
const G: i64 = 3;

// ── Modular arithmetic helpers ────────────────────────────────────────────────

#[inline]
pub fn mod_q(x: i64) -> i64 {
    ((x % Q) + Q) % Q
}

/// Shift coefficient into the centered representative (−Q/2, Q/2].
#[inline]
pub fn center(x: i64) -> i64 {
    let r = mod_q(x);
    if r > Q / 2 { r - Q } else { r }
}

/// Modular exponentiation: base^exp mod modulus.
pub fn pow_mod(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base = mod_q_with(base, modulus);
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        exp >>= 1;
        base = mul_mod(base, base, modulus);
    }
    result
}

#[inline]
fn mod_q_with(x: i64, m: i64) -> i64 {
    ((x % m) + m) % m
}

/// Multiplication mod m using i128 to avoid overflow.
#[inline]
pub fn mul_mod(a: i64, b: i64, m: i64) -> i64 {
    ((a as i128 * b as i128).rem_euclid(m as i128)) as i64
}

// ── NTT primitives ────────────────────────────────────────────────────────────

fn bit_rev_permute(a: &mut Vec<i64>) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let rev = i.reverse_bits() >> (usize::BITS as usize - bits);
        if i < rev {
            a.swap(i, rev);
        }
    }
}

/// In-place Cooley-Tukey NTT over Z_Q.
/// `omega` must be a primitive N-th root of unity (or its inverse for INTT).
fn ntt_inplace(a: &mut Vec<i64>, omega: i64) {
    let n = a.len();
    bit_rev_permute(a);

    let mut len = 2usize;
    while len <= n {
        // Root for this level: ω^(N/len), which has order `len`
        let step = n / len;
        let w_len = pow_mod(omega, step as i64, Q);
        let mut i = 0;
        while i < n {
            let mut w = 1i64;
            for j in 0..len / 2 {
                let u = a[i + j];
                let v = mul_mod(a[i + j + len / 2], w, Q);
                a[i + j] = (u + v) % Q;
                a[i + j + len / 2] = (u - v + Q) % Q;
                w = mul_mod(w, w_len, Q);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Primitive N-th root of unity in Z_Q.
fn omega_n() -> i64 {
    pow_mod(G, (Q - 1) / N as i64, Q)
}

/// Primitive 2N-th root of unity ψ = G^((Q-1)/(2N)) mod Q.
pub fn psi_2n() -> i64 {
    pow_mod(G, (Q - 1) / (2 * N as i64), Q)
}

// ── Polynomial operations ─────────────────────────────────────────────────────

/// Negacyclic polynomial multiplication: a × b mod (X^N + 1, Q).
///
/// Uses the twist trick: multiply by ψ^i before NTT, ψ^(-i) after INTT,
/// so that the standard length-N NTT computes the negacyclic convolution.
pub fn poly_mul(a: &[i64], b: &[i64]) -> Vec<i64> {
    debug_assert_eq!(a.len(), N);
    debug_assert_eq!(b.len(), N);

    let psi = psi_2n();
    let psi_inv = pow_mod(psi, Q - 2, Q);

    // Precompute twist tables
    let mut psi_pow = vec![1i64; N];
    let mut psi_inv_pow = vec![1i64; N];
    for i in 1..N {
        psi_pow[i] = mul_mod(psi_pow[i - 1], psi, Q);
        psi_inv_pow[i] = mul_mod(psi_inv_pow[i - 1], psi_inv, Q);
    }

    let omega = omega_n();

    // Twist + forward NTT
    let mut at: Vec<i64> = (0..N).map(|i| mul_mod(a[i], psi_pow[i], Q)).collect();
    let mut bt: Vec<i64> = (0..N).map(|i| mul_mod(b[i], psi_pow[i], Q)).collect();
    ntt_inplace(&mut at, omega);
    ntt_inplace(&mut bt, omega);

    // Pointwise multiply
    let mut ct: Vec<i64> = (0..N).map(|i| mul_mod(at[i], bt[i], Q)).collect();

    // Inverse NTT + untwist
    let omega_inv = pow_mod(omega, Q - 2, Q);
    ntt_inplace(&mut ct, omega_inv);
    let n_inv = pow_mod(N as i64, Q - 2, Q);
    for i in 0..N {
        ct[i] = mul_mod(mul_mod(ct[i], n_inv, Q), psi_inv_pow[i], Q);
    }

    ct
}

/// Component-wise addition mod Q.
pub fn poly_add(a: &[i64], b: &[i64]) -> Vec<i64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) % Q).collect()
}

/// Component-wise subtraction mod Q (result in [0, Q)).
pub fn poly_sub(a: &[i64], b: &[i64]) -> Vec<i64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y + Q) % Q).collect()
}

/// Scalar multiplication mod Q.
pub fn poly_scalar_mul(a: &[i64], s: i64) -> Vec<i64> {
    let s = mod_q(s);
    a.iter().map(|&x| mul_mod(x, s, Q)).collect()
}

/// Center all coefficients into (−Q/2, Q/2].
pub fn poly_center(a: &[i64]) -> Vec<i64> {
    a.iter().map(|&x| center(x)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_add_sub_roundtrip() {
        let a: Vec<i64> = (0..N as i64).map(|i| i % Q).collect();
        let b: Vec<i64> = (0..N as i64).map(|i| (i * 3 + 7) % Q).collect();
        let sum = poly_add(&a, &b);
        let diff = poly_sub(&sum, &b);
        for i in 0..N {
            assert_eq!(diff[i], a[i] % Q);
        }
    }

    #[test]
    fn test_poly_mul_commutativity() {
        let a: Vec<i64> = (0..N as i64).map(|i| i % 100).collect();
        let b: Vec<i64> = (0..N as i64).map(|i| (N as i64 - i) % 100).collect();
        let ab = poly_mul(&a, &b);
        let ba = poly_mul(&b, &a);
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_poly_mul_identity() {
        // Multiplying by 1 (constant polynomial) should return the same polynomial
        let a: Vec<i64> = (0..N as i64).map(|i| i % 1000).collect();
        let mut one = vec![0i64; N];
        one[0] = 1;
        let result = poly_mul(&a, &one);
        assert_eq!(result, a);
    }

    #[test]
    fn test_negacyclic_property() {
        // X * X^(N-1) = X^N ≡ -1 (mod X^N + 1)
        // So X^N = -1 in the ring, meaning x[0]*y[0] picks up sign flip from wrap
        let mut x = vec![0i64; N]; // X
        x[1] = 1;
        let mut y = vec![0i64; N]; // X^(N-1)
        y[N - 1] = 1;
        let result = poly_mul(&x, &y); // Should be -1 = Q-1 at index 0
        assert_eq!(result[0], Q - 1, "X^N should equal -1 in negacyclic ring");
        for i in 1..N {
            assert_eq!(result[i], 0);
        }
    }

    #[test]
    fn test_psi_is_primitive_2n_root() {
        let psi = psi_2n();
        // ψ^(2N) = 1
        assert_eq!(pow_mod(psi, 2 * N as i64, Q), 1);
        // ψ^N = -1 (mod Q), i.e. Q-1
        assert_eq!(pow_mod(psi, N as i64, Q), Q - 1);
    }
}
