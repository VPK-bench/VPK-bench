//! Performance benchmarks for NeurIPS experiments
//!
//! Run with: `cargo bench`
//! Results are written to `target/criterion/`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;
use phe::crypto::{Ckks, DimensionalScrambling, NoiseInjection, Rome};
use rand::Rng;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn random_unit_vector(dim: usize) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let raw: Array1<f64> = Array1::from_iter((0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0));
    let norm = raw.dot(&raw).sqrt();
    if norm < 1e-12 { raw } else { raw / norm }
}

const DIMS: &[usize] = &[384, 768, 1536];

// ── Dimensional Scrambling ────────────────────────────────────────────────────

fn bench_scrambling_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("scrambling/encrypt");
    for &dim in DIMS {
        let ds = DimensionalScrambling::new(dim, 0).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| ds.encrypt(black_box(&vec)))
        });
    }
    group.finish();
}

// ── Noise Injection ───────────────────────────────────────────────────────────

fn bench_noise_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise/encrypt");
    for &dim in DIMS {
        let ni = NoiseInjection::new_default(dim).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| ni.inject_noise(black_box(&vec)))
        });
    }
    group.finish();
}

// ── ROME ──────────────────────────────────────────────────────────────────────

fn bench_rome_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("rome/encrypt");
    for &dim in &[384usize, 768] {
        let padded = (dim * 3) / 2;
        let rome = Rome::new(dim, padded).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| rome.encrypt(black_box(&vec)))
        });
    }
    group.finish();
}

fn bench_rome_padding(c: &mut Criterion) {
    let dim = 384;
    let mut group = c.benchmark_group("rome/padding_ratio");
    for &ratio_pct in &[125usize, 150, 200] {
        let padded = dim * ratio_pct / 100;
        let rome = Rome::new(dim, padded).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(ratio_pct), &ratio_pct, |b, _| {
            b.iter(|| rome.encrypt(black_box(&vec)))
        });
    }
    group.finish();
}

// ── CKKS ──────────────────────────────────────────────────────────────────────

fn bench_ckks_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("ckks/encrypt");
    for &dim in &[384usize, 768] {
        let padded = (dim * 3) / 2;
        let ckks = Ckks::new(dim, padded).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |b, _| {
            b.iter(|| ckks.encrypt(black_box(&vec)))
        });
    }
    group.finish();
}

fn bench_ckks_padding(c: &mut Criterion) {
    let dim = 384;
    let mut group = c.benchmark_group("ckks/padding_ratio");
    for &ratio_pct in &[125usize, 150, 200] {
        let padded = dim * ratio_pct / 100;
        let ckks = Ckks::new(dim, padded).unwrap();
        let vec = random_unit_vector(dim);
        group.bench_with_input(BenchmarkId::from_parameter(ratio_pct), &ratio_pct, |b, _| {
            b.iter(|| ckks.encrypt(black_box(&vec)))
        });
    }
    group.finish();
}

// ── CKKS vs CKKS+NI (full pipeline path) ─────────────────────────────────────
// Kept empty — use the standalone binary `ckks_ni_latency` instead:
//   cargo run --release --bin ckks_ni_latency

// ── Groups ────────────────────────────────────────────────────────────────────

criterion_group!(
    crypto_benches,
    bench_scrambling_encrypt,
    bench_noise_encrypt,
    bench_rome_encrypt,
    bench_rome_padding,
    bench_ckks_encrypt,
    bench_ckks_padding,
);
criterion_main!(crypto_benches);
