#![warn(clippy::pedantic)]
#![warn(unused_results)]

use core::f32::consts::PI;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rand::{RngExt, rng};
//use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::hint::black_box; // Use core::f32 for no_std

use vqm::{cos_approx_f32, sin_approx_f32, sin_cos_approx_f32};

/*
#[cfg(feature = "std")]
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[cfg(all(not(feature = "std"), not(feature = "libm")))]
*/

// See: target/criterion/Matrix%20Math/report/index.html for results

#[allow(unused)]
fn bench_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("Math");

    _ = group.throughput(Throughput::Elements(1));

    _ = group.bench_function("sin", |b| {
        #[cfg(feature = "libm")]
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| libm::sinf(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("sin_approx", |b| {
        b.iter_batched(
            || rng().random_range(-PI / 2.0..PI / 2.0),
            |x| sin_approx_f32(black_box(x)),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("cos", |b| {
        #[cfg(feature = "libm")]
        b.iter_batched(|| rng().random_range(-PI / 2.0..PI / 2.0), |x| libm::cosf(black_box(x)), BatchSize::SmallInput);
    });

    _ = group.bench_function("cos_approx", |b| {
        b.iter_batched(
            || rng().random_range(-PI / 2.0..PI / 2.0),
            |x| cos_approx_f32(black_box(x)),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("sin_cos", |b| {
        b.iter_batched(
            || rng().random_range(-PI / 2.0..PI / 2.0),
            #[cfg(not(feature = "libm"))]
            |x| (black_box(x)).sin_cos(),
            #[cfg(feature = "libm")]
            |x| libm::sincosf(black_box(x)),
            BatchSize::SmallInput,
        );
    });

    _ = group.bench_function("sin_cos_approx", |b| {
        b.iter_batched(
            || rng().random_range(-PI / 2.0..PI / 2.0),
            |x| sin_cos_approx_f32(black_box(x)),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_math);
criterion_main!(benches);
