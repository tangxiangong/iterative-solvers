use criterion::{Criterion, criterion_group, criterion_main};
use iterative_solvers::utils::{
    dense::symmetric_tridiagonal,
    sparse::{symmetric_tridiagonal_csc, symmetric_tridiagonal_csr},
};
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1024 * 8;
    let h = 1.0 / (n as f64);
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];

    c.bench_function("diagm-dense", |bencher| {
        bencher.iter(|| {
            symmetric_tridiagonal(black_box(&a), black_box(&b)).unwrap();
        })
    });

    c.bench_function("diagm-csc", |bencher| {
        bencher.iter(|| {
            symmetric_tridiagonal_csc(black_box(&a), black_box(&b)).unwrap();
        })
    });

    c.bench_function("diagm-csr", |bencher| {
        bencher.iter(|| {
            symmetric_tridiagonal_csr(black_box(&a), black_box(&b)).unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
