use criterion::{Criterion, criterion_group, criterion_main};
use iterative_solvers::{
    CG,
    utils::{
        dense::symmetric_tridiagonal,
        sparse::{symmetric_tridiagonal_csc, symmetric_tridiagonal_csr},
    },
};
use std::{f64::consts::PI, hint::black_box};

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1024 * 8;
    let h = 1.0 / (n as f64);
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];
    let mat = symmetric_tridiagonal(&a, &b).unwrap();
    let mat_csc = symmetric_tridiagonal_csc(&a, &b).unwrap();
    let mat_csr = symmetric_tridiagonal_csr(&a, &b).unwrap();

    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    #[cfg(feature = "nalgebra")]
    let rhs = nalgebra::DVector::from_vec(rhs);

    #[cfg(feature = "faer")]
    let rhs = faer::Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);

    #[cfg(feature = "ndarray")]
    let rhs = ndarray::Array1::from_vec(rhs);

    let abstol = 1e-10;
    let reltol = 1e-8;

    c.bench_function("cg-dense", |b| {
        b.iter(|| {
            let solver = CG::new(
                black_box(&mat),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.solve();
        })
    });

    c.bench_function("cg-csc", |b| {
        b.iter(|| {
            let solver = CG::new(
                black_box(&mat_csc),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.solve();
        })
    });

    c.bench_function("cg-csr", |b| {
        b.iter(|| {
            let solver = CG::new(
                black_box(&mat_csr),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.solve();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
