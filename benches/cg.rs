use criterion::{Criterion, criterion_group, criterion_main};
use iterative_solvers::{CG, utils::dense::symmetric_tridiagonal};
#[cfg(feature = "nalgebra")]
use nalgebra::DVector;
#[cfg(feature = "nalgebra")]
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use std::{f64::consts::PI, hint::black_box};

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];
    let mat = symmetric_tridiagonal(&a, &b).unwrap();

    #[cfg(feature = "nalgebra")]
    let mat_csc = CscMatrix::from(&mat);
    #[cfg(feature = "nalgebra")]
    let mat_csr = CsrMatrix::from(&mat);

    #[cfg(feature = "nalgebra")]
    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    #[cfg(feature = "nalgebra")]
    let rhs = DVector::from_vec(rhs);

    #[cfg(feature = "faer")]
    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    #[cfg(feature = "faer")]
    let rhs = faer::Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);

    let abstol = 1e-10;
    let reltol = 1e-8;

    c.bench_function("cg", |b| {
        b.iter(|| {
            let mut solver = CG::new(
                black_box(&mat),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.next();
        })
    });

    #[cfg(feature = "nalgebra")]
    c.bench_function("cg-csc", |b| {
        b.iter(|| {
            let mut solver = CG::new(
                black_box(&mat_csc),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.next();
        })
    });

    #[cfg(feature = "nalgebra")]
    c.bench_function("cg-csr", |b| {
        b.iter(|| {
            let mut solver = CG::new(
                black_box(&mat_csr),
                black_box(&rhs),
                black_box(abstol),
                black_box(reltol),
            )
            .unwrap();
            let _ = solver.next();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
