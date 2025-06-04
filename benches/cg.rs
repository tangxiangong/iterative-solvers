use criterion::{Criterion, criterion_group, criterion_main};
use iterative_solvers::{CG, utils::symmetric_tridiagonal};
use nalgebra::DVector;
use std::{f64::consts::PI, hint::black_box};

fn criterion_benchmark(c: &mut Criterion) {
    let n = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];
    let mat = symmetric_tridiagonal(&a, &b).unwrap();
    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    let rhs = DVector::from_vec(rhs);
    let abstol = 1e-10;
    let reltol = 1e-8;
    c.bench_function("cg", |b| {
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
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
