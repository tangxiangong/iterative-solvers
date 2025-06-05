use criterion::{Criterion, criterion_group, criterion_main};
use iterative_solvers::utils::dense::diagm;
use std::hint::black_box;

fn criterion_benchmark(c: &mut Criterion) {
    // 测试不同大小的数据
    let sizes = [100, 1000, 5000];

    for &size in &sizes {
        let diagonal = vec![2.0; size];
        let sub_diagonal = vec![-1.0; size - 1];

        let group_name = format!("diag_size_{}", size);
        let mut group = c.benchmark_group(&group_name);

        group.bench_function("diag", |b| {
            b.iter(|| {
                diagm(black_box(&diagonal), black_box(0));
                diagm(black_box(&sub_diagonal), black_box(1));
                diagm(black_box(&sub_diagonal), black_box(-1));
            })
        });

        group.finish();
    }

    // 单独测试对角线情况（offset=0）
    let large_data = vec![1.0; 10000];
    let mut group = c.benchmark_group("diagonal_only");

    group.bench_function("diag_diagonal", |b| {
        b.iter(|| diagm(black_box(&large_data), black_box(0)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
