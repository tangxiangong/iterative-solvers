use std::f64::consts::PI;

use iterative_solvers::{CG, utils::symmetric_tridiagonal};
use nalgebra::DVector;

fn main() {
    let n = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];
    let mat = symmetric_tridiagonal(&a, &b).unwrap();
    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    let solution: Vec<_> = (1..n).map(|i| (i as f64 * h * PI).sin()).collect();
    let solution = DVector::from_vec(solution);
    let rhs = DVector::from_vec(rhs);
    let mut solver = CG::new(&mat, &rhs, 1e-10).unwrap();
    for residual in &mut solver {
        println!("residual: {}", residual);
    }
    let result = solver.result();
    let e = (solution - result.solution()).norm();
    println!("error: {}", e);
}
