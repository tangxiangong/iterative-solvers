use crate::{IterSolverResult, solver::IterativeSolver};
use nalgebra::{DMatrix, DVector};

pub struct CG<'mat> {
    mat: &'mat DMatrix<f64>,
    x: DVector<f64>,
    r: DVector<f64>,
    c: DVector<f64>,
    u: DVector<f64>,
    tol: f64,
    residual: f64,
    prev_residual: f64,
    iteration: usize,
}

impl<'mat> CG<'mat> {
    pub fn new(mat: &'mat DMatrix<f64>, rhs: &'mat DVector<f64>, tol: f64) -> Self {
        let n = mat.nrows();
        let x = DVector::zeros(n);
        let r = rhs.clone();
        let c = DVector::zeros(n);
        let u = DVector::zeros(n);
        let residual = r.norm();
        let prev_residual = residual;
        let iteration = 0;

        Self {
            mat,
            x,
            r,
            c,
            u,
            tol,
            residual,
            prev_residual,
            iteration,
        }
    }

    pub fn new_with_initial_guess(
        mat: &'mat DMatrix<f64>,
        rhs: &'mat DVector<f64>,
        tol: f64,
        initial_guess: DVector<f64>,
    ) -> Self {
        let n = mat.nrows();
        let x = initial_guess;
        let r = rhs - mat * &x;
        let c = DVector::zeros(n);
        let u = DVector::zeros(n);
        let residual = r.norm();
        let prev_residual = residual;
        let iteration = 0;

        Self {
            mat,
            x,
            r,
            c,
            u,
            tol,
            residual,
            prev_residual,
            iteration,
        }
    }

    #[inline]
    fn converged(&self) -> bool {
        self.residual <= self.tol
    }

    #[inline]
    fn done(&self) -> bool {
        (self.iteration >= self.max_iter()) || self.converged()
    }

    #[inline]
    fn max_iter(&self) -> usize {
        self.x.len()
    }

    pub fn iterate(&mut self) -> Option<f64> {
        if self.done() {
            return None;
        }

        // u = r + beta * u
        let beta = self.residual.powi(2) / self.prev_residual.powi(2);
        self.u.axpy(1.0, &self.r, beta);

        // c = A * u
        self.mat.mul_to(&self.u, &mut self.c);

        // update solution and residual
        // x = x + alpha * u
        // r = r - alpha * c
        let alpha = self.residual.powi(2) / self.u.dot(&self.c);
        self.x.axpy(alpha, &self.u, 1.0);
        self.r.axpy(-alpha, &self.c, 1.0);

        self.prev_residual = self.residual;
        self.residual = self.r.norm();

        self.iteration += 1;

        Some(self.residual)
    }
}

impl<'mat> Iterator for CG<'mat> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterate()
    }
}

pub fn cg<'mat>(mat: &'mat DMatrix<f64>, rhs: &'mat DVector<f64>, tol: f64) -> DVector<f64> {
    let mut iter = CG::new(mat, rhs, tol);
    let _: Vec<_> = iter.by_ref().collect();
    iter.x
}

impl<'mat> IterativeSolver<'mat> for CG<'mat> {
    fn solve(
        &self,
        mat: &'mat DMatrix<f64>,
        rhs: &'mat DVector<f64>,
    ) -> IterSolverResult<DVector<f64>> {
        Ok(cg(mat, rhs, self.tol))
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::utils::diag;

    #[test]
    fn test_cg() {
        let n = 1024;
        let h = 1.0 / 1024.0;
        let a = vec![2.0 / (h * h); n - 1];
        let b = vec![-1.0 / (h * h); n - 2];
        let mat = diag(&a, 0) + diag(&b, 1) + diag(&b, -1);
        let rhs: Vec<_> = (1..n)
            .map(|i| PI * PI * (i as f64 * h * PI).sin())
            .collect();
        let solution: Vec<_> = (1..n).map(|i| (i as f64 * h * PI).sin()).collect();
        let solution = DVector::from_vec(solution);
        let rhs = DVector::from_vec(rhs);
        let x = cg(&mat, &rhs, 1e-10);
        let e = (solution - x).norm();
        println!("error: {}", e);
    }
}
