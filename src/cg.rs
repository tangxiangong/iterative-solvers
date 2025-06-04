//! Conjugate Gradient (CG) method.

use crate::{IterSolverError, IterSolverResult};
use nalgebra::{DMatrix, DVector};

/// Conjugate Gradient (CG) method for solving linear systems Ax = b.
///
/// The Conjugate Gradient method is an iterative algorithm for solving systems of linear
/// equations where the coefficient matrix A is symmetric and positive definite. It is
/// particularly effective for large sparse systems.
///
/// # Algorithm Overview
///
/// The CG method generates a sequence of approximations to the solution by minimizing
/// the quadratic form associated with the linear system. At each iteration, it computes
/// a search direction that is conjugate to all previous search directions with respect
/// to the matrix A.
///
/// # Convergence
///
/// For a symmetric positive definite matrix, the CG method is guaranteed to converge
/// to the exact solution in at most n iterations (where n is the size of the matrix),
/// though in practice it often converges much faster, especially for well-conditioned
/// systems.
///
/// # Examples
///
/// ```rust
/// use nalgebra::{DMatrix, DVector};
/// use iterative_solvers::{CG, IterativeSolver};
///
/// // Create a simple 2x2 symmetric positive definite system
/// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
/// let rhs = DVector::from_vec(vec![1.0, 2.0]);
/// let tolerance = 1e-6;
///
/// let mut cg = CG::new(&mat, &rhs, tolerance);
/// let solution = cg.solve(&mat, &rhs).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CG<'mat> {
    mat: &'mat DMatrix<f64>,
    state: CGState,
    r: DVector<f64>,
    c: DVector<f64>,
    u: DVector<f64>,
    tol: f64,
    prev_residual: f64,
}

/// State of the CG method.
///
/// This struct contains the current solution, residual, and iteration count.
/// It is used to store the state of the CG method and to track the progress of the solution.
#[derive(Debug, Clone)]
pub struct CGState {
    solution: DVector<f64>,
    residual: f64,
    iteration: usize,
}

impl CGState {
    /// Create a new `CGState` with the given solution, residual, and iteration count.
    ///
    /// # Arguments
    ///
    /// * `solution` - The current solution vector.
    /// * `residual` - The residual of the current solution.
    /// * `iteration` - The current iteration count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::DVector;
    /// use iterative_solvers::CGState;
    ///
    /// let solution = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let residual = 1.0;
    /// let iteration = 0;
    /// let state = CGState::new(solution, residual, iteration);
    /// ```
    pub fn new(solution: DVector<f64>, residual: f64, iteration: usize) -> Self {
        Self {
            solution,
            residual,
            iteration,
        }
    }

    /// Get the current solution vector.
    pub fn solution(&self) -> &DVector<f64> {
        &self.solution
    }

    /// Get the residual of the current solution.
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

impl<'mat> CG<'mat> {
    /// Create a new `CG` solver with the given matrix, right-hand side, and tolerance.
    ///
    /// # Arguments
    ///
    /// * `mat` - The coefficient matrix A.
    /// * `rhs` - The right-hand side vector b.
    /// * `tol` - The tolerance for the residual.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DMatrix, DVector};
    /// use iterative_solvers::CG;
    ///
    /// // Create a simple 2x2 symmetric positive definite system
    /// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    /// let rhs = DVector::from_vec(vec![1.0, 2.0]);
    /// let tolerance = 1e-6;
    ///
    /// let mut cg = CG::new(&mat, &rhs, tolerance);
    /// let solution = cg.solve(&mat, &rhs).unwrap();
    /// ```
    pub fn new(
        mat: &'mat DMatrix<f64>,
        rhs: &'mat DVector<f64>,
        tol: f64,
    ) -> IterSolverResult<Self> {
        if !mat.is_square() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix is not square, whose shape is ({}, {})",
                mat.nrows(),
                mat.ncols()
            )));
        }
        if mat.nrows() != rhs.len() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix with order {}, and the rhs with length {}, do not match",
                mat.nrows(),
                rhs.len()
            )));
        }
        let n = mat.nrows();
        let x = DVector::zeros(n);
        let r = rhs.clone();
        let c = DVector::zeros(n);
        let u = DVector::zeros(n);
        let residual = r.norm();
        let prev_residual = residual;
        let iteration = 0;
        let state = CGState::new(x, residual, iteration);
        Ok(Self {
            mat,
            state,
            r,
            c,
            u,
            tol,
            prev_residual,
        })
    }

    /// Create a new `CG` solver with the given matrix, right-hand side, tolerance, and initial guess.
    ///
    /// # Arguments
    ///
    /// * `mat` - The coefficient matrix A.
    /// * `rhs` - The right-hand side vector b.
    /// * `tol` - The tolerance for the residual.
    /// * `initial_guess` - The initial guess for the solution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DMatrix, DVector};
    /// use iterative_solvers::CG;
    ///
    /// // Create a simple 2x2 symmetric positive definite system
    /// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    /// let rhs = DVector::from_vec(vec![1.0, 2.0]);
    /// let tolerance = 1e-6;
    /// let initial_guess = DVector::from_vec(vec![0.0, 0.0]);
    ///
    /// let mut cg = CG::new_with_initial_guess(&mat, &rhs, tolerance, initial_guess);
    /// let solution = cg.result();
    /// ```
    pub fn new_with_initial_guess(
        mat: &'mat DMatrix<f64>,
        rhs: &'mat DVector<f64>,
        tol: f64,
        initial_guess: DVector<f64>,
    ) -> IterSolverResult<Self> {
        if !mat.is_square() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix is not square, whose shape is ({}, {})",
                mat.nrows(),
                mat.ncols()
            )));
        }
        if mat.nrows() != rhs.len() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix with order {}, and the rhs with length {}, do not match",
                mat.nrows(),
                rhs.len()
            )));
        }
        if initial_guess.len() != mat.nrows() {
            return Err(IterSolverError::DimensionError(format!(
                "The initial guess with length {}, and the matrix with order {}, do not match",
                initial_guess.len(),
                mat.nrows()
            )));
        }
        let n = mat.nrows();
        let x = initial_guess;
        let r = rhs - mat * &x;
        let c = DVector::zeros(n);
        let u = DVector::zeros(n);
        let residual = r.norm();
        let prev_residual = residual;
        let iteration = 0;
        let state = CGState::new(x, residual, iteration);

        Ok(Self {
            mat,
            state,
            r,
            c,
            u,
            tol,
            prev_residual,
        })
    }

    /// Check if the solver has converged.
    #[inline]
    fn converged(&self) -> bool {
        self.state.residual <= self.tol
    }

    /// Check if the solver has reached the maximum number of iterations.
    #[inline]
    fn done(&self) -> bool {
        (self.state.iteration >= self.max_iter()) || self.converged()
    }

    /// Get the maximum number of iterations.
    #[inline]
    fn max_iter(&self) -> usize {
        self.state.solution.len()
    }

    /// Consume the solver and return the result.
    pub fn result(mut self) -> CGState {
        self.by_ref().count();
        self.state
    }
}

impl<'mat> Iterator for CG<'mat> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done() {
            return None;
        }

        // u = r + beta * u
        let beta = self.state.residual.powi(2) / self.prev_residual.powi(2);
        self.u.axpy(1.0, &self.r, beta);

        // c = A * u
        self.mat.mul_to(&self.u, &mut self.c);

        // update solution and residual
        // x = x + alpha * u
        // r = r - alpha * c
        let alpha = self.state.residual.powi(2) / self.u.dot(&self.c);
        self.state.solution.axpy(alpha, &self.u, 1.0);
        self.r.axpy(-alpha, &self.c, 1.0);

        self.prev_residual = self.state.residual;
        self.state.residual = self.r.norm();

        self.state.iteration += 1;

        Some(self.state.residual)
    }
}

/// Solve a linear system Ax = b using the Conjugate Gradient method.
///
/// # Arguments
///
/// * `mat` - The coefficient matrix A.
/// * `rhs` - The right-hand side vector b.
/// * `tol` - The tolerance for the residual.
///
/// # Examples
///
/// ```rust
/// use nalgebra::{DMatrix, DVector};
/// use iterative_solvers::cg;
///
/// // Create a simple 2x2 symmetric positive definite system
/// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
/// let rhs = DVector::from_vec(vec![1.0, 2.0]);
/// let tolerance = 1e-6;
///
/// let solution = cg(&mat, &rhs, tolerance).unwrap();
/// ```
pub fn cg<'mat>(
    mat: &'mat DMatrix<f64>,
    rhs: &'mat DVector<f64>,
    tol: f64,
) -> IterSolverResult<CGState> {
    let mut solver = CG::new(mat, rhs, tol)?;
    solver.by_ref().count();
    Ok(solver.state)
}

/// Solve a linear system Ax = b using the Conjugate Gradient method.
///
/// # Arguments
///
/// * `mat` - The coefficient matrix A.
/// * `rhs` - The right-hand side vector b.
/// * `tol` - The tolerance for the residual.
/// * `initial_guess` - The initial guess for the solution.
///
/// # Examples
///
/// ```rust
/// use nalgebra::{DMatrix, DVector};
/// use iterative_solvers::cg_with_initial_guess;
///
/// // Create a simple 2x2 symmetric positive definite system
/// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
/// let rhs = DVector::from_vec(vec![1.0, 2.0]);
/// let tolerance = 1e-6;
/// let initial_guess = DVector::from_vec(vec![0.0, 0.0]);
///
/// let solution = cg_with_initial_guess(&mat, &rhs, tolerance, initial_guess).unwrap();
/// ```
pub fn cg_with_initial_guess<'mat>(
    mat: &'mat DMatrix<f64>,
    rhs: &'mat DVector<f64>,
    tol: f64,
    initial_guess: DVector<f64>,
) -> IterSolverResult<CGState> {
    let mut solver = CG::new_with_initial_guess(mat, rhs, tol, initial_guess)?;
    solver.by_ref().count();
    Ok(solver.state)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::utils::symmetric_tridiagonal;

    #[test]
    fn test_cg() {
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
        let state = cg(&mat, &rhs, 1e-10).unwrap();
        let e = (solution - state.solution()).norm();
        assert!(e < 1e-4);
    }
}
