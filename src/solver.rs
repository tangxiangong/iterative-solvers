//! A trait for iterative solvers.

use crate::{IterSolverError, IterSolverResult};
use nalgebra::{DMatrix, DVector};

/// A linear system `Ax = b` with a matrix `A` and a right-hand side vector `b`.
pub struct LinearSystem {
    mat: DMatrix<f64>,
    rhs: DVector<f64>,
}

impl LinearSystem {
    /// Create a new `LinearSystem` with a matrix `A` and a right-hand side vector `b`.
    ///
    /// # Arguments
    ///
    /// * `mat` - The matrix `A`.
    /// * `rhs` - The right-hand side vector `b`.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or if the matrix and the right-hand side vector do not match.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DMatrix, DVector};
    /// use iterative_solvers::LinearProblem;
    ///
    /// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    /// let rhs = DVector::from_vec(vec![1.0, 2.0]);
    /// let problem = LinearProblem::new(mat, rhs).unwrap();
    /// ```
    pub fn new(mat: DMatrix<f64>, rhs: DVector<f64>) -> IterSolverResult<Self> {
        if !mat.is_square() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix is not square, whose shape is ({}, {})",
                mat.shape().0,
                mat.shape().1
            )));
        }
        if mat.nrows() != rhs.len() {
            return Err(IterSolverError::DimensionError(format!(
                "The matrix with order {}, and the rhs with length {}, do not match",
                mat.nrows(),
                rhs.len()
            )));
        }
        Ok(Self { mat, rhs })
    }

    /// Solve the linear problem `Ax = b` using the given solver.
    ///
    /// # Arguments
    ///
    /// * `solver` - The solver to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the solver fails to solve the linear problem.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use nalgebra::{DMatrix, DVector};
    /// use iterative_solvers::{LinearProblem, CG};
    ///
    /// let mat = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    /// let rhs = DVector::from_vec(vec![1.0, 2.0]);
    /// let problem = LinearProblem::new(mat, rhs).unwrap();
    /// let solver = CG::new(&mat, &rhs, 1e-6).unwrap();
    /// let solution = problem.solve(solver).unwrap();
    /// ```
    pub fn solve(&self, solver: impl IterativeSolver) -> IterSolverResult<DVector<f64>> {
        solver.solve(&self.mat, &self.rhs)
    }
}

/// A trait for iterative solvers.
pub trait IterativeSolver {
    /// Solve the linear problem `Ax = b` using the given solver.
    ///
    /// # Arguments
    ///
    /// * `mat` - The matrix `A`.
    /// * `rhs` - The right-hand side vector `b`.
    ///
    /// # Returns
    ///
    /// A `DVector` containing the solution to the linear problem.
    fn solve(&self, mat: &DMatrix<f64>, rhs: &DVector<f64>) -> IterSolverResult<DVector<f64>>;
}
