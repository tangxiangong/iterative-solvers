use crate::{CG, IterSolverError, IterSolverResult};
use nalgebra::{DMatrix, DVector};

pub struct LinearProblem {
    mat: DMatrix<f64>,
    rhs: DVector<f64>,
}

impl LinearProblem {
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

    pub fn solve(&self, solver: Solver) -> IterSolverResult<DVector<f64>> {
        match solver {
            Solver::ConjugateGradient(cg) => cg.solve(&self.mat, &self.rhs),
        }
    }
}

pub enum Solver<'mat> {
    ConjugateGradient(CG<'mat>),
}

pub(crate) trait IterativeSolver<'mat> {
    fn solve(
        &self,
        mat: &'mat DMatrix<f64>,
        rhs: &'mat DVector<f64>,
    ) -> IterSolverResult<DVector<f64>>;
}
