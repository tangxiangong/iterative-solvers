use faer::{
    Conj, Mat,
    linalg::matmul::{dot::inner_prod, matmul},
    matrix_free::LinOp,
};

use crate::{IterSolverError, IterSolverResult, is_vector};

pub trait MatOp: LinOp<f64> {
    fn is_square(&self) -> bool;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn dot(&self, rhs: &Mat<f64>) -> IterSolverResult<f64>;

    /// self = alpha * x + beta * self
    fn axpy(&mut self, alpha: f64, x: &Mat<f64>, beta: f64);

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Mat<f64>, beta: f64, y: &mut Mat<f64>);
}

impl MatOp for Mat<f64> {
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    fn dot(&self, rhs: &Mat<f64>) -> IterSolverResult<f64> {
        if !is_vector(self) {
            return Err(IterSolverError::InvalidInput(
                "The input parameter is not a vector".to_string(),
            ));
        }
        if !is_vector(rhs) {
            return Err(IterSolverError::InvalidInput(
                "The input parameter is not a vector".to_string(),
            ));
        }
        if self.nrows() != rhs.nrows() {
            return Err(IterSolverError::InvalidInput(
                "The input parameter is not a vector".to_string(),
            ));
        }
        Ok(inner_prod(
            self.row(0),
            Conj::No,
            rhs.row(0).transpose(),
            Conj::No,
        ))
    }

    /// self = alpha * x + beta * self
    fn axpy(&mut self, alpha: f64, x: &Mat<f64>, beta: f64) {
        *self *= beta;
        *self = alpha * x + &*self;
    }

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Mat<f64>, beta: f64, y: &mut Mat<f64>) {
        *y *= beta;
        matmul(y, faer::Accum::Add, self, x, alpha, faer::Par::Seq);
    }
}
