use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{
    CscMatrix, CsrMatrix,
    ops::{
        Op,
        serial::{spmm_csc_dense, spmm_csr_dense},
    },
};

/// A trait for matrix operations.
pub trait MatrixOp {
    /// Get the number of rows in the matrix.
    fn nrows(&self) -> usize;

    /// Get the number of columns in the matrix.
    fn ncols(&self) -> usize;

    /// Check if the matrix is square.
    fn is_square(&self) -> bool;

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &DVector<f64>, beta: f64, y: &mut DVector<f64>);

    /// Check if the matrix is empty.
    fn is_empty(&self) -> bool;
}

impl MatrixOp for DMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn is_square(&self) -> bool {
        self.is_square()
    }

    fn gemv(&self, alpha: f64, x: &DVector<f64>, beta: f64, y: &mut DVector<f64>) {
        y.gemv(alpha, self, x, beta)
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl MatrixOp for CsrMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0 || self.ncols() == 0
    }

    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn gemv(&self, alpha: f64, x: &DVector<f64>, beta: f64, y: &mut DVector<f64>) {
        spmm_csr_dense(beta, y, alpha, Op::NoOp(self), Op::NoOp(x))
    }
}

impl MatrixOp for CscMatrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }

    fn is_empty(&self) -> bool {
        self.nrows() == 0 || self.ncols() == 0
    }

    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn gemv(&self, alpha: f64, x: &DVector<f64>, beta: f64, y: &mut DVector<f64>) {
        spmm_csc_dense(beta, y, alpha, Op::NoOp(self), Op::NoOp(x))
    }
}
