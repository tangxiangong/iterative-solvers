#[cfg(feature = "nalgebra")]
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "nalgebra")]
use nalgebra_sparse::{
    CscMatrix, CsrMatrix,
    ops::{
        Op,
        serial::{spmm_csc_dense, spmm_csr_dense},
    },
};

#[cfg(feature = "faer")]
use faer::{
    Mat,
    linalg::matmul::matmul,
    matrix_free::LinOp,
    sparse::{SparseColMat, SparseRowMat, linalg::matmul::sparse_dense_matmul},
};
#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2};

#[cfg(feature = "ndarray")]
use sprs::CsMat;

#[cfg(feature = "nalgebra")]
pub(crate) type Vector<T> = DVector<T>;

#[cfg(feature = "nalgebra")]
pub(crate) type Matrix<T> = DMatrix<T>;

#[cfg(feature = "nalgebra")]
pub(crate) type SparseCsrMatrix<T> = CsrMatrix<T>;

#[cfg(feature = "nalgebra")]
pub(crate) type SparseCscMatrix<T> = CscMatrix<T>;

#[cfg(feature = "faer")]
pub(crate) type Vector<T> = Mat<T>;

#[cfg(feature = "faer")]
pub(crate) type Matrix<T> = Mat<T>;

#[cfg(feature = "faer")]
pub(crate) type SparseCsrMatrix<T> = SparseRowMat<usize, T>;

#[cfg(feature = "faer")]
pub(crate) type SparseCscMatrix<T> = SparseColMat<usize, T>;

#[cfg(feature = "ndarray")]
pub(crate) type Vector<T> = Array1<T>;

#[cfg(feature = "ndarray")]
pub(crate) type Matrix<T> = Array2<T>;

/// A trait for matrix operations.
pub trait MatrixOp {
    /// Get the number of rows in the matrix.
    fn nrows(&self) -> usize;

    /// Get the number of columns in the matrix.
    fn ncols(&self) -> usize;

    /// Check if the matrix is square.
    fn is_square(&self) -> bool;

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Vector<f64>, beta: f64, y: &mut Vector<f64>);

    /// Check if the matrix is empty.
    fn is_empty(&self) -> bool;

    #[cfg(feature = "faer")]
    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }
}

impl MatrixOp for Matrix<f64> {
    fn nrows(&self) -> usize {
        self.nrows()
    }

    fn ncols(&self) -> usize {
        self.ncols()
    }
    fn is_square(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.is_square()
        }
        #[cfg(feature = "faer")]
        {
            self.nrows() == self.ncols()
        }
        #[cfg(feature = "ndarray")]
        {
            self.nrows() == self.ncols()
        }
    }

    fn gemv(&self, alpha: f64, x: &Vector<f64>, beta: f64, y: &mut Vector<f64>) {
        #[cfg(feature = "nalgebra")]
        {
            y.gemv(alpha, self, x, beta)
        }
        #[cfg(feature = "faer")]
        {
            *y *= beta;
            matmul(y, faer::Accum::Add, self, x, alpha, faer::Par::Seq);
        }
        #[cfg(feature = "ndarray")]
        {
            use ndarray::linalg::general_mat_vec_mul;

            general_mat_vec_mul(alpha, self, x, beta, y);
        }
    }

    fn is_empty(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.is_empty()
        }
        #[cfg(feature = "faer")]
        {
            self.nrows() == 0 || self.ncols() == 0
        }
        #[cfg(feature = "ndarray")]
        {
            self.nrows() == 0 || self.ncols() == 0
        }
    }
}

#[cfg(not(feature = "ndarray"))]
impl MatrixOp for SparseCsrMatrix<f64> {
    fn nrows(&self) -> usize {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self)
        }
    }

    fn ncols(&self) -> usize {
        #[cfg(feature = "nalgebra")]
        {
            self.ncols()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::ncols(self)
        }
    }

    fn is_empty(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows() == 0 || self.ncols() == 0
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self) == 0 || <Self as LinOp<f64>>::ncols(self) == 0
        }
    }

    fn is_square(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows() == self.ncols()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self) == <Self as LinOp<f64>>::ncols(self)
        }
    }

    fn gemv(&self, alpha: f64, x: &Vector<f64>, beta: f64, y: &mut Vector<f64>) {
        #[cfg(feature = "nalgebra")]
        {
            spmm_csr_dense(beta, y, alpha, Op::NoOp(self), Op::NoOp(x))
        }
        #[cfg(feature = "faer")]
        {
            *y *= beta;
            let col_mat = self.to_col_major().unwrap();
            sparse_dense_matmul(
                y.as_mut(),
                faer::Accum::Add,
                col_mat.as_ref(),
                x.as_ref(),
                alpha,
                faer::Par::Seq,
            );
        }
    }
}

#[cfg(not(feature = "ndarray"))]
impl MatrixOp for SparseCscMatrix<f64> {
    fn nrows(&self) -> usize {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self)
        }
    }

    fn ncols(&self) -> usize {
        #[cfg(feature = "nalgebra")]
        {
            self.ncols()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::ncols(self)
        }
    }

    fn is_empty(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows() == 0 || self.ncols() == 0
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self) == 0 || <Self as LinOp<f64>>::ncols(self) == 0
        }
    }

    fn is_square(&self) -> bool {
        #[cfg(feature = "nalgebra")]
        {
            self.nrows() == self.ncols()
        }
        #[cfg(feature = "faer")]
        {
            <Self as LinOp<f64>>::nrows(self) == <Self as LinOp<f64>>::ncols(self)
        }
    }

    fn gemv(&self, alpha: f64, x: &Vector<f64>, beta: f64, y: &mut Vector<f64>) {
        #[cfg(feature = "nalgebra")]
        {
            spmm_csc_dense(beta, y, alpha, Op::NoOp(self), Op::NoOp(x))
        }
        #[cfg(feature = "faer")]
        {
            *y *= beta;
            sparse_dense_matmul(
                y.as_mut(),
                faer::Accum::Add,
                self.as_ref(),
                x.as_ref(),
                alpha,
                faer::Par::Seq,
            );
        }
    }
}

#[cfg(feature = "ndarray")]
impl MatrixOp for CsMat<f64> {
    fn gemv(&self, alpha: f64, x: &Vector<f64>, beta: f64, y: &mut Vector<f64>) {
        *y *= beta;
        *y += &(alpha * (self * x));
    }

    fn nrows(&self) -> usize {
        self.rows()
    }

    fn ncols(&self) -> usize {
        self.cols()
    }

    fn is_empty(&self) -> bool {
        self.rows() == 0 || self.cols() == 0
    }

    fn is_square(&self) -> bool {
        self.rows() == self.cols()
    }
}
