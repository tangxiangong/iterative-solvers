use faer::{
    Mat,
    linalg::matmul::matmul,
    matrix_free::LinOp,
    sparse::{SparseColMat, SparseRowMat, linalg::matmul::sparse_dense_matmul},
};

pub trait MatOp: LinOp<f64> {
    fn is_square(&self) -> bool;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

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

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Mat<f64>, beta: f64, y: &mut Mat<f64>) {
        *y *= beta;
        matmul(y, faer::Accum::Add, self, x, alpha, faer::Par::Seq);
    }
}

impl MatOp for SparseRowMat<usize, f64> {
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Mat<f64>, beta: f64, y: &mut Mat<f64>) {
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

impl MatOp for SparseColMat<usize, f64> {
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }

    fn len(&self) -> usize {
        self.nrows() * self.ncols()
    }

    /// y = alpha * self * x + beta * y
    fn gemv(&self, alpha: f64, x: &Mat<f64>, beta: f64, y: &mut Mat<f64>) {
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
