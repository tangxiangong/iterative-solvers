#[cfg(feature = "faer")]
use faer::{unzip, zip};

use crate::{IterSolverError, IterSolverResult, Matrix, SparseCscMatrix, SparseCsrMatrix, Vector};

/// Check if the matrix is a vector.
pub(crate) fn is_vector(_mat: &Vector<f64>) -> bool {
    #[cfg(feature = "nalgebra")]
    {
        true
    }
    #[cfg(feature = "faer")]
    {
        _mat.ncols() == 1
    }
}

/// Compute the dot product of two vectors.
pub(crate) fn dot(lhs: &Vector<f64>, rhs: &Vector<f64>) -> IterSolverResult<f64> {
    if !is_vector(lhs) {
        return Err(IterSolverError::InvalidInput(
            "The input parameter is not a vector".to_string(),
        ));
    }
    if !is_vector(rhs) {
        return Err(IterSolverError::InvalidInput(
            "The input parameter is not a vector".to_string(),
        ));
    }
    if lhs.nrows() != rhs.nrows() {
        return Err(IterSolverError::InvalidInput(
            "The input parameter is not a vector".to_string(),
        ));
    }

    // 计算两个列向量的点积

    #[cfg(feature = "faer")]
    {
        let mut result = 0.0;
        zip!(lhs, rhs).for_each(|unzip!(lhs_val, rhs_val)| {
            result += lhs_val * rhs_val;
        });
        Ok(result)
    }
    #[cfg(feature = "nalgebra")]
    {
        Ok(lhs.dot(rhs))
    }
}

/// self = alpha * x + beta * self
pub(crate) fn axpy(v: &mut Vector<f64>, alpha: f64, x: &Vector<f64>, beta: f64) {
    #[cfg(feature = "nalgebra")]
    {
        v.axpy(alpha, x, beta);
    }
    #[cfg(feature = "faer")]
    {
        if beta != 1.0 {
            *v *= beta;
        }
        if alpha != 0.0 {
            *v += alpha * x;
        }
    }
}

pub fn zeros(n: usize) -> Vector<f64> {
    #[cfg(feature = "nalgebra")]
    {
        Vector::zeros(n)
    }
    #[cfg(feature = "faer")]
    {
        Vector::zeros(n, 1)
    }
}

pub fn norm_l2(mat: &Vector<f64>) -> f64 {
    #[cfg(feature = "nalgebra")]
    {
        mat.norm()
    }
    #[cfg(feature = "faer")]
    {
        mat.norm_l2()
    }
}

pub(crate) fn from_diagonal(data: &[f64]) -> Matrix<f64> {
    #[cfg(feature = "faer")]
    {
        let n = data.len();
        let mut mat = Matrix::zeros(n, n);

        data.iter().enumerate().for_each(|(idx, &val)| unsafe {
            *mat.get_mut_unchecked(idx, idx) = val;
        });
        mat
    }
    #[cfg(feature = "nalgebra")]
    {
        Matrix::from_diagonal(&Vector::from_column_slice(data))
    }
}

/// # Safety
///
/// This function is unsafe because it does not check if the row and column indices are valid.
pub(crate) unsafe fn get_mut_unchecked<T>(mat: &mut Matrix<T>, row: usize, col: usize) -> &mut T {
    #[cfg(feature = "nalgebra")]
    {
        unsafe { mat.get_unchecked_mut((row, col)) }
    }
    #[cfg(feature = "faer")]
    {
        unsafe { mat.get_mut_unchecked(row, col) }
    }
}

pub(crate) fn empty_spcsr() -> SparseCsrMatrix<f64> {
    #[cfg(feature = "nalgebra")]
    {
        SparseCsrMatrix::zeros(0, 0)
    }
    #[cfg(feature = "faer")]
    {
        SparseCsrMatrix::try_new_from_triplets(0, 0, &[]).unwrap()
    }
}

pub(crate) fn empty_spcsc() -> SparseCscMatrix<f64> {
    #[cfg(feature = "nalgebra")]
    {
        SparseCscMatrix::zeros(0, 0)
    }
    #[cfg(feature = "faer")]
    {
        SparseCscMatrix::try_new_from_triplets(0, 0, &[]).unwrap()
    }
}
