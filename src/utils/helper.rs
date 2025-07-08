#[cfg(feature = "faer")]
use faer::{unzip, zip};

use crate::{IterSolverError, IterSolverResult, Vector};

/// Check if the matrix is a vector.
pub fn is_vector(_mat: &Vector<f64>) -> bool {
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
pub fn dot(lhs: &Vector<f64>, rhs: &Vector<f64>) -> IterSolverResult<f64> {
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
pub fn axpy(v: &mut Vector<f64>, alpha: f64, x: &Vector<f64>, beta: f64) {
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
