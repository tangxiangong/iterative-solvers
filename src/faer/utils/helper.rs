use faer::{Mat, unzip, zip};

use crate::{IterSolverError, IterSolverResult};

/// Check if the matrix is a vector.
pub fn is_vector(mat: &Mat<f64>) -> bool {
    mat.ncols() == 1
}

/// Compute the dot product of two vectors.
pub fn dot(lhs: &Mat<f64>, rhs: &Mat<f64>) -> IterSolverResult<f64> {
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
    let mut result = 0.0;
    zip!(lhs, rhs).for_each(|unzip!(lhs_val, rhs_val)| {
        result += lhs_val * rhs_val;
    });
    Ok(result)
}

/// self = alpha * x + beta * self
pub fn axpy(mat: &mut Mat<f64>, alpha: f64, x: &Mat<f64>, beta: f64) {
    if beta != 1.0 {
        *mat *= beta;
    }
    if alpha != 0.0 {
        *mat += alpha * x;
    }
}
