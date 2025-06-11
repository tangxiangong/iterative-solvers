use faer::{Conj, Mat, linalg::matmul::dot::inner_prod};

use crate::{IterSolverError, IterSolverResult};

pub fn is_vector(mat: &Mat<f64>) -> bool {
    mat.ncols() == 1
}

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
    Ok(inner_prod(
        lhs.row(0),
        Conj::No,
        rhs.row(0).transpose(),
        Conj::No,
    ))
}

/// self = alpha * x + beta * self
pub fn axpy(mat: &mut Mat<f64>, alpha: f64, x: &Mat<f64>, beta: f64) {
    *mat *= beta;
    *mat = alpha * x + &*mat;
}
