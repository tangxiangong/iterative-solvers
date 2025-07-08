//! Utility functions for creating dense matrices.

use crate::{IterSolverError, IterSolverResult};
use nalgebra::{DMatrix, DVector};

/// Creates a diagonal matrix with the given data placed on a specified diagonal.
///
/// This function constructs a matrix where the provided data is placed on a diagonal
/// that can be offset from the main diagonal. The resulting matrix size is determined
/// by the length of the data and the absolute value of the offset.
///
/// # Arguments
///
/// * `data` - A slice of f64 values to be placed on the diagonal
/// * `offset` - The diagonal offset:
///   - `0`: Main diagonal
///   - Positive: above the main diagonal (upper-diagonal)
///   - Negative: below the main diagonal (lower-diagonal)
///
/// # Examples
///
/// ```rust
/// use iterative_solvers::utils::dense::diagm;
///
/// // Main diagonal
/// let data = vec![1.0, 2.0, 3.0];
/// let mat = diagm(&data, 0);
/// // Creates:
/// // [1.0, 0.0, 0.0]
/// // [0.0, 2.0, 0.0]
/// // [0.0, 0.0, 3.0]
///
/// // Upper-diagonal (offset = 1)
/// let mat = diagm(&data, 1);
/// // Creates:
/// // [0.0, 1.0, 0.0, 0.0]
/// // [0.0, 0.0, 2.0, 0.0]
/// // [0.0, 0.0, 0.0, 3.0]
/// // [0.0, 0.0, 0.0, 0.0]
/// ```
pub fn diagm(data: &[f64], offset: i32) -> DMatrix<f64> {
    if data.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    match offset {
        0 => DMatrix::from_diagonal(&DVector::from_column_slice(data)),
        offset => {
            let offset_usize = offset.unsigned_abs() as usize;
            let n = data.len() + offset_usize;
            let mut mat = DMatrix::zeros(n, n);

            unsafe {
                if offset > 0 {
                    data.iter().enumerate().for_each(|(idx, &val)| {
                        *mat.get_unchecked_mut((idx, idx + offset_usize)) = val
                    });
                } else {
                    data.iter().enumerate().for_each(|(idx, &val)| {
                        *mat.get_unchecked_mut((idx + offset_usize, idx)) = val
                    });
                }
            }
            mat
        }
    }
}

/// Creates a tridiagonal matrix from diagonal, lower diagonal, and upper diagonal vectors.
///
/// This function constructs a tridiagonal matrix where:
/// - The main diagonal contains elements from the `diagonal` vector
/// - The lower-diagonal (below main) contains elements from the `lower` vector
/// - The upper-diagonal (above main) contains elements from the `upper` vector
///
/// # Arguments
///
/// * `diagonal` - A slice containing the main diagonal elements
/// * `lower` - A slice containing the lower diagonal elements (lower-diagonal)
/// * `upper` - A slice containing the upper diagonal elements (upper-diagonal)
///
/// # Dimension Requirements
///
/// For a valid tridiagonal matrix:
/// - `diagonal.len()` must equal `lower.len() + 1`
/// - `lower.len()` must equal `upper.len()`
///
/// This is because an n×n tridiagonal matrix has:
/// - n diagonal elements
/// - (n-1) lower-diagonal elements
/// - (n-1) upper-diagonal elements
///
/// # Examples
///
/// ```rust
/// use iterative_solvers::utils::dense::tridiagonal;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let lower = vec![1.0, 1.0];
/// let upper = vec![1.0, 1.0];
///
/// let result = tridiagonal(&diagonal, &lower, &upper).unwrap();
/// // Creates:
/// // [2.0, 1.0, 0.0]
/// // [1.0, 3.0, 1.0]
/// // [0.0, 1.0, 4.0]
/// ```
///
/// # Errors
///
/// Returns `IterSolverError::DimensionError` if the input vectors have incompatible dimensions.
pub fn tridiagonal(
    diagonal: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> IterSolverResult<DMatrix<f64>> {
    if diagonal.len() != lower.len() + 1 || lower.len() != upper.len() {
        return Err(IterSolverError::DimensionError(format!(
            "For tridiagonal matrix, the length of `diagonal` {}, the length of `lower` {} and `upper` {} do not match",
            diagonal.len(),
            lower.len(),
            upper.len()
        )));
    }
    Ok(diagm(diagonal, 0) + diagm(lower, -1) + diagm(upper, 1))
}

/// Creates a symmetric tridiagonal matrix from diagonal and sub-diagonal vectors.
///
/// This function constructs a symmetric tridiagonal matrix where the lower-diagonal
/// and upper-diagonal elements are identical. This is a common structure in numerical
/// methods, particularly for solving differential equations and eigenvalue problems.
///
/// # Arguments
///
/// * `diagonal` - A slice containing the main diagonal elements
/// * `sub_diagonal` - A slice containing the lower-diagonal elements, which will be
///   mirrored to create the upper-diagonal
///
/// # Dimension Requirements
///
/// For a valid symmetric tridiagonal matrix:
/// - `diagonal.len()` must equal `sub_diagonal.len() + 1`
///
/// # Examples
///
/// ```rust
/// use iterative_solvers::utils::dense::symmetric_tridiagonal;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let sub_diagonal = vec![1.0, 1.5];
///
/// let result = symmetric_tridiagonal(&diagonal, &sub_diagonal).unwrap();
/// // Creates:
/// // [2.0, 1.0, 0.0]
/// // [1.0, 3.0, 1.5]
/// // [0.0, 1.5, 4.0]
/// ```
///
/// # Errors
///
/// Returns `IterSolverError::DimensionError` if the input vectors have incompatible dimensions.
///
/// # Note
///
/// This function internally calls `tridiagonal(diagonal, sub_diagonal, sub_diagonal)`,
/// ensuring that the lower and upper diagonals are identical.
pub fn symmetric_tridiagonal(
    diagonal: &[f64],
    sub_diagonal: &[f64],
) -> IterSolverResult<DMatrix<f64>> {
    tridiagonal(diagonal, sub_diagonal, sub_diagonal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diag() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = diagm(&data, 0);
        println!("{mat:?}");
        let mat = diagm(&data, 1);
        println!("{mat:?}");
        let mat = diagm(&data, -1);
        println!("{mat:?}");
        let mat = diagm(&data, 2);
        println!("{mat:?}");
        let mat = diagm(&data, -2);
        println!("{mat:?}");
    }
}
