//! Utility functions for creating sparse matrices.

use faer::sparse::{SparseColMat, SparseRowMat, Triplet};

use crate::{IterSolverError, IterSolverResult};

/// Creates a diagonal sparse CSR matrix with the given data placed on a specified diagonal.
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
/// # Returns
///
/// A `SparseRowMat<usize, f64>` containing the diagonal matrix. If `data` is empty, returns
/// a 0×0 matrix.
///
/// # Examples
///
/// ```rust
/// use iterative_solvers::utils::sparse::diagm_csr;
///
/// // Main diagonal
/// let data = vec![1.0, 2.0, 3.0];
/// let mat = diagm_csr(&data, 0);
/// // Creates:
/// // [1.0, 0.0, 0.0]
/// // [0.0, 2.0, 0.0]
/// // [0.0, 0.0, 3.0]
///
/// // Upper-diagonal (offset = 1)
/// let mat = diagm_csr(&data, 1);
/// // Creates:
/// // [0.0, 1.0, 0.0, 0.0]
/// // [0.0, 0.0, 2.0, 0.0]
/// // [0.0, 0.0, 0.0, 3.0]
/// // [0.0, 0.0, 0.0, 0.0]
/// ```
pub fn diagm_csr(data: &[f64], offset: i32) -> SparseRowMat<usize, f64> {
    if data.is_empty() {
        return SparseRowMat::try_new_from_triplets(0, 0, &[]).unwrap();
    }

    let offset_usize = offset.unsigned_abs() as usize;
    let n = data.len() + offset_usize;

    let triplets = match offset {
        0 => data
            .iter()
            .enumerate()
            .map(|(i, &val)| Triplet::new(i, i, val))
            .collect::<Vec<_>>(),
        offset => {
            if offset > 0 {
                data.iter()
                    .enumerate()
                    .map(|(i, &val)| Triplet::new(i, i + offset_usize, val))
                    .collect::<Vec<_>>()
            } else {
                data.iter()
                    .enumerate()
                    .map(|(i, &val)| Triplet::new(i + offset_usize, i, val))
                    .collect::<Vec<_>>()
            }
        }
    };
    SparseRowMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

/// Creates a tridiagonal sparse CSR matrix from diagonal, lower diagonal, and upper diagonal vectors.
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
/// use iterative_solvers::utils::sparse::tridiagonal_csr;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let lower = vec![1.0, 1.0];
/// let upper = vec![1.0, 1.0];
///
/// let result = tridiagonal_csr(&diagonal, &lower, &upper).unwrap();
/// // Creates:
/// // [2.0, 1.0, 0.0]
/// // [1.0, 3.0, 1.0]
/// // [0.0, 1.0, 4.0]
/// ```
///
/// # Errors
///
/// Returns `IterSolverError::DimensionError` if the input vectors have incompatible dimensions.
pub fn tridiagonal_csr(
    diagonal: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> IterSolverResult<SparseRowMat<usize, f64>> {
    if diagonal.len() != lower.len() + 1 || lower.len() != upper.len() {
        return Err(IterSolverError::DimensionError(format!(
            "For tridiagonal matrix, the length of `diagonal` {}, the length of `lower` {} and `upper` {} do not match",
            diagonal.len(),
            lower.len(),
            upper.len()
        )));
    }
    Ok(diagm_csr(diagonal, 0) + diagm_csr(lower, -1) + diagm_csr(upper, 1))
}

/// Creates a symmetric tridiagonal sparse CSR matrix from diagonal and sub-diagonal vectors.
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
/// use iterative_solvers::utils::sparse::symmetric_tridiagonal_csr;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let sub_diagonal = vec![1.0, 1.5];
///
/// let result = symmetric_tridiagonal_csr(&diagonal, &sub_diagonal).unwrap();
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
pub fn symmetric_tridiagonal_csr(
    diagonal: &[f64],
    sub_diagonal: &[f64],
) -> IterSolverResult<SparseRowMat<usize, f64>> {
    tridiagonal_csr(diagonal, sub_diagonal, sub_diagonal)
}

/// Creates a diagonal sparse CSC matrix with the given data placed on a specified diagonal.
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
/// # Returns
///
/// A `SparseColMat<usize, f64>` containing the diagonal matrix. If `data` is empty, returns
/// a 0×0 matrix.
///
/// # Examples
///
/// ```rust
/// use iterative_solvers::utils::sparse::diagm_csr;
///
/// // Main diagonal
/// let data = vec![1.0, 2.0, 3.0];
/// let mat = diagm_csr(&data, 0);
/// // Creates:
/// // [1.0, 0.0, 0.0]
/// // [0.0, 2.0, 0.0]
/// // [0.0, 0.0, 3.0]
///
/// // Upper-diagonal (offset = 1)
/// let mat = diagm_csr(&data, 1);
/// // Creates:
/// // [0.0, 1.0, 0.0, 0.0]
/// // [0.0, 0.0, 2.0, 0.0]
/// // [0.0, 0.0, 0.0, 3.0]
/// // [0.0, 0.0, 0.0, 0.0]
/// ```
pub fn diagm_csc(data: &[f64], offset: i32) -> SparseColMat<usize, f64> {
    if data.is_empty() {
        return SparseColMat::try_new_from_triplets(0, 0, &[]).unwrap();
    }

    let offset_usize = offset.unsigned_abs() as usize;
    let n = data.len() + offset_usize;

    let triplets = match offset {
        0 => data
            .iter()
            .enumerate()
            .map(|(i, &val)| Triplet::new(i, i, val))
            .collect::<Vec<_>>(),
        offset => {
            if offset > 0 {
                data.iter()
                    .enumerate()
                    .map(|(i, &val)| Triplet::new(i, i + offset_usize, val))
                    .collect::<Vec<_>>()
            } else {
                data.iter()
                    .enumerate()
                    .map(|(i, &val)| Triplet::new(i + offset_usize, i, val))
                    .collect::<Vec<_>>()
            }
        }
    };
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

/// Creates a tridiagonal sparse CSC matrix from diagonal, lower diagonal, and upper diagonal vectors.
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
/// use iterative_solvers::utils::sparse::tridiagonal_csc;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let lower = vec![1.0, 1.0];
/// let upper = vec![1.0, 1.0];
///
/// let result = tridiagonal_csc(&diagonal, &lower, &upper).unwrap();
/// // Creates:
/// // [2.0, 1.0, 0.0]
/// // [1.0, 3.0, 1.0]
/// // [0.0, 1.0, 4.0]
/// ```
///
/// # Errors
///
/// Returns `IterSolverError::DimensionError` if the input vectors have incompatible dimensions.
pub fn tridiagonal_csc(
    diagonal: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> IterSolverResult<SparseColMat<usize, f64>> {
    if diagonal.len() != lower.len() + 1 || lower.len() != upper.len() {
        return Err(IterSolverError::DimensionError(format!(
            "For tridiagonal matrix, the length of `diagonal` {}, the length of `lower` {} and `upper` {} do not match",
            diagonal.len(),
            lower.len(),
            upper.len()
        )));
    }
    Ok(diagm_csc(diagonal, 0) + diagm_csc(lower, -1) + diagm_csc(upper, 1))
}

/// Creates a symmetric tridiagonal sparse CSC matrix from diagonal and sub-diagonal vectors.
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
/// use iterative_solvers::utils::sparse::symmetric_tridiagonal_csr;
///
/// let diagonal = vec![2.0, 3.0, 4.0];
/// let sub_diagonal = vec![1.0, 1.5];
///
/// let result = symmetric_tridiagonal_csr(&diagonal, &sub_diagonal).unwrap();
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
pub fn symmetric_tridiagonal_csc(
    diagonal: &[f64],
    sub_diagonal: &[f64],
) -> IterSolverResult<SparseColMat<usize, f64>> {
    tridiagonal_csc(diagonal, sub_diagonal, sub_diagonal)
}

#[cfg(test)]
mod tests {
    use super::super::dense::diagm;
    use super::*;

    #[test]
    fn test_diagm_csr_main_diagonal() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = diagm_csr(&data, 0);

        // 转换为稠密矩阵进行验证
        let dense = mat.to_dense();
        let expected = diagm(&data, 0);

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_diagm_csr_upper_diagonal() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = diagm_csr(&data, 1);

        // 转换为稠密矩阵进行验证
        let dense = mat.to_dense();
        let expected = diagm(&data, 1);

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_diagm_csr_lower_diagonal() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = diagm_csr(&data, -1);

        // 转换为稠密矩阵进行验证
        let dense = mat.to_dense();
        let expected = diagm(&data, -1);

        assert_eq!(dense, expected);
    }

    #[test]
    fn test_diagm_csr_empty() {
        let data: Vec<f64> = vec![];
        let mat = diagm_csr(&data, 0);

        assert_eq!(mat.nrows(), 0);
        assert_eq!(mat.ncols(), 0);
    }

    #[test]
    fn test_diagm_csr_large_offset() {
        let data = vec![1.0, 2.0];
        let mat = diagm_csr(&data, 10);

        // 转换为稠密矩阵进行验证
        let dense = mat.to_dense();
        let expected = diagm(&data, 10);

        assert_eq!(dense, expected);
    }
}
