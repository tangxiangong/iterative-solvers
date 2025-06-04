//! Error and Result type for iterative solvers.

use thiserror::Error;

/// Error type for iterative solvers.
#[derive(Error, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum IterSolverError {
    /// Dimension Match Error of Matrices and Vectors.
    #[error("Dimension Not Match: {0}")]
    DimensionError(String),
}

/// Result type for iterative solvers.
pub type IterSolverResult<T> = Result<T, IterSolverError>;
