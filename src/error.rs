use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub enum IterSolverError {
    #[error("Dimension Not Match: {0}")]
    DimensionError(String),
}

pub type IterSolverResult<T> = Result<T, IterSolverError>;
