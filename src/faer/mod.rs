pub mod cg;
pub use cg::*;
pub mod ops;
pub use ops::*;

pub mod utils;

pub(crate) fn is_vector(mat: &faer::Mat<f64>) -> bool {
    mat.ncols() == 1
}
