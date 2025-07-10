use pyo3::prelude::*;

mod spmatrix;
use spmatrix::PyCSRMatrix;
mod cg;
use cg::cg_solver;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCSRMatrix>()?;
    m.add_function(wrap_pyfunction!(cg_solver, m)?)?;
    Ok(())
}
