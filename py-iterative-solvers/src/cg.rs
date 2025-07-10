use crate::spmatrix::PyCSRMatrix;
use iterative_solvers::cg::cg_with_initial_guess;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

#[pyfunction]
pub fn cg_solver<'py>(
    py: Python<'py>,
    mat: &PyCSRMatrix,
    rhs: PyReadonlyArray1<'py, f64>,
    x0: PyReadonlyArray1<'py, f64>,
    abstol: f64,
    reltol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mat = CsrMatrix::try_from(mat)?;
    let rhs = DVector::from_vec(rhs.as_slice()?.to_vec());
    let x0 = DVector::from_vec(x0.as_slice()?.to_vec());
    let solver = cg_with_initial_guess(&mat, &rhs, x0, abstol, reltol)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let x = solver.solution();
    Ok(x.as_slice().to_pyarray(py))
}
