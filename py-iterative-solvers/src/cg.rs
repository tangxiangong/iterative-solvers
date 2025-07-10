use iterative_solvers::cg::cg_with_initial_guess;
use nalgebra::DVector;
use nalgebra_sparse::CsrMatrix;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::spmatrix::PyCSRMatrix;

#[pyfunction]
pub fn cg_solver<'py>(
    py: Python<'py>,
    mat: &PyCSRMatrix,
    rhs: PyReadonlyArray1<'py, f64>,
    x0: PyReadonlyArray1<'py, f64>,
    abstol: f64,
    reltol: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let data_array = mat.data.bind(py).readonly();
    let indices_array = mat.indices.bind(py).readonly();
    let indptr_array = mat.indptr.bind(py).readonly();

    let data_slice = data_array.as_slice()?;
    let indices_slice = indices_array.as_slice()?;
    let indptr_slice = indptr_array.as_slice()?;
    let rhs_slice = rhs.as_slice()?;
    let x0_slice = x0.as_slice()?;

    let indices_usize = unsafe {
        std::slice::from_raw_parts(indices_slice.as_ptr() as *const usize, indices_slice.len())
    };
    let indptr_usize = unsafe {
        std::slice::from_raw_parts(indptr_slice.as_ptr() as *const usize, indptr_slice.len())
    };

    let mat_csr = CsrMatrix::try_from_csr_data(
        mat.shape.0,
        mat.shape.1,
        indptr_usize.to_vec(),
        indices_usize.to_vec(),
        data_slice.to_vec(),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CSR construction failed: {e}"))
    })?;

    let rhs_vec = DVector::from_column_slice(rhs_slice);
    let x0_vec = DVector::from_column_slice(x0_slice);

    let solver = py
        .allow_threads(|| cg_with_initial_guess(&mat_csr, &rhs_vec, x0_vec, abstol, reltol))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CG solver failed: {e}"))
        })?;

    let solution = solver.solution();
    Ok(PyArray1::from_vec(py, solution.data.as_vec().clone()))
}
