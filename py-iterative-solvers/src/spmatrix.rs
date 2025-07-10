use nalgebra_sparse::CsrMatrix;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

#[pyclass]
pub struct PyCSRMatrix {
    pub data: Py<PyArray1<f64>>,
    pub indices: Py<PyArray1<i64>>,
    pub indptr: Py<PyArray1<i64>>,
    pub shape: (usize, usize),
}

#[pymethods]
impl PyCSRMatrix {
    #[new]
    pub fn new(
        py: Python<'_>,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<i64>,
        indptr: PyReadonlyArray1<i64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        Ok(Self {
            data: data.to_owned_array().to_pyarray(py).unbind(),
            indices: indices.to_owned_array().to_pyarray(py).unbind(),
            indptr: indptr.to_owned_array().to_pyarray(py).unbind(),
            shape,
        })
    }
}

impl TryFrom<&PyCSRMatrix> for CsrMatrix<f64> {
    type Error = PyErr;

    fn try_from(mat: &PyCSRMatrix) -> Result<Self, Self::Error> {
        let (data_vec, indices_vec, indptr_vec, shape) = Python::with_gil(|py| {
            let data_array = mat.data.bind(py);
            let indices_array = mat.indices.bind(py);
            let indptr_array = mat.indptr.bind(py);

            let data = data_array.readonly();
            let indices = indices_array.readonly();
            let indptr = indptr_array.readonly();

            Ok::<_, PyErr>((
                data.as_slice()?.to_vec(),
                indices.as_slice()?.to_vec(),
                indptr.as_slice()?.to_vec(),
                mat.shape,
            ))
        })?;

        let indices_usize: Vec<usize> = indices_vec.iter().map(|&x| x as usize).collect();
        let indptr_usize: Vec<usize> = indptr_vec.iter().map(|&x| x as usize).collect();

        let csr_matrix =
            CsrMatrix::try_from_csr_data(shape.0, shape.1, indptr_usize, indices_usize, data_vec)
                .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Failed to create CSR matrix: {e}"
                ))
            })?;

        Ok(csr_matrix)
    }
}
