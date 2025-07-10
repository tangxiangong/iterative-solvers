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

impl PyCSRMatrix {
    /// 极致性能优化版本：最佳性能的CSR矩阵转换
    pub fn to_csr(&self, py: Python<'_>) -> PyResult<CsrMatrix<f64>> {
        // 直接获取slice，避免复杂的指针操作
        let data_array = self.data.bind(py).readonly();
        let indices_array = self.indices.bind(py).readonly();
        let indptr_array = self.indptr.bind(py).readonly();

        let data_slice = data_array.as_slice()?;
        let indices_slice = indices_array.as_slice()?;
        let indptr_slice = indptr_array.as_slice()?;

        // 零拷贝data向量
        let data_vec = data_slice.to_vec();

        // 高性能类型转换
        let indices_len = indices_slice.len();
        let indptr_len = indptr_slice.len();

        let mut indices_usize = Vec::<usize>::with_capacity(indices_len);
        let mut indptr_usize = Vec::<usize>::with_capacity(indptr_len);

        unsafe {
            indices_usize.set_len(indices_len);
            indptr_usize.set_len(indptr_len);
        }

        let indices_out_ptr = indices_usize.as_mut_ptr();
        let indptr_out_ptr = indptr_usize.as_mut_ptr();

        // 使用最优化的类型转换循环
        for i in 0..indices_len {
            unsafe {
                *indices_out_ptr.add(i) = *indices_slice.get_unchecked(i) as usize;
            }
        }

        for i in 0..indptr_len {
            unsafe {
                *indptr_out_ptr.add(i) = *indptr_slice.get_unchecked(i) as usize;
            }
        }

        CsrMatrix::try_from_csr_data(
            self.shape.0,
            self.shape.1,
            indptr_usize,
            indices_usize,
            data_vec,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CSR construction failed: {e}"))
        })
    }
}
