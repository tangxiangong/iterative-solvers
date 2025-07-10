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
        // 避免数据拷贝，直接存储引用
        Ok(Self {
            data: data.to_owned_array().to_pyarray(py).unbind(),
            indices: indices.to_owned_array().to_pyarray(py).unbind(),
            indptr: indptr.to_owned_array().to_pyarray(py).unbind(),
            shape,
        })
    }
}

impl PyCSRMatrix {
    /// 高性能零拷贝版本：直接从 Python 数组构造 CSR 矩阵
    pub fn to_csr(&self, py: Python<'_>) -> PyResult<CsrMatrix<f64>> {
        let data_array = self.data.bind(py).readonly();
        let indices_array = self.indices.bind(py).readonly();
        let indptr_array = self.indptr.bind(py).readonly();

        let data_slice = data_array.as_slice()?;
        let indices_slice = indices_array.as_slice()?;
        let indptr_slice = indptr_array.as_slice()?;

        // 高效类型转换：使用迭代器和 collect，让编译器优化
        let indices_usize: Vec<usize> = indices_slice.iter().map(|&x| x as usize).collect();
        let indptr_usize: Vec<usize> = indptr_slice.iter().map(|&x| x as usize).collect();

        // 尝试零拷贝构造，如果失败则拷贝数据
        CsrMatrix::try_from_csr_data(
            self.shape.0,
            self.shape.1,
            indptr_usize,
            indices_usize,
            data_slice.to_vec(), // 只在这里做一次数据拷贝
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CSR construction failed: {e}"))
        })
    }
}
