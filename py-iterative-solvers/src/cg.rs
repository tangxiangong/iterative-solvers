use crate::spmatrix::PyCSRMatrix;
use iterative_solvers::cg::cg_with_initial_guess;
use nalgebra::DVector;
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
    // 使用极致优化版本进行CSR矩阵转换
    let mat_csr = mat.to_csr(py)?;

    // 获取原始slice引用，零拷贝
    let rhs_slice = rhs.as_slice()?;
    let x0_slice = x0.as_slice()?;

    // 直接从slice创建DVector，最小必要的拷贝
    let rhs_vec = DVector::from_row_slice(rhs_slice);
    let x0_vec = DVector::from_row_slice(x0_slice);

    // 执行CG求解，这里是纯Rust代码
    let solver =
        cg_with_initial_guess(&mat_csr, &rhs_vec, x0_vec, abstol, reltol).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CG solver failed: {e}"))
        })?;

    let solution = solver.solution();

    // 直接转换到Python数组，最小开销
    Ok(solution.as_slice().to_pyarray(py))
}
