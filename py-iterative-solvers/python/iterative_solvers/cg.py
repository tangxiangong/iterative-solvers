import numpy as np
import scipy.sparse as sp
from ._core import cg_solver, PyCSRMatrix

Mat = sp.csr_array

Vec = np.ndarray[tuple[int], np.dtype[np.float64]]


def cg(
    mat: Mat,
    rhs: Vec,
    initial_guess: Vec | None = None,
    abstol: float = 1e-10,
    reltol: float = 1e-8,
):
    if initial_guess is None:
        initial_guess = np.zeros_like(rhs)

    # 确保数据类型正确
    data = mat.data.astype(np.float64)
    indices = mat.indices.astype(np.int64)
    indptr = mat.indptr.astype(np.int64)
    shape: tuple[int, int] = mat.shape  # type: ignore

    # 创建优化的 PyCSRMatrix 对象并调用求解器
    mat_ = PyCSRMatrix(data, indices, indptr, shape)
    return cg_solver(mat_, rhs, initial_guess, abstol, reltol)
