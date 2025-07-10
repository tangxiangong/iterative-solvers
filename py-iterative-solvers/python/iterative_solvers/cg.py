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
    indices = mat.indices.astype(np.int64)
    indptr = mat.indptr.astype(np.int64)
    data = mat.data.astype(np.float64)
    shape = mat.shape
    mat_ = PyCSRMatrix(data, indices, indptr, shape)  # type: ignore
    return cg_solver(mat_, rhs, initial_guess, abstol, reltol)
