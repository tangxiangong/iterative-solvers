import numpy

def hello_from_bin() -> str: ...

class PyCSRMatrix:
    data: numpy.ndarray
    indices: numpy.ndarray
    indptr: numpy.ndarray
    shape: tuple[int, int]

    def __init__(
        self,
        data: numpy.ndarray,
        indices: numpy.ndarray,
        indptr: numpy.ndarray,
        shape: tuple[int, int],
    ): ...

def cg_solver(
    mat: PyCSRMatrix,
    rhs: numpy.ndarray,
    x0: numpy.ndarray,
    abstol: float,
    reltol: float,
) -> numpy.ndarray: ...
