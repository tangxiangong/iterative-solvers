import numpy as np
import scipy.sparse as sp

Mat = np.ndarray[tuple[int, int], np.dtype[np.float64]] | sp.csr_matrix | sp.csc_matrix

Vec = np.ndarray[tuple[int], np.dtype[np.float64]]


class CG:
    """Conjugate Gradient (CG) iterative solver for symmetric positive definite linear systems.

    The Conjugate Gradient method is an iterative algorithm for solving systems of linear
    equations Ax = b where the coefficient matrix A is symmetric and positive definite.
    It is particularly effective for large sparse systems.

    Parameters
    ----------
    mat : ndarray or sparse matrix
        Coefficient matrix A. Must be square, symmetric, and positive definite.
        Supported types: numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix.
    rhs : ndarray
        Right-hand side vector b.
    initial_guess : ndarray, optional
        Initial guess for the solution. If None, zero vector is used.
        Default is None.
    abstol : float, optional
        Absolute tolerance for convergence. Default is 1e-10.
    reltol : float, optional
        Relative tolerance for convergence. Default is 1e-8.

    Attributes
    ----------
    mat : Mat
        The coefficient matrix.
    rhs : Vec
        The right-hand side vector.
    solution : Vec
        Current solution estimate.
    residual : float
        Current residual norm.
    iteration : int
        Current iteration number.
    tol : float
        Convergence tolerance (max of abstol and reltol * initial_residual).

    Notes
    -----
    The stopping criterion is ||r_k|| ≤ max(abstol, reltol * ||r_0||),
    where r_k is the residual at iteration k and r_0 is the initial residual.

    The CG method is guaranteed to converge in at most n iterations for an
    n×n symmetric positive definite matrix, though it often converges much
    faster in practice.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sp
    >>> from cg import CG
    >>>
    >>> # Create a symmetric positive definite system
    >>> n = 100
    >>> A = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n))
    >>> b = np.random.rand(n)
    >>>
    >>> # Solve using CG
    >>> cg = CG(A, b)
    >>> for iteration, residual, solution in cg:
    ...     if iteration % 10 == 0:
    ...         print(f"Iteration {iteration}, residual: {residual:.2e}")
    """

    def __init__(
        self,
        mat: Mat,
        rhs: Vec,
        initial_guess: Vec | None = None,
        abstol: float = 1e-10,
        reltol: float = 1e-8,
    ):
        if mat.shape[0] != mat.shape[1]:  # type: ignore
            raise ValueError(
                "The matrix is not square, whose shape is ({}, {})",
                mat.shape[0],  # type: ignore
                mat.shape[1],  # type: ignore
            )

        if mat.shape[0] != rhs.shape[0]:  # type: ignore
            raise ValueError(
                "The square matrix with order {}, and the rhs with length {}, do not match",
                mat.shape[0],  # type: ignore
                rhs.shape[0],
            )

        if initial_guess is not None:
            if initial_guess.shape[0] != mat.shape[0]:  # type: ignore
                raise ValueError(
                    "The initial guess with length {}, does not match the matrix with order {}",
                )

        self.mat = mat
        self.rhs = rhs

        if initial_guess is None:
            self.solution = np.zeros_like(rhs)
            self.r = rhs.copy()
        else:
            self.solution = initial_guess
            self.r = rhs - mat.dot(initial_guess)

        self.c = np.zeros_like(rhs)
        self.u = np.zeros_like(rhs)
        self.residual = float(np.linalg.norm(self.r))
        self.prev_residual = self.residual
        self.iteration = 0
        self.tol = max(abstol, reltol * self.residual)

    def converged(self) -> bool:
        """Check if the solver has converged.

        Returns
        -------
        bool
            True if the current residual is below the tolerance threshold.
        """
        return self.residual <= self.tol

    def done(self) -> bool:
        """Check if the iteration should terminate.

        Returns
        -------
        bool
            True if either converged or maximum iterations reached.
        """
        return self.iteration >= self.max_iter() or self.converged()

    def max_iter(self) -> int:
        """Get the maximum number of iterations.

        Returns
        -------
        int
            Maximum number of iterations (equal to problem size).
        """
        return self.solution.shape[0]

    def __iter__(self):
        """Return the iterator object (self).

        Returns
        -------
        CG
            Iterator object for the CG solver.
        """
        return self

    def __next__(self) -> tuple[int, float, Vec]:
        """Perform one iteration of the Conjugate Gradient algorithm.

        Returns
        -------
        tuple[int, float, Vec]
            A tuple containing:
            - iteration : int
                Current iteration number.
            - residual : float
                Current residual norm ||r_k||.
            - solution : Vec
                Current solution estimate x_k.

        Raises
        ------
        StopIteration
            When the algorithm has converged or reached maximum iterations.
        """
        if self.done():
            raise StopIteration

        beta = self.residual**2 / self.prev_residual**2

        self.u *= beta
        self.u += self.r

        self.c[:] = self.mat.dot(self.u)

        alpha = self.residual**2 / np.dot(self.u, self.c)
        self.solution += alpha * self.u
        self.r -= alpha * self.c

        self.prev_residual = self.residual
        self.residual = float(np.linalg.norm(self.r))
        self.iteration += 1

        return self.iteration, self.residual, self.solution


def cg(
    mat: Mat,
    rhs: Vec,
    initial_guess: Vec | None = None,
    abstol: float = 1e-10,
    reltol: float = 1e-8,
) -> tuple[float, Vec]:
    """Solve a linear system using the Conjugate Gradient method.

    This is a convenience function that creates a CG solver and runs it to completion.

    Parameters
    ----------
    mat : ndarray or sparse matrix
        Coefficient matrix A. Must be square, symmetric, and positive definite.
    rhs : ndarray
        Right-hand side vector b.
    initial_guess : ndarray, optional
        Initial guess for the solution. If None, zero vector is used.
    abstol : float, optional
        Absolute tolerance for convergence. Default is 1e-10.
    reltol : float, optional
        Relative tolerance for convergence. Default is 1e-8.

    Returns
    -------
    tuple[float, Vec]
        A tuple containing:
        - residual : float
            Final residual norm.
        - solution : Vec
            Solution vector.
    """
    cg_solver = CG(mat, rhs, initial_guess, abstol, reltol)

    for _ in cg_solver:
        pass

    return cg_solver.residual, cg_solver.solution


if __name__ == "__main__":
    n = 1024
    h = 1.0 / n
    a = [2.0 / (h * h)] * (n - 1)
    b = [-1.0 / (h * h)] * (n - 2)
    mat = np.diag(a) + np.diag(b, k=1) + np.diag(b, k=-1)
    rhs = np.array([np.pi**2 * np.sin(i * h * np.pi) for i in range(1, n)])
    mat = sp.csr_matrix(mat)
    cg_solver = CG(mat, rhs)
    import time

    start = time.time()
    for iteration, residual, _ in cg_solver:
        print(f"Iteration: {iteration}, Residual: {residual}")
    end = time.time()
    print(f"Time: {(end - start) * 1000 * 1000} μs")
