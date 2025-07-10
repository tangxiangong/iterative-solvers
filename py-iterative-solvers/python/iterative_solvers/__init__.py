from iterative_solvers._core import hello_from_bin, PyCSRMatrix, cg_solver


def main() -> None:
    print(hello_from_bin())


__all__ = ["hello_from_bin", "PyCSRMatrix", "cg_solver", "main"]
