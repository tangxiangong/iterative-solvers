# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] - 2025-07-12

### 🚀 Features

- Add Justfile for benchmarking nalgebra and faer features with clean-up tasks
- Add PyCSRMatrix class and cg_solver function for Conjugate Gradient method
- Implement Conjugate Gradient function in cg.py
- Remove Python binding and add feature `ndarray` to support `ndarray` backend
- Add `ndarray` backend support for dense and sparse matrices

### 🐛 Bug Fixes

- Replace MacOS alias files with actual license files in py-iterative-solvers
- Create proper symbolic links for license files in py-iterative-solvers

### 💼 Other

- Version 0.2.1
- Enhance benchmarks to support `nalgebra` and `faer` features for matrix operations
- Update benchmarks for CG and diagonal matrix to include dense, CSC, and CSR variants without nalgebra dependencies
- Add benchmark results for nalgebra and faer backends, highlighting performance improvements and regressions across various matrix operations
- Implement Conjugate Gradient solver for symmetric positive definite systems in Python
- Add Jupyter notebook and .gitignore for benchmarking
- Switch to sparse matrix representation in CG benchmark
- Increase benchmark matrix size to 8192 and rename bench targets

### 🚜 Refactor

- Merge two features `nalgebra` and `faer` into the same files.
- Streamline diagonal matrix creation and utility functions
- Clean up __init__.py and update _core.pyi exports
- Optimize CG solver and CSR matrix conversion
- Update CG benchmark to use solve method
- Enhance CG solver and CSR matrix handling
- Improve data handling in CG solver
- Update CG benchmark notebook for clarity and performance

### 📚 Documentation

- Update README files and add copyright notices in source files
- Add CLAUDE.md guide

### ⚡ Performance

- Update benchmark results with latest performance measurements

### 🧪 Testing

- Update diagonal matrix test output formatting to use debug print

### ⚙️ Miscellaneous Tasks

- Add `nalgebra` and `nalgebra-sparse` dependencies, and configure docs.rs for feature support
- Restructure Cargo.toml to support workspace configuration and update dependencies for iterative-solvers
- Initialize py-iterative-solvers project with Rust and Python integration, including setup files and dependencies
- Update GitHub Actions workflow to exclude py-iterative-solvers and refine build/test steps
- Update Python version requirement in pyproject.toml to support Python 3.12
- Update pyproject.toml to include development dependencies and modify matrix type in CG benchmark
- Update Cargo.toml to include nalgebra and nalgebra-sparse dependencies
- Simplify development dependencies section in pyproject.toml
- Remove obsolete uv.lock file
- Update .gitignore to exclude additional files
- Remove Python bindings

## [0.2.1] - 2025-06-15

### 🚀 Features

- Add struct `MINRES` but not implement
- In the `IterSolverError` enumeration, add the `InvalidInput` error type to handle invalid input
- Implement the structure of the Conjugate Gradient (CG) method and its related functions, supporting the solution of linear equations, and adding features for matrix operations.
- Add tool module, including related functions for sparse and dense matrices, and introduce new utility functions in the `faer` module.
- Utility functions for adding sparse matrices, including the creation features for diagonal, tridiagonal, and symmetric tridiagonal matrices.
- Add practical functions for adding dense matrices, including the creation functions for diagonal matrices, tridiagonal matrices, and symmetric tridiagonal matrices.
- New practical functions for vector operations, including vector checking, dot product calculation, and linear combination (axpy) functionality.
- Add auxiliary modules and export related functions to enhance the practicality of sparse and dense matrices.
- Implement the gemv operation for sparse matrices, optimize the matrix multiplication function, and remove unnecessary dot and axpy methods.
- Modify the conjugate gradient (CG) method to support more general matrix types, and update the related functions to use the new matrix operation and vector operation tools.

### 🐛 Bug Fixes

- Uncomment the `faer` module, enable its functionality, and export the relevant content.
- Fix function `dot`

### 💼 Other

- Enhanced benchmark tests for the conjugate gradient method, supporting sparse matrices in CSC and CSR formats, while removing the diagonal benchmark test files that are no longer in use.
- Bump version 0.2.1

### 🚜 Refactor

- Reformat code
- Temporarily block `benches` and `examples`
- Move the `is_vector` function to  `utils::helper`.

### 📚 Documentation

- Update the documentation
- Update the document, mark the `faer` backend as supported, and modify the Cargo feature version requirements.

### ⚙️ Miscellaneous Tasks

- Update version to 0.2.0
- Rename bench name
- Delete deny toml
- Update the `Cargo.toml` file to set the default feature to `faer`, enable the `faer` dependency, and comment out the benchmark test configuration.
- Change the default feature to `nalgebra`,  rebase `cg` and `diagm` benchmark test files

## [0.2.0] - 2025-06-05

### 🚀 Features

- Added tool functions for adding sparse matrices
- Implement support for matrix operations and sparse matrices
- Support for conjugate gradient (CG) methods that support general matrix operations
- Added operation module and exported

### 💼 Other

- Add Julia benchmark for conjugate gradient solver
- Add benchmark for conjugate gradient solver
- Updates import paths for dense matrix utilities

### 🚜 Refactor

- Refact CG solver to simplify state management and improve tolerance handling
- Remove solver module and its exports
- Rename result() to solve() for clarity
- Remove solver module and related traits
- Restructure code into feature-based modules
- Update the import path of the sparse matrix tool

### 📚 Documentation

- Update the README file, add document links, version badges, and license information
- Update CG solver examples with tolerance parameters and iteration output
- Update the README file to add information about supported backend linear algebra libraries
- Add documentation comments
- Updated the README file, modified the sample code to store symmetric tridiagonal matrices in CSC format, and added output information about the conjugate direction. Adjusted variable naming to improve readability.

### ⚙️ Miscellaneous Tasks

- Update to version 0.1.0
- Exclude julia-bench from language detection
- Update package metadata and benchmarks configuration
- Add optional features and dependency configurations
- Update
- Comment out the import and usage of the faer module
- Update the version number to 0.2.0, add the keyword "cg", and comment out the dependency of the faer module.

## [0.1.0] - 2025-06-04

### 🚀 Features

- Add bench for `diag`
- Add function `diag` to generate diagonal matrices
- Implement conjugate gradient (CG) method
- Add modules `cg` and `utils`
- Add dependency `thiserror`
- Add custom error enum
- Modify the CG structure to support borrowing matrices and vectors, and update related methods to improve performance.
- Add `LinearProblem` structure and solver interface, supporting dimension checking for matrices and right-hand vectors
- Add modules  `errors` and `solver`
- Improve the implementation of the conjugate gradient (CG) method, add state management and examples, and support initial guess
- Increase the error types and result types of the iterative solver, including descriptions of dimension matching errors.
- Refactor the linear problem structure into a linear system, add document comments and examples, and improve the solver interface to support multiple solvers.
- Enhanced the creation functions for diagonal and tridiagonal matrices, added detailed documentation comments and examples, and supported the construction of symmetric tridiagonal matrices.
- Optimize the module export structure
- Update baseline test, replace the call to the `diag` function with `diagm` to reflect the latest implementation

### 🐛 Bug Fixes

- Correct the import path, change `na` to `nalgebra`

### 💼 Other

- Add a conjugate gradient method example, demonstrating the solution process of a symmetric tridiagonal matrix and error calculation.

### 📚 Documentation

- Update the README file, add an example and detailed explanation of the conjugate gradient method, including the discretization process of the differential equation and its solution method.

### ⚙️ Miscellaneous Tasks

- Initialization
- Add benches

<!-- generated by git-cliff -->
