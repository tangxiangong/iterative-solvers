# IterativeSolvers

English | [简体中文](README-zh.md)

Rust implementations of iterative algorithms for solving linear systems.

[![docs.rs](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/iterative-solvers/latest/iterative_solvers/)
[![crates.io](https://img.shields.io/crates/v/iterative-solvers.svg)](https://crates.io/crates/iterative-solvers)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)


## Iterative Algorithms

- [x] Conjugate Gradient (CG)

## Supported Linear Algebra Libraries

- [x] [nalgebra](https://github.com/dimforge/nalgebra) (default)
- [x] [faer](https://github.com/sarah-quinones/faer-rs)

You can choose your preferred backend using Cargo features:

```toml
[dependencies]
iterative-solvers = { version = "*", default-features = false, features = ["faer"] }
```

## Usage

Consider the following differential equation:

```math
-\frac{d^2 u}{dx^2} = \pi^2 \sin(\pi x) \quad \text{for} \quad x \in (0, 1)
```

with the boundary conditions:

```math
u(0) = 0, \quad u(1) = 0.
```

The solution to this problem is:

```math
u(x) = \sin(\pi x).
```

Now we use the central difference method to discretize the differential equation:

```math
-\frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = \pi^2 \sin(\pi x_i) \quad \text{for} \quad i = 1, \ldots, N-1,
```

where $h = 1/N$ is the step size, and $u_i$ is the approximation of $u(x_i)$. The boundary conditions are:

```math
u_0 = 0, \quad u_N = 0.
```
The coefficient matrix of this linear system is a symmetric tridiagonal matrix, whose diagonal elements are $2/h^2$ and the sub-diagonal elements are $-1/h^2$.

We can solve this linear system using the Conjugate Gradient (CG) method.

```rust
use iterative_solvers::{cg, CG, utils::sparse::symmetric_tridiagonal_csc};
use nalgebra::DVector;
use std::f64::consts::PI;

fn main() {
    let n = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); n - 1];
    let b = vec![-1.0 / (h * h); n - 2];
    // Store the symmetric tridiagonal matrix in CSC format
    let mat = symmetric_tridiagonal_csc(&a, &b).unwrap();
    // Generate the right-hand side vector
    let rhs: Vec<_> = (1..n)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    // Generate the exact solution
    let solution: Vec<_> = (1..n).map(|i| (i as f64 * h * PI).sin()).collect();
    let solution = DVector::from_vec(solution);
    let rhs = DVector::from_vec(rhs);
    // Solve the linear system using the CG method
    let result = cg(&mat, &rhs, 1e-10, 1e-8).unwrap();
    // Calculate the error
    let e = (solution - result.solution()).norm();
    println!("error: {}", e);
}
```

If you want to know the residual, the approximate solution and the conjugate direction at each iteration, the iterator will help you.

```rust
let abstol = 1e-10;
let reltol = 1e-8;
let mut solver = CG::new(&mat, &rhs, abstol, reltol).unwrap();
while let Some(residual) = solver.next() {
    println!("residual: {residual}");
    println!("solution: {:#?}", solver.solution());
    println!("conjugate direction: {:#?}", solver.conjugate_direction());
}
let e = (solution - solver.solution()).norm();
println!("error: {}", e);
```

## Acknowledgments

This library is a Rust implementation based on the Julia package [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl), which is licensed under the [MIT License](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/master/LICENSE).

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
