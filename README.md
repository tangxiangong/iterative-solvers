# IterativeSolvers

English | [简体中文](README-zh.md)

This library provides Rust implementations of iterative algorithms for solving linear system, drawing heavy inspiration from the Julia package [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl).

[![docs.rs](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/iterative-solvers/latest/iterative_solvers/)
[![crates.io](https://img.shields.io/crates/v/iterative-solvers.svg)](https://crates.io/crates/iterative-solvers)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)


## Iterative Algorithms

- [x] Conjugate Gradient (CG)

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
use iterative_solvers::{cg, utils::symmetric_tridiagonal};
use nalgebra::{DMatrix, DVector};

fn main() {
    let N = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); N - 1];
    let b = vec![-1.0 / (h * h); N - 2];
    // create a symmetric tridiagonal matrix
    let mat = symmetric_tridiagonal(&a, &b).unwrap();
    // create the right-hand side vector
    let rhs: Vec<_> = (1..N)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    // create the exact solution
    let solution: Vec<_> = (1..N)
        .map(|i| (i as f64 * h * PI).sin()).collect();
    let solution = DVector::from_vec(solution);
    let rhs = DVector::from_vec(rhs);
    // solve the linear system
    let abstol = 1e-10;
    let reltol = 1e-8;
    let result = cg(&mat, &rhs, abstol, reltol).unwrap();
    // compute the error
    let e = (solution - result.solution()).norm();
    println!("error: {}", e);
}
```

If you want to know the residual and the approximate solution at each iteration, the iterator will help you.

```rust
let abstol = 1e-10;
let reltol = 1e-8;
let mut solver = CG::new(&mat, &rhs, abstol, reltol).unwrap();
while let Some(residual) = solver.next() {
    println!("residual: {residual}");
    println!("solution: {:#?}", solver.solution());
}
let e = (solution - solver.solution()).norm();
println!("error: {}", e);
```

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
