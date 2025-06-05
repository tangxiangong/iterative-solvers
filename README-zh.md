# IterativeSolvers

[English](README.md) | 简体中文

本库提供了数值线性代数中迭代算法的 Rust 实现，深受 Julia 包 [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl) 的启发。

[![文档](https://img.shields.io/badge/文档-最新-blue.svg)](https://docs.rs/iterative-solvers/latest/iterative_solvers/)
[![crates.io](https://img.shields.io/crates/v/iterative-solvers.svg)](https://crates.io/crates/iterative-solvers)
[![许可证: MIT/Apache-2.0](https://img.shields.io/badge/许可证-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)



## 迭代算法

- [x] 共轭梯度法 (CG)

## 支持的后端线性代数库

- [x] [nalgebra](https://github.com/dimforge/nalgebra) (默认)
- [ ] [faer](https://github.com/sarah-quinones/faer-rs)

您可以使用 Cargo 特性选择您喜欢的后端：

```toml
[dependencies]
iterative-solvers = { version = "0.1", default-features = false, features = ["faer"] }
```


## 使用方法

考虑以下微分方程：

```math
-\frac{d^2 u}{dx^2} = \pi^2 \sin(\pi x) \quad \text{当} \quad x \in (0, 1)
```

边界条件为：

```math
u(0) = 0, \quad u(1) = 0.
```

该问题的解为：

```math
u(x) = \sin(\pi x).
```

现在我们使用中心差分法离散化微分方程：

```math
-\frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = \pi^2 \sin(\pi x_i) \quad \text{当} \quad i = 1, \ldots, N-1,
```

其中 $h = 1/N$ 是步长，$u_i$ 是 $u(x_i)$ 的近似值。边界条件为：

```math
u_0 = 0, \quad u_N = 0.
```

该线性系统的系数矩阵是一个对称三对角矩阵，其对角元素为 $2/h^2$，次对角元素为 $-1/h^2$。

我们可以使用共轭梯度法（CG）求解该线性系统。

```rust
use iterative_solvers::{cg, utils::symmetric_tridiagonal};
use nalgebra::{DMatrix, DVector};

fn main() {
    let N = 1024;
    let h = 1.0 / 1024.0;
    let a = vec![2.0 / (h * h); N - 1];
    let b = vec![-1.0 / (h * h); N - 2];
    // 创建对称三对角矩阵
    let mat = symmetric_tridiagonal(&a, &b).unwrap();
    // 创建右端向量
    let rhs: Vec<_> = (1..N)
        .map(|i| PI * PI * (i as f64 * h * PI).sin())
        .collect();
    // 创建精确解
    let solution: Vec<_> = (1..N)
        .map(|i| (i as f64 * h * PI).sin()).collect();
    let solution = DVector::from_vec(solution);
    let rhs = DVector::from_vec(rhs);
    // 求解线性系统
    let abstol = 1e-10;
    let reltol = 1e-8;
    let result = cg(&mat, &rhs, abstol, reltol).unwrap();
    // 计算误差
    let e = (solution - result.solution()).norm();
    println!("error: {}", e);
}
```

如果您想知道每次迭代的残差和近似解，迭代器将为您提供帮助。

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

## 许可协议

本项目采用以下任一许可协议：

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) 或 https://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) 或 https://opensource.org/licenses/MIT)

您可以选择其中任意一种。

### 贡献

除非您明确声明，否则根据 Apache-2.0 许可协议的定义，您有意提交的任何贡献都将按照上述双重许可协议进行许可，不附加任何额外条款或条件。
