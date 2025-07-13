#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
//!
//! ## Features
//!
//! This crate supports three different linear algebra backends through feature flags:
//!
//! ### `nalgebra` feature (default)
//!
//! Uses the [nalgebra](https://docs.rs/nalgebra) crate for matrix operations.
//! This is the traditional and widely-used linear algebra library in Rust.
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = "0.2"
//! ```
//!
//! ### `faer` feature
//!
//! Uses the [faer](https://docs.rs/faer) crate for matrix operations.
//! This is a newer, high-performance linear algebra library.
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = { version = "0.2", default-features = false, features = ["faer"] }
//! ```
//!
//! ### `ndarray` feature
//!
//! Uses the [ndarray](https://docs.rs/ndarray) crate for matrix operations.
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = { version = "0.2", default-features = false, features = ["ndarray"] }
//! ```
//!
//! ### Using Different Features
//!
//! Each function in this crate provides examples for all backends.
//! Look for sections marked:
//! - **"With nalgebra feature:"** - Examples using nalgebra matrices
//! - **"With faer feature:"** - Examples using faer matrices
//! - **"With ndarray feature:"** - Examples using ndarray matrices
//!
//!
//! ### Current Documentation
//!
#![cfg_attr(
    feature = "nalgebra",
    doc = "This documentation was generated with the `nalgebra` feature enabled."
)]
#![cfg_attr(
    feature = "faer",
    doc = "This documentation was generated with the `faer` feature enabled."
)]
#![cfg_attr(
    feature = "ndarray",
    doc = "This documentation was generated with the `ndarray` feature enabled."
)]
//!
//! **Note**: The features are mutually exclusive - you can only use one at a time.

pub mod error;
pub use error::*;

pub mod cg;
pub use cg::*;

pub mod utils;

pub mod ops;
pub use ops::*;

#[cfg(feature = "ndarray")]
extern crate blas_src;
