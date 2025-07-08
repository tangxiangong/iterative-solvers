#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
//!
//! ## Features
//!
//! This crate supports two different linear algebra backends through feature flags:
//!
//! ### `nalgebra` feature (enabled in docs)
//!
//! Uses the [nalgebra](https://docs.rs/nalgebra) crate for matrix operations.
//! This is the traditional and widely-used linear algebra library in Rust.
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = { version = "0.2", features = ["nalgebra"] }
//! ```
//!
//! ### `faer` feature (default)
//!
//! Uses the [faer](https://docs.rs/faer) crate for matrix operations.
//! This is a newer, high-performance linear algebra library.
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = "0.2"  # faer is the default feature
//! ```
//!
//! Or explicitly specify:
//!
//! ```toml
//! [dependencies]
//! iterative-solvers = { version = "0.2", features = ["faer"] }
//! ```
//!
//! ### Using Different Features
//!
//! Each function in this crate provides examples for both backends.
//! Look for sections marked:
//! - **"With nalgebra feature:"** - Examples using nalgebra matrices
//! - **"With faer feature:"** - Examples using faer matrices
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
//!
//! **Note**: The features are mutually exclusive - you can only use one at a time.

pub mod error;
pub use error::*;

pub mod cg;
pub use cg::*;

pub mod utils;

pub mod ops;
pub use ops::*;
