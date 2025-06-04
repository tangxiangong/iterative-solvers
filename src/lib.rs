#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

pub mod cg;
pub use cg::*;

pub mod utils;

pub mod error;
pub use error::*;

mod solver;
pub use solver::*;
