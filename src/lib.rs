#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

extern crate nalgebra as na;

pub mod cg;
pub use cg::*;

pub mod utils;
pub use utils::*;
