#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]

pub mod error;
pub use error::*;

#[cfg(feature = "nalgebra")]
mod nalgebra;

#[cfg(feature = "faer")]
mod faer;

#[cfg(feature = "nalgebra")]
pub use nalgebra::*;

#[cfg(feature = "faer")]
pub use faer::*;
