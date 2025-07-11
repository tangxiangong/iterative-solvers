# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a Rust workspace containing iterative linear algebra solvers with multi-language bindings:

- **Main Rust crate** (`./`): Core library implementing iterative algorithms (currently Conjugate Gradient)
- **Python bindings** (`py-iterative-solvers/`): PyO3-based Python wrapper using maturin
- **Julia benchmarks** (`julia-bench/`): Comparative benchmarks against Julia implementations

### Key Architecture Features

- **Multi-backend support**: Supports nalgebra (default), faer, and ndarray backends via Cargo features
- **Feature flags are mutually exclusive** - only one backend can be active at a time
- **Workspace structure** with shared dependencies and version management
- **Cross-language compatibility** with Python bindings and Julia comparisons

## Development Commands

### Rust Development

```bash
# Build with default features (nalgebra)
cargo build

# Build with specific backend
cargo build --no-default-features --features faer
cargo build --no-default-features --features nalgebra

# Run examples
cargo run --example cg

# Generate documentation
cargo doc --open

# Run doctests
cargo test --doc
```

### Testing

The project uses integration tests and doctests rather than unit tests in a separate `tests/` directory.

```bash
# Run all tests with default features
cargo test

# Test with specific backend
cargo test --no-default-features --features faer
```

### Benchmarking

Uses Just for benchmark automation:

```bash
# Install just if needed: cargo install just

# Run comparative benchmarks across all backends
just bench

# Run benchmarks for specific backend
just bench-nalgebra
just bench-faer

# Clean benchmark results
just clean-bench
```

Individual benchmark commands:
```bash
cargo bench --no-default-features --features nalgebra
cargo bench --no-default-features --features faer
```

### Python Development

```bash
cd py-iterative-solvers

# Build Python wheel (requires maturin)
maturin develop

# Run Python tests
python test.py

# Run Jupyter benchmarks
jupyter lab bench/bench.ipynb
```

## Code Organization

### Core Modules (`src/`)

- `lib.rs`: Main library entry point with feature documentation
- `cg.rs`: Conjugate Gradient implementation with backend abstractions
- `error.rs`: Error types using thiserror
- `ops.rs`: Matrix operation traits for backend abstraction
- `utils/`: Utility functions for matrix construction (dense.rs, sparse.rs, helper.rs)

### Backend Abstraction Pattern

The library uses trait-based abstraction to support multiple linear algebra backends:

1. **Traits in `ops.rs`** define common operations (matrix-vector multiplication, norms, etc.)
2. **Feature flags** control which backend implementation is compiled
3. **Each algorithm** (like CG) works generically over these traits

### Multi-language Support

- **Rust**: Primary implementation with generic backend support
- **Python**: PyO3 bindings in `py-iterative-solvers/` with maturin build system
- **Julia**: Benchmark comparisons in `julia-bench/` for performance validation

## Important Notes

- No separate unit test directory - tests are embedded as doctests and integration tests
- Benchmark results are saved to `benchmark.txt` for comparison
- Documentation is feature-dependent - built with nalgebra features for docs.rs
- Python bindings use stable ABI (abi3-py39) for compatibility
