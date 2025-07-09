bench-nalgebra:
    cargo bench --no-default-features --features nalgebra

bench-faer:
    cargo bench --no-default-features --features faer

bench:
    @echo "Running comparative benchmarks for all features..."
    @touch benchmark.txt
    @echo "=== nalgebra backend ===" > benchmark.txt
    cargo bench --no-default-features --features nalgebra >> benchmark.txt 2>&1
    @echo -e "\n=== faer backend ===" >> benchmark.txt
    cargo bench --no-default-features --features faer >> benchmark.txt 2>&1
    # @echo -e "\n=== ndarray backend ===" >> benchmark.txt
    # cargo bench --no-default-features --features ndarray >> benchmark.txt 2>&1
    @echo "Comparison results saved to benchmark.txt"

clean-bench:
    @echo "Cleaning benchmark results..."
    rm benchmark.txt
    @echo "Benchmark results cleaned"
