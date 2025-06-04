using IterativeSolvers, BenchmarkTools, LinearAlgebra

N = 1024
h = 1/N

A = diagm(0=>2/h^2 * ones(N-1), 1=>-1/h^2 * ones(N-2), -1=>-1/h^2 * ones(N-2))
b = π^2 * sin.(π * (1:N-1) * h)
abstol = 1e-10
reltol = 1e-8

@benchmark cg($A, $b, abstol=$abstol, reltol=$reltol)
