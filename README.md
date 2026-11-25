# Sparse PageRank with dangling vertices

Compute directed weighted PageRank using sparse matrix-vector multiplication and a convergence residual.

Language: **Julia**.

[![Validation](https://github.com/Michaelhoang8/julia-sparse-pagerank/actions/workflows/ci.yml/badge.svg)](https://github.com/Michaelhoang8/julia-sparse-pagerank/actions/workflows/ci.yml)

## Run

```sh
julia main.jl 3 fixtures/edges.csv
```

## Behavior

- Repeated edges add their weights; outgoing columns normalize to one.
- Dangling vertices redistribute their probability uniformly.
- Tests compare the iterative answer against an independent dense linear-system solution.

## Validation

```sh
julia test/runtests.jl
```

The workflow executes the checked-in fixtures and boundary tests. See the workflow log for the actual result; tests are included with the source.

## Scope

Vertex IDs must be integers from 1 through n. The input is a simple numeric CSV, not a general quoted-field parser. Results are numerical approximations; max-iteration exhaustion raises an error.

Code: [MIT](LICENSE). Fixtures are constructed examples.
