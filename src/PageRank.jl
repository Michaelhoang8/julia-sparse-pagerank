module PageRank
using SparseArrays, LinearAlgebra
export transition_matrix, pagerank

"""Column-stochastic matrix: repeated directed edges have additive weights."""
function transition_matrix(n::Integer, edges)
    n > 0 || throw(ArgumentError("vertex count must be positive"))
    rows = Int[]; cols = Int[]; weights = Float64[]
    for (source, target, weight) in edges
        source isa Integer && target isa Integer || throw(ArgumentError("vertex IDs must be integers"))
        1 <= source <= n && 1 <= target <= n || throw(ArgumentError("vertex outside 1:n"))
        isfinite(weight) && weight > 0 || throw(ArgumentError("weights must be finite and positive"))
        push!(rows, target); push!(cols, source); push!(weights, weight)
    end
    matrix = sparse(rows, cols, weights, n, n)
    totals = vec(sum(matrix; dims=1))
    all(isfinite, totals) || throw(ArgumentError("outgoing weight overflow"))
    for column in 1:n
        totals[column] == 0 && continue
        for index in nzrange(matrix, column)
            nonzeros(matrix)[index] /= totals[column]
        end
    end
    matrix, findall(iszero, totals)
end

"""Power iteration with uniform teleportation and explicit dangling redistribution."""
function pagerank(n::Integer, edges; damping=0.85, tolerance=1e-12, maxiter=10_000)
    isfinite(damping) && 0 <= damping < 1 || throw(ArgumentError("damping must be in [0,1)"))
    isfinite(tolerance) && 0 < tolerance < 1 || throw(ArgumentError("tolerance must be in (0,1)"))
    maxiter isa Integer && maxiter > 0 || throw(ArgumentError("maxiter must be a positive integer"))
    matrix, dangling = transition_matrix(n, edges)
    rank = fill(1.0 / n, n)
    for iteration in 1:maxiter
        next = damping .* (matrix * rank .+ sum(rank[dangling]) / n) .+ (1 - damping) / n
        residual = norm(next - rank, 1)
        if residual <= tolerance
            return (scores=next, iterations=iteration, residual=residual)
        end
        rank = next
    end
    throw(ErrorException("PageRank did not converge within maxiter"))
end
end
