using Test, LinearAlgebra
include("../src/PageRank.jl")
using .PageRank

@testset "Sparse PageRank" begin
    edges = [(1,2,2.0),(1,3,1.0),(2,3,1.0)]
    result = pagerank(3, edges)
    # Independent dense linear-system oracle with the known transition columns.
    dense = [0.0 0.0 1/3; 2/3 0.0 1/3; 1/3 1.0 1/3]
    oracle = (I - 0.85dense) \ fill(0.15/3, 3)
    @test result.scores ≈ oracle atol=1e-11
    @test sum(result.scores) ≈ 1 atol=1e-12
    @test all(>(0), result.scores)
    @test result.residual <= 1e-12
    @test pagerank(3, Tuple{Int,Int,Float64}[]).scores ≈ fill(1/3,3)
    @test pagerank(2, [(1,2,1.0),(2,1,1.0)]).scores ≈ [0.5,0.5]
    @test pagerank(1, [(1,1,1.0)]).scores ≈ [1.0]
    @test pagerank(3, edges; damping=0).scores ≈ fill(1/3,3)
    @test pagerank(3, [(1,2,1.0),(1,2,1.0),(1,3,1.0),(2,3,1.0)]).scores ≈ result.scores
    @test_throws ArgumentError pagerank(0, [])
    @test_throws ArgumentError pagerank(3, [(0,1,1.0)])
    @test_throws ArgumentError pagerank(3, [(1,4,1.0)])
    @test_throws ArgumentError pagerank(3, [(1,2,-1.0)])
    @test_throws ArgumentError pagerank(3, [(1,2,Inf)])
    @test_throws ArgumentError pagerank(3, [(1,2,floatmax(Float64)),(1,2,floatmax(Float64))])
    @test_throws ArgumentError pagerank(3, edges; damping=1)
    @test_throws ArgumentError pagerank(3, edges; tolerance=0)
    @test_throws ArgumentError pagerank(3, edges; maxiter=0)
    @test_throws ErrorException pagerank(3, edges; maxiter=1)
end
