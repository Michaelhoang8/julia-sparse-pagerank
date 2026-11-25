include("src/PageRank.jl")
using .PageRank, Printf

function main(args)
    if args == ["--help"]
        println("Usage: julia main.jl <vertex-count> edges.csv\nCSV header: source,target,weight; vertices are numbered 1 through n.")
        return 0
    end
    length(args) == 2 || throw(ArgumentError("expected vertex count and edge CSV; use --help"))
    n = parse(Int, args[1])
    lines = readlines(args[2])
    !isempty(lines) && strip(lines[1]) == "source,target,weight" || throw(ArgumentError("invalid CSV header"))
    edges = Tuple{Int,Int,Float64}[]
    for (number, line) in enumerate(lines[2:end])
        fields = split(strip(line), ',')
        length(fields) == 3 || throw(ArgumentError("expected 3 fields on row $(number+1)"))
        push!(edges, (parse(Int, fields[1]), parse(Int, fields[2]), parse(Float64, fields[3])))
    end
    result = pagerank(n, edges)
    println("vertex,score")
    for vertex in sortperm(result.scores; rev=true)
        @printf("%d,%.12f\n", vertex, result.scores[vertex])
    end
    println(stderr, "iterations=$(result.iterations), residual=$(result.residual)")
    0
end
try
    exit(main(ARGS))
catch error
    println(stderr, sprint(showerror, error)); exit(1)
end
