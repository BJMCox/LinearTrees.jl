using AbstractTrees, StableRNGs

@testset "print_tree snapshot on the Figure 1 tree" begin
    T = Float64; N(; kw...) = Node{T,T}(; kw...)
    nodes = [N(feature = 1, threshold = 2.3, left = 2, right = 3, lcoef = 0.3, lintercept = 0.2,
               rcoef = 0.8, rintercept = -0.3, model = PLIN), N(lintercept = 0.0), N(lintercept = 0.0)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -100.0, 100.0, 0.0, 1, true)
    io = IOBuffer(); print_tree(io, TreeView(tree, [:age]))
    s = String(take!(io))
    @test occursin("age ≤ 2.3  [plin]", s)
    @test occursin("0.3·age + 0.2", s) && occursin("0.8·age − 0.3", s)
    @test occursin("leaf", s)
end

@testset "show summarises" begin
    rng = StableRNG(32); X = rand(rng, 50, 2); y = X[:, 1]
    t = fit_tree(X, y)
    s = sprint(show, MIME("text/plain"), t)
    @test startswith(s, "LinearTree{Float64, Float64, MSE}")
    @test occursin("nodes", s)
end
