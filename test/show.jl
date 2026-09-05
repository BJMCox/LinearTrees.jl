using AbstractTrees, StableRNGs

@testset "print_tree snapshot on the Figure 1 tree" begin
    T = Float64; N(; kw...) = LinearTrees.Node{T,T}(; kw...)
    nodes = [N(feature = 1, threshold = 2.3, left = 2, right = 3, lcoef = 0.3, lintercept = 0.2,
               rcoef = 0.8, rintercept = -0.3, model = PLIN), N(lintercept = 0.0), N(lintercept = 0.0)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -100.0, 100.0, 0.0, 1, true)
    io = IOBuffer(); print_tree(io, TreeView(tree, [:age]))
    s = String(take!(io))
    @test occursin("age ≤ 2.3  [plin]", s)
    @test occursin("0.3·age + 0.2", s) && occursin("0.8·age − 0.3", s)
    @test occursin("leaf", s)
end

@testset "fmtnum keeps an exponent's sign ASCII, only the leading sign turns into a minus" begin
    # fails if fmtnum replaces every "-", turning "1.0e-5" into the unreadable "1.0e−5"
    T = Float64; N(; kw...) = LinearTrees.Node{T,T}(; kw...)
    nodes = [N(feature = 1, threshold = 0.0, left = 2, right = 3, model = PCON),
             N(lintercept = 1.0e-5), N(lintercept = -2.5)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -100.0, 100.0, 0.0, 1, true)
    io = IOBuffer(); print_tree(io, TreeView(tree, [:age]))
    s = String(take!(io))
    @test occursin("1.0e-5", s)
    @test occursin("−2.5", s)
end

@testset "show on a Softmax tree prints its header and vector pieces" begin
    # a vector-valued tree takes its own `fmtnum`/`piece` branch, which prints
    # every class's coefficient as a bracketed vector instead of folding a sign
    rng = StableRNG(34)
    X = rand(rng, 200, 3)
    y = [X[i, 1] > 0.5 ? 1 : X[i, 3] > 0.5 ? 2 : 3 for i in 1:200]
    t = fit_tree(X, y, Softmax(3))
    s = sprint(show, MIME("text/plain"), t)
    @test occursin("Softmax} with $(length(t.nodes)) nodes", s)   # the loss name and the node count
    @test occursin("$(count(LinearTrees.isleaf, t.nodes)) leaves, 3 features", s)
    @test occursin("leaf  intercept = [", s)   # the SVector intercept, not a scalar
end
