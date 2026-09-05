using StableRNGs, JSON3

include(joinpath(@__DIR__, "fixtures", "partition", "cases.jl"))

@testset "partition_column! is a stable partition of the span" begin
    # idx column: rows 1..12 in some sorted order; span covers positions 3:10
    idx = Int32[7, 2, 9, 4, 1, 11, 6, 3, 12, 8, 5, 10]
    isleft = zeros(Bool, 12)
    for i in (4, 11, 3, 8)
        isleft[i] = true
    end
    perm = Vector{Int32}(undef, 12)
    before = copy(idx)
    nleft = LinearTrees.partition_column!(idx, 3:10, isleft, perm)
    @test nleft == 4
    @test idx[3:6] == Int32[4, 11, 3, 8]          # left rows, original relative order
    @test idx[7:10] == Int32[9, 1, 6, 12]         # right rows, original relative order
    @test idx[1:2] == before[1:2] && idx[11:12] == before[11:12]   # outside the span untouched
end

@testset "partition! applies the same left set to every column" begin
    n = 50; p = 3
    rng = StableRNG(11)
    span = 10:40
    # every column holds the same row set over the span, each in its own order, as in `st.idx`
    base = Int32.(sortperm(rand(rng, n)))
    idx = reduce(hcat, [base for _ in 1:p])
    for j in 1:p
        idx[span, j] = base[span][sortperm(rand(rng, length(span)))]
    end
    leftrows = idx[span[1:3:end][1:10], 1]   # 10 of the span's rows, picked by position in column 1
    isleft = zeros(Bool, n)
    perms = [Vector{Int32}(undef, n) for _ in 1:2]
    expected = [vcat(filter(i -> i in leftrows, idx[span, j]), filter(i -> !(i in leftrows), idx[span, j])) for j in 1:p]
    nleft = LinearTrees.partition!(idx, span, leftrows, isleft, perms, 1:2)
    @test nleft == 10
    for j in 1:p
        @test idx[span, j] == expected[j]
    end
    @test !any(isleft[idx[span, 1]])   # marker cleared for reuse by the next node
end

@testset "partitioned growth reproduces the recorded trees" begin
    for (name, X, y, loss, kw) in partition_cases()
        ref = from_dict(JSON3.read(read(joinpath(@__DIR__, "fixtures", "partition", "$name.json"), String), Dict{String,Any}))
        for nt in (1, Threads.nthreads())
            t = fit_tree(X, y, loss; nthreads = nt, kw...)
            @test t.nodes == ref.nodes
            @test t.catmasks == ref.catmasks
        end
    end
end
