using StableRNGs, JSON3

# test/loss.jl includes this file first for its fit-identity check; a second
# include would redefine `partition_cases` and warn
@isdefined(partition_cases) || include(joinpath(@__DIR__, "fixtures", "partition", "cases.jl"))

# The scan reads `st.idx` as "sorted by feature j, ties in ascending row order",
# so `presort!` owes each column exactly the permutation `sortperm` with a stable
# algorithm gives, and both of its sorts owe the same one. The two row counts
# straddle `RADIX_MIN_ROWS`, so the first runs the comparison sort and the second
# the radix sort. A radix key that folded `-0.0` into `0.0`, mishandled the sign
# of negative values, an unstable pass, or a threshold that let the two paths
# disagree, would break this.
@testset "presort! matches a stable comparison sort per feature" begin
    rng = StableRNG(23)
    for n in (LinearTrees.RADIX_MIN_ROWS - 1, LinearTrees.RADIX_MIN_ROWS)
        X = Matrix{Float64}(undef, n, 4)
        X[:, 1] = rand(rng, n)
        X[:, 2] = Float64.(rand(rng, 1:3, n))               # three values over n rows: ties everywhere
        X[:, 3] = [iseven(i) ? -0.0 : 0.0 for i in 1:n]     # signed zeros only
        X[:, 4] = randn(rng, n) .* 1e300                    # both signs, wide exponent range
        for M in (X, Float32.(X))                           # the key is per float width
            idx = Matrix{Int32}(undef, n, 4)
            LinearTrees.presort!(idx, M, 1)
            for j in 1:4
                @test idx[:, j] == Int32.(sortperm(view(M, :, j); alg = MergeSort))
            end
        end
    end
end

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
    roworder = Int32.(1:n)                   # the natural order a root node starts from
    leftrows = idx[span[1:3:end][1:10], 1]   # 10 of the span's rows, picked by position in column 1
    isleft = zeros(Bool, n)
    for i in leftrows
        isleft[i] = true                     # `grow_subtree` owns the marks; `partition!` only reads them
    end
    scratch = [LinearTrees.Scratch{Float64,Float64}() for _ in 1:2]
    expected = [vcat(filter(i -> i in leftrows, idx[span, j]), filter(i -> !(i in leftrows), idx[span, j])) for j in 1:p]
    nleft = LinearTrees.partition!(idx, roworder, span, isleft, scratch, 1:2)
    @test nleft == 10
    for j in 1:p
        @test idx[span, j] == expected[j]
    end
    # `roworder` gets the same stable split, which is what lets each child take a
    # view of it instead of a fresh row vector: the order the parent's sums ran
    # over is preserved on both sides
    @test roworder[span] == vcat(filter(i -> i in leftrows, Int32.(span)), filter(i -> !(i in leftrows), Int32.(span)))
    @test roworder[1:(first(span) - 1)] == Int32.(1:(first(span) - 1))   # outside the span untouched
end

function partition_holdout(X, kw)
    rng = StableRNG(2027)
    n, p = 600, size(X, 2)
    Xhold = rand(rng, n, p)
    p == 6 && (Xhold[:, 6] .= round.(Xhold[:, 6]; digits = 1))
    for j in get(kw, :categorical, Int[])
        Xhold[:, j] .= Float64.(rand(rng, 1:Int(maximum(view(X, :, j))), n))
    end
    return Xhold
end

# Low-gain nodes may differ when Julia or `--check-bounds` changes floating-point
# reduction order. Their public model function must still reproduce the recorded
# fit on data independent of the fitting sample. The direct partition tests above
# separately lock the stable row-set and ordering recurrence.
@testset "partitioned growth reproduces recorded held-out scores" begin
    for (name, X, y, loss, kw) in partition_cases()
        ref = from_dict(JSON3.read(read(joinpath(@__DIR__, "fixtures", "partition", "$name.json"), String), Dict{String,Any}))
        Xhold = partition_holdout(X, kw)
        expected = score(ref, Xhold; clip = false)
        for nt in (1, Threads.nthreads())
            t = fit_tree(X, y, loss; nthreads = nt, kw...)
            @test all(isapprox.(score(t, Xhold; clip = false), expected; rtol = 1e-6, atol = 1e-6))
        end
    end
end
