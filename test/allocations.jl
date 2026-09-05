using StableRNGs, DataFrames, CategoricalArrays

@testset "hot paths do not allocate" begin
    # @allocated on a non-const global can allocate for reasons unrelated to
    # the callee, so each measurement runs inside a local function with typed
    # locals -- the arguments below are boxed globals otherwise.
    function scan_alloc()
        x = sort(rand(1000)); z = rand(1000); h = ones(1000); w = ones(1000)
        LinearTrees.scan_feature(x, z, h, w, BIC(), 5, 1e-12)
        return @allocated LinearTrees.scan_feature(x, z, h, w, BIC(), 5, 1e-12)
    end
    @test scan_alloc() == 0

    function score_row_alloc()
        X = rand(50, 2); t = fit_tree(X, X[:, 1])
        LinearTrees.score_row(t, X, 1, true)
        return @allocated LinearTrees.score_row(t, X, 1, true)
    end
    @test score_row_alloc() == 0
end

@testset "shap recursion does not allocate per row after warm-up" begin
    # Fails the moment a `copy(path)` (or any other fresh Vector{PathElem})
    # comes back into `visit!`: the first row grows the pool, so a second row
    # can only allocate if the recursion builds paths instead of reusing them.
    function shap_recurse_alloc()
        rng = StableRNG(31)
        X = rand(rng, 600, 4)
        y = sin.(3 .* X[:, 1]) .+ 2 .* X[:, 2] .* (X[:, 3] .> 0.5) .+ X[:, 4] .^ 2 .+
            0.05 .* randn(rng, 600)
        t = fit_tree(X, y; max_depth = 8)
        φ = zeros(2, 4)
        pool = LinearTrees.PathPool()
        x1 = view(X, 1, :); x2 = view(X, 2, :)
        LinearTrees.shap_recurse!(φ, t, x1, 1, 1, pool, 1.0, 1.0, 0)
        return @allocated LinearTrees.shap_recurse!(φ, t, x2, 2, 1, pool, 1.0, 1.0, 0)
    end
    @test shap_recurse_alloc() == 0
end

@testset "table encoding does not allocate per row" begin
    # `Tables.getcolumn` is typed `AbstractVector`, so without a function
    # barrier every element read, conversion and store inside `encode_column!`
    # dispatches at run time and boxes its result: one allocation per row per
    # column. Fails the moment the per-row loops move back into the function
    # that fetched the column.
    function encode_alloc()
        rng = StableRNG(41)
        n = 5_000
        df = DataFrame(a = rand(rng, n), b = rand(rng, n),
            c = categorical(rand(rng, ["x", "y", "z"], n)))
        enc = LinearTrees.TableEncoder(df, :error)
        LinearTrees.encode(enc, df; nthreads = 1)
        return @allocated LinearTrees.encode(enc, df; nthreads = 1)
    end
    # the output matrix, the column fetches and the level map are a fixed cost;
    # anything per row would be 5_000 allocations and megabytes on top
    @test encode_alloc() < 200_000
end
