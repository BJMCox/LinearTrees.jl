@testset "prediction on Figure 1 tree" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    nodes = [
        N(feature = 1, threshold = 2.3, left = 2, right = 3, lcoef = 0.3, lintercept = 0.2,
          rcoef = 0.8, rintercept = -0.3, xmin = 0.0, xmax = 5.0, model = PLIN),
        N(feature = 3, threshold = 0.4, left = 4, right = 5, lcoef = 0.5, lintercept = -0.4,
          rcoef = 0.9, rintercept = 0.1, xmin = 0.0, xmax = 1.0, model = PLIN),
        N(lintercept = 0.0),                          # leaf: pred = 0.8 x1 - 0.3
        N(feature = 2, threshold = -1.7, left = 6, right = 7, lcoef = -1.2, lintercept = 0.7,
          rcoef = 0.6, rintercept = -0.1, xmin = -3.0, xmax = 3.0, model = PLIN),
        N(lintercept = 0.0),                          # leaf: 0.3 x1 + 0.9 x3 + 0.3
        N(lintercept = 0.0),                          # leaf: 0.3 x1 - 1.2 x2 + 0.5 x3 + 0.5
        N(lintercept = 0.0),                          # leaf: 0.3 x1 + 0.6 x2 + 0.5 x3 - 0.3
    ]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -100.0, 100.0, 0.0, 3, true)
    X = [3.0 0.0 0.0;      # right of root
         1.0 0.0 0.8;      # left, then x3 > 0.4
         1.0 -2.0 0.2;     # left, x3 <= 0.4, x2 <= -1.7
         1.0 0.0 0.2]      # left, x3 <= 0.4, x2 > -1.7
    s = score(tree, X)
    @test s ≈ [0.8*3 - 0.3, 0.3 + 0.9*0.8 + 0.3, 0.3 + 1.2*2 + 0.5*0.2 + 0.5, 0.3 + 0.5*0.2 - 0.3]
    @test predict(tree, X) == s
    out = similar(s); predict!(out, tree, X); @test out == s
end

@testset "predict! rejects a mismatched output length" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    nodes = [N(feature = 1, left = 2, right = 2, lcoef = 2.0, lintercept = 0.0,
               rcoef = 2.0, rintercept = 0.0, xmin = 0.0, xmax = 1.0, model = LIN),
             N(lintercept = 1.0)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -1.0, 1.0, 0.0, 1, true)
    X = reshape(collect(1.0:5.0), 5, 1)
    @test_throws DimensionMismatch predict!(zeros(3), tree, X)   # short out: used to silently write 3 values
    @test_throws DimensionMismatch predict!(zeros(7), tree, X)   # long out: used to BoundsError inside score_row
end

@testset "clamps and lin nodes" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    # root lin on x1 with range [0, 1], child leaf +1, score bound [-1, 1]
    nodes = [N(feature = 1, left = 2, right = 2, lcoef = 2.0, lintercept = 0.0,
               rcoef = 2.0, rintercept = 0.0, xmin = 0.0, xmax = 1.0, model = LIN),
             N(lintercept = 1.0)]
    tree = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -1.0, 1.0, 0.0, 1, true)
    @test score(tree, [5.0;;]) == [1.0]            # feature clamp to 1 gives 2, score clamp to 1
    @test score(tree, [5.0;;]; clip = false) == [3.0]  # feature clamp only
    off = LinearTree{T,T,MSE}(nodes, UInt64[], MSE(), -1.0, 1.0, 0.0, 1, false)
    @test score(off, [5.0;;]) == [11.0]
    root = LinearTree{T,T,MSE}([N(lintercept = 7.0)], UInt64[], MSE(), -1.0, 1.0, 0.0, 1, true)
    @test score(root, zeros(1, 1)) == [1.0]        # con root is clamped too
end

@testset "row_blocks takes a per-caller row minimum" begin
    # `PARALLEL_MIN_ROWS` is calibrated on `score_row`, which costs a tree walk
    # per row. `shap` costs a walk of every node per row -- thousands of times
    # more -- so it must be able to thread far below that gate. Fails while
    # `row_blocks` hard-codes the threshold: 1_000 rows then run in one block.
    nt = min(4, Threads.nthreads())
    # blocks run on separate tasks, so they are counted with an atomic and the
    # rows they cover are marked in place; pushing to a shared Vector here
    # trips Julia's concurrent-resize check
    nblocks = Threads.Atomic{Int}(0)
    covered = zeros(Int, 1_000)
    count_block(rs) = (Threads.atomic_add!(nblocks, 1); covered[rs] .+= 1)
    LinearTrees.row_blocks(count_block, 1_000, nt; minrows = 100)
    @test nblocks[] == nt
    @test all(==(1), covered)
    # the default is still the shared gate
    nblocks[] = 0
    LinearTrees.row_blocks(count_block, 1_000, nt)
    @test nblocks[] == 1
    @test all(==(2), covered)
end
