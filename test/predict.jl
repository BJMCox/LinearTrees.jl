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

@testset "categorical routing" begin
    T = Float64
    N(; kw...) = Node{T,T}(; kw...)
    nodes = [N(feature = 1, left = 2, right = 3, lintercept = 1.0, rintercept = 2.0,
               catstart = 1, catwords = 1, model = PCON),
             N(), N()]
    masks = [UInt64(0b101)]                         # levels 1 and 3 go left
    tree = LinearTree{T,T,MSE}(nodes, masks, MSE(), -10.0, 10.0, 0.0, 1, true)
    @test score(tree, [1.0; 2.0; 3.0; 7.0;;]) == [1.0, 2.0, 1.0, 2.0]   # 7 is unseen, routes right
end
