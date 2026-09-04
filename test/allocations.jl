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
