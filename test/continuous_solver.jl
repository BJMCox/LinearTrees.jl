const CTreeCore = LinearTrees.Continuous
using StableRNGs

function exact_rational_rank(A)
    B = Rational{BigInt}.(A)
    m, n = size(B)
    row = 0
    for column in 1:n
        pivot = 0
        for i in row+1:m
            if !iszero(B[i, column])
                pivot = i
                break
            end
        end
        iszero(pivot) && continue
        row += 1
        if pivot != row
            B[[row, pivot], :] = B[[pivot, row], :]
        end
        value = B[row, column]
        for i in 1:m
            i == row && continue
            factor = B[i, column] / value
            iszero(factor) && continue
            for j in column:n
                B[i, j] -= factor * B[row, j]
            end
        end
        row == m && break
    end
    return row
end

function independent_face_samples(nodes, leaves)
    samples = Tuple{Int,Int,Vector{Float64}}[]
    p = length(nodes[first(leaves)].lo)
    for right in 2:length(leaves), left in 1:right-1
        a, b = nodes[leaves[left]], nodes[leaves[right]]
        for normal in 1:p
            threshold = if a.hi[normal] == b.lo[normal]
                a.hi[normal]
            elseif b.hi[normal] == a.lo[normal]
                b.hi[normal]
            else
                continue
            end
            lo, hi = max.(a.lo, b.lo), min.(a.hi, b.hi)
            all(j -> j == normal || lo[j] < hi[j], 1:p) || continue
            for fraction in (0.0, 0.5, 1.0)
                x = [(lo[j] + fraction * (hi[j] - lo[j])) for j in 1:p]
                x[normal] = threshold
                push!(samples, (left, right, x))
            end
        end
    end
    return samples
end

function leaf_polynomial(model, slot, x)
    terms, raw = model.terms, model.rawcoef
    offset = (slot - 1) * length(terms)
    value = zeros(size(raw, 2))
    for (column, term) in pairs(terms)
        basis = prod(j -> x[j], term; init=1.0)
        value .+= basis .* view(raw, offset + column, :)
    end
    return value
end

function compare_incremental(old, candidate, X, Y, interactions)
    full = CTreeCore.fit_fixed(candidate, X, Y; pairs=interactions, solver=:svd,
        coefficient_precision=0.07, noise_shape=2.5, noise_rate=0.8)
    terms = CTreeCore._terms(size(X, 2), interactions)
    context = CTreeCore.IncrementalContext(old, size(X, 2), terms)
    incremental = CTreeCore._incremental_design(context, candidate, X)
    post = CTreeCore._posterior(incremental.B, Y, 0.07, 2.5, 0.8)

    @test size(incremental.N, 2) == size(full.N, 2)
    @test incremental.N * incremental.N' ≈ full.N * full.N' atol=2e-9 rtol=2e-9
    @test incremental.N * post.coef ≈ full.rawcoef atol=3e-9 rtol=3e-9
    @test incremental.B * post.coef ≈ CTreeCore.predict(full, X) atol=3e-9 rtol=3e-9
    @test post.rate ≈ full.post.rate atol=3e-9 rtol=3e-9
    @test post.score ≈ full.post.score atol=3e-9 rtol=3e-9
    return full, context
end

@testset "Continuous constraint solver" begin
    xs = collect(range(-0.95, 0.95; length=15))
    X = hcat(repeat(xs; inner=length(xs)), repeat(xs; outer=length(xs)))
    Y = hcat((@. 1 + X[:, 1] - 2X[:, 2] + X[:, 1] * X[:, 2]),
        (@. -0.5 + 0.3X[:, 1] + X[:, 2] - 0.7X[:, 1] * X[:, 2]))
    interactions = [(1, 2)]
    terms = CTreeCore._terms(2, interactions)

    old = CTreeCore.grow(CTreeCore.root(2), 1, 1, 0.0)
    old = CTreeCore.grow(old, 2, 2, 0.0)
    candidate = CTreeCore.grow(old, 3, 1, 0.5)
    candidate = CTreeCore.grow(candidate, 6, 2, 0.0)
    candidate = CTreeCore.grow(candidate, 7, 2, 0.0)

    old_C = CTreeCore._constraints(old, CTreeCore.leafindices(old), 2, terms)
    candidate_C = CTreeCore._constraints(
        candidate, CTreeCore.leafindices(candidate), 2, terms)
    @test size(old_C, 2) - exact_rational_rank(old_C) == 7
    @test size(candidate_C, 2) - exact_rational_rank(candidate_C) == 12

    full, context = compare_incremental(old, candidate, X, Y, interactions)
    @test size(full.N, 2) == 12
    @test context.builds == 1 && context.hits == 0

    # The nodal hat at the old T-junction requires refitting unchanged neighbours.
    witness = zeros(size(candidate_C, 2))
    for (slot, leaf) in Base.pairs(full.leaves)
        node = candidate[leaf]
        x = (node.lo[1] + node.hi[1]) / 2
        y = (node.lo[2] + node.hi[2]) / 2
        x >= 0.5 && continue
        ax, ay = x < 0 ? 1.0 : -2.0, y < 0 ? 1.0 : -1.0
        witness[4slot-3:4slot] .= (1.0, ax, ay, ax * ay)
    end
    @test full.N * (full.N' * witness) ≈ witness atol=2e-10

    # Discover faces from leaf boxes, then compare the fitted polynomials directly.
    old_fit = CTreeCore.fit_fixed(old, X, Y; pairs=interactions, solver=:svd)
    @test size(old_fit.N, 2) == 7
    face_samples = independent_face_samples(old_fit.nodes, old_fit.leaves)
    @test length(face_samples) == 9
    for (left, right, point) in face_samples
        @test leaf_polynomial(old_fit, left, point) ≈
            leaf_polynomial(old_fit, right, point) atol=2e-10
    end
    junction = zeros(2)
    values = [leaf_polynomial(old_fit, slot, junction) for slot in 1:3]
    @test values[1] ≈ values[2] atol=2e-10
    @test values[1] ≈ values[3] atol=2e-10

    for threshold in (1e-4, 1e-8, 1e-12)
        thin = CTreeCore.grow(old, 3, 1, threshold)
        thin = CTreeCore.grow(thin, 6, 2, 0.0)
        thin = CTreeCore.grow(thin, 7, 2, 0.0)
        compare_incremental(old, thin, X, Y, interactions)
    end
end

@testset "Full search agrees with constraint reuse" begin
    grid = collect(range(-0.9, 0.9; length=9))
    X = reduce(vcat, ([x y z] for x in grid for y in grid for z in grid))
    Y = hcat((@. 8abs(X[:, 1] + 0.3) + 5abs(X[:, 1] - 0.3) + X[:, 2]),
        (@. -4abs(X[:, 1] + 0.3) + 7abs(X[:, 1] - 0.3) - X[:, 3]))
    for policy in (:full, :graph_pruned)
        settings = (; pairs=[(1, 3)], candidate_search=policy, max_splits=6,
            max_depth=4, min_leaf=6, thresholds=fill([-0.3, 0.3], 3), split_penalty=0.0)
        reused = CTreeCore.fit(X, Y; settings..., reuse_constraints=true)
        rebuilt = CTreeCore.fit(X, Y; settings..., reuse_constraints=false)
        @test length(reused.moves) > 1
        @test reused.moves == rebuilt.moves
        @test reused.evaluations == rebuilt.evaluations
        @test reused.skipped_dimension == rebuilt.skipped_dimension > 0
        @test reused.history ≈ rebuilt.history atol=3e-9 rtol=3e-9
        @test reused.rawcoef ≈ rebuilt.rawcoef atol=3e-9 rtol=3e-9
        reused_cov = reused.N * (reused.post.precision \ reused.N')
        rebuilt_cov = rebuilt.N * (rebuilt.post.precision \ rebuilt.N')
        @test reused_cov ≈ rebuilt_cov atol=3e-9 rtol=3e-9
        @test reused.post.rate ≈ rebuilt.post.rate atol=3e-9 rtol=3e-9
        @test reused.post.score ≈ rebuilt.post.score atol=3e-9 rtol=3e-9
        @test CTreeCore.predict(reused, X) ≈ CTreeCore.predict(rebuilt, X) atol=3e-9 rtol=3e-9
        @test reused.cache_builds >= length(reused.moves) && reused.cache_hits > 0
        @test rebuilt.cache_builds == rebuilt.cache_hits == 0
    end
end

@testset "Continuous QR rank fallback" begin
    k = 625
    delta = 20 * (k + 1) * eps(Float64)
    C = hcat(ones(k), delta * Matrix{Float64}(I, k, k))
    qr_basis = CTreeCore._nullspace_qr(C)
    svd_basis = CTreeCore._nullspace_svd(C)
    @test size(qr_basis, 2) == size(svd_basis, 2) == 625
    @test qr_basis * qr_basis' ≈ svd_basis * svd_basis' atol=2e-9 rtol=2e-9
end

@testset "Sparse constraints preserve the tree posterior" begin
    p = 8
    interactions = [(j, k) for j in 1:p for k in j+1:p]
    nodes = CTreeCore.grow(CTreeCore.root(p), 1, 1, 0.1)
    nodes = CTreeCore.grow(nodes, 2, 2, -0.2)
    nodes = CTreeCore.grow(nodes, 3, 2, -0.2)
    X = 2rand(StableRNG(812), 128, p) .- 1
    Y = hcat((@. 1 + abs(X[:, 1] - 0.1) + X[:, 2] * X[:, 3]),
        (@. X[:, 4] - 2abs(X[:, 2] + 0.2)))
    model = CTreeCore.fit_fixed(nodes, X, Y; pairs=interactions)
    reference = CTreeCore.fit_fixed(nodes, X, Y; pairs=interactions, solver=:svd)
    C = CTreeCore._constraints(nodes, model.leaves, p, model.terms)
    sparse_basis = CTreeCore._try_nullspace_spqr(C)
    @test sparse_basis !== nothing
    @test sparse_basis * sparse_basis' ≈ reference.N * reference.N' atol=2e-9 rtol=2e-9
    @test model.N * model.N' ≈ reference.N * reference.N' atol=2e-9 rtol=2e-9
    @test model.N' * model.N ≈ I atol=2e-10 rtol=2e-10
    @test norm(C * model.N) <= 2e-10 * norm(C)
    @test model.rawcoef ≈ reference.rawcoef atol=2e-9 rtol=2e-9
    @test model.post.rate ≈ reference.post.rate atol=2e-9 rtol=2e-9
    @test model.post.score ≈ reference.post.score atol=2e-9 rtol=2e-9
    @test model.N * (model.post.precision \ model.N') ≈
        reference.N * (reference.post.precision \ reference.N') atol=2e-9 rtol=2e-9
    @test CTreeCore.predict(model, X) ≈ CTreeCore.predict(reference, X) atol=2e-9 rtol=2e-9
    for (left, right, point) in independent_face_samples(nodes, model.leaves)
        @test leaf_polynomial(model, left, point) ≈
            leaf_polynomial(model, right, point) atol=2e-9
    end
end

@testset "Sparse QR numerical rank guards" begin
    # Many individually tiny discarded columns can form a significant direction.
    k = 625
    C = zeros(k + 1, 3)
    C[1, 1] = 1
    C[2:end, 2] .= (k + 1) * eps(Float64) / 20
    @test CTreeCore._try_nullspace_spqr(C) === nothing

    # Kahan's matrix has healthy pivots but a singular value below the rank cutoff.
    n, sine = 60, 0.8
    cosine = sqrt(1 - sine^2)
    A = Diagonal(sine.^(0:n-1)) *
        (Matrix{Float64}(I, n, n) - cosine * triu(ones(n, n), 1))
    C = Matrix(transpose(A))
    @test CTreeCore._try_nullspace_spqr(C) === nothing
end

@testset "Graph pruning retains sibling refinements" begin
    grid = collect(range(-0.9, 0.9; length=9))
    X = hcat(repeat(grid; inner=length(grid)), repeat(grid; outer=length(grid)))
    nodes = CTreeCore.grow(CTreeCore.root(2), 1, 1, 0.0)
    thresholds = [Float64[0.0], Float64[0.0]]
    graph = Set{Tuple{Int,Int}}()
    full = CTreeCore._coordinated_candidates(nodes, X, thresholds, graph;
        max_depth=4, min_leaf=4, prune_crosses=false, remaining=3)
    pruned = CTreeCore._coordinated_candidates(nodes, X, thresholds, graph;
        max_depth=4, min_leaf=4, prune_crosses=true, remaining=3)
    signature(tree) = Tuple((node.feature, node.threshold) for node in tree if node.feature != 0)
    full_siblings = Set(signature(tree) for (tree, kind, _) in full if kind == :siblings)
    pruned_siblings = Set(signature(tree) for (tree, kind, _) in pruned if kind == :siblings)
    @test !isempty(full_siblings)
    @test pruned_siblings == full_siblings
    @test all(kind != :cross for (_, kind, _) in pruned)
end
