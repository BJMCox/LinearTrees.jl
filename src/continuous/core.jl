module Continuous

using LinearAlgebra
using SparseArrays

struct TreeNode
    feature::Int
    threshold::Float64
    left::Int
    right::Int
    depth::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
end

struct Posterior
    coef::Matrix{Float64}
    precision::Cholesky{Float64,Matrix{Float64}}
    shape::Float64
    rate::Vector{Float64}
    score::Float64
end

struct Fit
    nodes::Vector{TreeNode}
    terms::Vector{Vector{Int}}
    leaves::Vector{Int}
    slots::Vector{Int}
    N::Matrix{Float64}
    post::Posterior
    rawcoef::Matrix{Float64}
    history::Vector{Float64}
    moves::Vector{Tuple{Symbol,Int}}
    evaluations::Int
    skipped_dimension::Int
    cache_builds::Int
    cache_hits::Int
end

struct UnchangedBasis
    leaves::Vector{Int}
    N::Matrix{Float64}
end

mutable struct IncrementalContext
    nodes::Vector{TreeNode}
    leaves::Vector{Int}
    terms::Vector{Vector{Int}}
    p::Int
    cache::Dict{Vector{Int},UnchangedBasis}
    traces::Dict{Int,Vector{Vector{Tuple{Int,Bool}}}}
    builds::Int
    hits::Int
end

root(p::Int) = [TreeNode(0, NaN, 0, 0, 0, fill(-1.0, p), fill(1.0, p))]
leafindices(nodes) = findall(node -> node.feature == 0, nodes)

function grow(nodes::Vector{TreeNode}, leaf::Int, feature::Int, threshold::Float64)
    out = copy(nodes)
    node = nodes[leaf]
    left = length(out) + 1
    right = left + 1
    left_hi = copy(node.hi)
    left_hi[feature] = threshold
    right_lo = copy(node.lo)
    right_lo[feature] = threshold
    push!(out, TreeNode(0, NaN, 0, 0, node.depth + 1, copy(node.lo), left_hi))
    push!(out, TreeNode(0, NaN, 0, 0, node.depth + 1, right_lo, copy(node.hi)))
    out[leaf] = TreeNode(feature, threshold, left, right, node.depth, node.lo, node.hi)
    return out
end

function route(nodes::Vector{TreeNode}, x)
    index = 1
    while nodes[index].feature != 0
        node = nodes[index]
        index = x[node.feature] <= node.threshold ? node.left : node.right
    end
    return index
end

function _terms(p::Int, pairs::Vector{Tuple{Int,Int}})
    terms = Vector{Vector{Int}}([Int[]])
    append!(terms, ([j] for j in 1:p))
    append!(terms, ([j, k] for (j, k) in pairs))
    return terms
end

function _slots(nodes, leaves)
    slots = zeros(Int, length(nodes))
    for (slot, leaf) in pairs(leaves)
        slots[leaf] = slot
    end
    return slots
end

function _shared_face(a::TreeNode, b::TreeNode, p::Int)
    for j in 1:p
        threshold = if a.hi[j] == b.lo[j]
            a.hi[j]
        elseif b.hi[j] == a.lo[j]
            b.hi[j]
        else
            continue
        end
        overlap = true
        for k in 1:p
            k == j && continue
            if max(a.lo[k], b.lo[k]) >= min(a.hi[k], b.hi[k])
                overlap = false
                break
            end
        end
        overlap && return j, threshold
    end
    return nothing
end

function _trace_groups(terms, normal)
    groups = Dict{Tuple{Int,Int},Vector{Tuple{Int,Bool}}}()
    order = Tuple{Int,Int}[]
    for (column, term) in pairs(terms)
        # Terms have degree at most two. Zero marks an absent coordinate.
        reduced = (0, 0)
        for j in term
            j == normal && continue
            reduced = iszero(reduced[1]) ? (j, 0) : (reduced[1], j)
        end
        if !haskey(groups, reduced)
            groups[reduced] = Tuple{Int,Bool}[]
            push!(order, reduced)
        end
        push!(groups[reduced], (column, normal in term))
    end
    return [groups[key] for key in order]
end

function _constraints(nodes, leaves, p, terms)
    term_count = length(terms)
    width = length(leaves) * term_count
    traces = Dict{Int,Vector{Vector{Tuple{Int,Bool}}}}()
    faces = Tuple{Int,Int,Int,Float64}[]
    row_count = 0
    for right in 2:length(leaves), left in 1:right-1
        face = _shared_face(nodes[leaves[left]], nodes[leaves[right]], p)
        face === nothing && continue
        normal, threshold = face
        haskey(traces, normal) || (traces[normal] = _trace_groups(terms, normal))
        push!(faces, (left, right, normal, threshold))
        row_count += length(traces[normal])
    end

    C = zeros(Float64, row_count, width)
    row = 0
    for (left, right, normal, threshold) in faces
        for group in traces[normal]
            row += 1
            for (column, contains_normal) in group
                weight = contains_normal ? threshold : 1.0
                C[row, (left - 1) * term_count + column] = weight
                C[row, (right - 1) * term_count + column] = -weight
            end
        end
    end
    return C
end

function _nullspace_svd(C::Matrix{Float64})
    width = size(C, 2)
    size(C, 1) == 0 && return Matrix{Float64}(I, width, width)
    factor = try
        svd(C; full = true)
    catch err
        err isa LAPACKException || rethrow()
        svd(C; full = true, alg = LinearAlgebra.QRIteration())
    end
    tolerance = maximum(size(C)) * eps(Float64) * maximum(factor.S; init = 0.0)
    rank = count(>(tolerance), factor.S)
    V = Matrix(adjoint(factor.Vt))
    return V[:, rank+1:end]
end

"Pivoted-QR nullspace with conservative SVD fallbacks for uncertain rank."
function _nullspace_dense_qr(C::Matrix{Float64})
    width = size(C, 2)
    size(C, 1) == 0 && return Matrix{Float64}(I, width, width)
    factor = try
        qr(transpose(C), ColumnNorm())
    catch err
        err isa LAPACKException || rethrow()
        return _nullspace_svd(C)
    end
    packed = getfield(factor, :factors)
    diagonal = abs.(view(packed, diagind(packed)))
    scale = maximum(diagonal; init = 0.0)
    tolerance = maximum(size(C)) * eps(Float64) * scale
    any(value -> tolerance / 16 < value <= 16 * tolerance, diagonal) &&
        return _nullspace_svd(C)
    rank = count(>(tolerance), diagonal)
    if rank > 0 && minimum(view(diagonal, 1:rank)) <= sqrt(eps(Float64)) * scale
        return _nullspace_svd(C)
    end

    tail = zeros(Float64, width, width - rank)
    for column in axes(tail, 2)
        tail[rank + column, column] = 1.0
    end
    Q = LinearAlgebra.QRPackedQ(packed, getfield(factor, :τ))
    N = Matrix(Q * tail)
    isempty(N) && return N

    probe_columns = unique((1, cld(size(N, 2), 2), size(N, 2)))
    probes = view(N, :, collect(probe_columns))
    matrix_scale = max(opnorm(C, Inf), 1.0)
    residual_limit = 64 * maximum(size(C)) * eps(Float64) * matrix_scale
    residual_ok = maximum(abs, C * probes; init=0.0) <= residual_limit
    gram = transpose(probes) * probes
    orthogonality_limit = 64 * width * eps(Float64)
    orthogonality_ok = maximum(abs, gram - I; init=0.0) <= orthogonality_limit
    return residual_ok && orthogonality_ok ? N : _nullspace_svd(C)
end

"Return an SPQR nullspace, or nothing when numerical rank is uncertain."
function _try_nullspace_spqr(C::Matrix{Float64})
    width = size(C, 2)
    size(C, 1) == 0 && return Matrix{Float64}(I, width, width)
    # BLAS avoids generic iterator checks for these strided Float64 rows.
    scale = maximum(BLAS.nrm2, eachrow(C); init=0.0)
    iszero(scale) && return Matrix{Float64}(I, width, width)
    tolerance = maximum(size(C)) * eps(Float64) * scale
    A = sparse(transpose(C))
    factor = try
        qr(A; tol=tolerance / 16)
    catch err
        message = err isa ErrorException ? err.msg : nothing
        message isa String && message == "Sparse QR factorization failed" || rethrow()
        return nothing
    end
    # SPQR chooses and orders its accepted pivots during factorization.
    r = rank(factor)
    diagonal = abs.(diag(factor.R))
    accepted = view(diagonal, 1:r)
    if !all(isfinite, accepted) ||
            (r > 0 && minimum(accepted) <= max(16tolerance, sqrt(eps(Float64)) * scale))
        return nothing
    end
    if r > 0
        # Healthy diagonals alone can conceal numerical rank deficiency.
        leading = UpperTriangular(Matrix(factor.R[1:r, 1:r]))
        reciprocal_condition = try
            inv(cond(leading, 1))
        catch err
            err isa LAPACKException || rethrow()
            return nothing
        end
        reciprocal_condition > sqrt(eps(Float64)) || return nothing
    end
    tail = zeros(Float64, width, width - r)
    for column in axes(tail, 2)
        tail[r + column, column] = 1.0
    end
    # Q is a computed property on older Julia versions.
    Q = factor.Q::SparseArrays.SPQR.QRSparseQ{Float64,Int}
    # Use the stored inverse row permutation instead of inverting prow twice.
    N = (Q * tail)[getfield(factor, :rpivinv), :]
    isempty(N) && return N
    # Individually discarded columns can form a significant direction together.
    BLAS.nrm2(transpose(A) * N) <= tolerance / 4 || return nothing
    columns = collect(unique((1, cld(size(N, 2), 2), size(N, 2))))
    probes = view(N, :, columns)
    maximum(abs, transpose(probes) * probes - I; init=0.0) <= 64 * width * eps(Float64) ||
        return nothing
    return N
end

function _nullspace_qr(C::Matrix{Float64})
    # Whole-fit benchmarks include the effect of basis sparsity on later solves.
    # Small and dense systems do not amortize SPQR conversion and rank guards.
    if size(C, 2) >= 64 && count(!iszero, C) <= length(C) ÷ 5
        N = _try_nullspace_spqr(C)
        return N === nothing ? _nullspace_svd(C) : N
    end
    return _nullspace_dense_qr(C)
end

function _projected_design(nodes::Vector{TreeNode}, slots::Vector{Int},
        X::Matrix{Float64}, terms::Vector{Vector{Int}}, N::Matrix{Float64})
    term_count = length(terms)
    # Wider bases benefit from BLAS with contiguous output columns. Keep the
    # scalar loop for small bases, where one GEMV call per row costs more.
    transposed = term_count >= 11
    n, dimension = size(X, 1), size(N, 2)
    B = Matrix{Float64}(undef, transposed ? (dimension, n) : (n, dimension))
    basis = zeros(Float64, term_count)
    for i in axes(X, 1)
        x = view(X, i, :)
        for (column, term) in pairs(terms)
            value = 1.0
            for j in term
                value *= x[j]
            end
            basis[column] = value
        end
        offset = (slots[route(nodes, x)] - 1) * term_count
        if transposed
            block = offset+1:offset+term_count
            mul!(view(B, :, i), transpose(view(N, block, :)), basis)
        else
            for k in axes(N, 2)
                value = 0.0
                @simd for column in 1:term_count
                    value += basis[column] * N[offset + column, k]
                end
                B[i, k] = value
            end
        end
    end
    return transposed ? transpose(B) : B
end

function _posterior(B, Y, coefficient_precision, noise_shape, noise_rate)
    precision = cholesky(Symmetric(B' * B + coefficient_precision * I))
    coef = precision \ (B' * Y)
    residual = Y - B * coef
    shape = noise_shape + size(Y, 1) / 2
    rate = Vector{Float64}(undef, size(Y, 2))
    for output in axes(Y, 2)
        rate[output] = noise_rate +
            (sum(abs2, view(residual, :, output)) +
             coefficient_precision * sum(abs2, view(coef, :, output))) / 2
    end
    common_score = size(B, 2) / 2 * log(coefficient_precision) -
        sum(log, diag(precision.U))
    score = size(Y, 2) * common_score - shape * sum(log, rate)
    return Posterior(coef, precision, shape, rate, score)
end

function _make_fit(nodes, terms, leaves, N, post;
        history = [post.score], moves = Tuple{Symbol,Int}[], evaluations = 1,
        skipped_dimension = 0, cache_builds = 0, cache_hits = 0)
    model_nodes = Vector{TreeNode}(nodes)
    slots = _slots(model_nodes, leaves)
    return Fit(model_nodes, terms, leaves, slots, N, post, N * post.coef,
        history, moves, evaluations, skipped_dimension, cache_builds, cache_hits)
end

function _full_design(nodes::Vector{TreeNode}, X::Matrix{Float64},
        terms::Vector{Vector{Int}}, solver::Symbol)
    p = size(X, 2)
    leaves = leafindices(nodes)
    slots = _slots(nodes, leaves)
    C = _constraints(nodes, leaves, p, terms)
    N = solver === :qr ? _nullspace_qr(C) :
        solver === :svd ? _nullspace_svd(C) :
        throw(ArgumentError("solver must be :qr or :svd"))
    length(terms) <= size(N, 2) || (N = _nullspace_svd(C))
    B = _projected_design(nodes, slots, X, terms, N)
    return (; N, B, leaves)
end

"Fit a fixed geometry, using `solver=:svd` as the full numerical oracle."
function fit_fixed(nodes::Vector{TreeNode}, X::Matrix{Float64}, Y::Matrix{Float64};
        pairs::Vector{Tuple{Int,Int}}, coefficient_precision::Float64 = 0.01,
        noise_shape::Float64 = 2.0, noise_rate::Float64 = 1.0,
        solver::Symbol = :qr)
    terms = _terms(size(X, 2), pairs)
    fitted = _full_design(nodes, X, terms, solver)
    post = _posterior(fitted.B, Y, coefficient_precision, noise_shape, noise_rate)
    return _make_fit(nodes, terms, fitted.leaves, fitted.N, post)
end

function IncrementalContext(nodes, p, terms)
    IncrementalContext(nodes, leafindices(nodes), terms, p,
        Dict{Vector{Int},UnchangedBasis}(),
        Dict{Int,Vector{Vector{Tuple{Int,Bool}}}}(), 0, 0)
end

function _unchanged_basis(ctx::IncrementalContext, nodes)
    # Fresh content-hashed keys are never mutated after insertion.
    refined = [k for k in ctx.leaves if nodes[k].feature != 0]
    if haskey(ctx.cache, refined)
        ctx.hits += 1
        return ctx.cache[refined]
    end
    unchanged = [k for k in ctx.leaves if nodes[k].feature == 0]
    C = _constraints(ctx.nodes, unchanged, ctx.p, ctx.terms)
    basis = UnchangedBasis(unchanged, _nullspace_qr(C))
    ctx.cache[refined] = basis
    ctx.builds += 1
    return basis
end

function _incremental_design(ctx::IncrementalContext, nodes::Vector{TreeNode},
        X::Matrix{Float64}; current_dimension::Int=0)
    basis = _unchanged_basis(ctx, nodes)
    leaves = leafindices(nodes)
    term_count = length(ctx.terms)
    slots = _slots(nodes, leaves)
    fixed_slots = Dict(leaf => slot for (slot, leaf) in pairs(basis.leaves))
    new_leaf_count = count(leaf -> !haskey(fixed_slots, leaf), leaves)
    unchanged_dimension = size(basis.N, 2)
    M = zeros(Float64, term_count * length(leaves),
        unchanged_dimension + term_count * new_leaf_count)
    column = unchanged_dimension
    for (slot, leaf) in pairs(leaves)
        offset = (slot - 1) * term_count
        if haskey(fixed_slots, leaf)
            source = (fixed_slots[leaf] - 1) * term_count
            M[offset+1:offset+term_count, 1:unchanged_dimension] .=
                view(basis.N, source+1:source+term_count, :)
        else
            for j in 1:term_count
                M[offset + j, column + j] = 1.0
            end
            column += term_count
        end
    end

    faces = Tuple{Int,Int,Int,Float64}[]
    row_count = 0
    for right in 2:length(leaves), left in 1:right-1
        haskey(fixed_slots, leaves[left]) && haskey(fixed_slots, leaves[right]) && continue
        face = _shared_face(nodes[leaves[left]], nodes[leaves[right]], ctx.p)
        face === nothing && continue
        normal, threshold = face
        haskey(ctx.traces, normal) ||
            (ctx.traces[normal] = _trace_groups(ctx.terms, normal))
        push!(faces, (left, right, normal, threshold))
        row_count += length(ctx.traces[normal])
    end

    reduced = zeros(Float64, row_count, size(M, 2))
    row = 0
    for (left, right, normal, threshold) in faces
        for group in ctx.traces[normal]
            row += 1
            for (j, contains_normal) in group
                weight = contains_normal ? threshold : 1.0
                for col in axes(M, 2)
                    reduced[row, col] += weight *
                        (M[(left - 1) * term_count + j, col] -
                         M[(right - 1) * term_count + j, col])
                end
            end
        end
    end
    N = M * _nullspace_qr(reduced)
    if size(N, 2) < term_count
        N = _nullspace_svd(_constraints(nodes, leaves, ctx.p, ctx.terms))
    end
    # Search already rejects refinements that add no degrees of freedom.
    size(N, 2) > current_dimension || return nothing
    B = _projected_design(nodes, slots, X, ctx.terms, N)
    return (; N, B, leaves)
end

function _fit_refined(ctx, nodes, X, Y, coefficient_precision, noise_shape, noise_rate,
        current_dimension::Int)
    fitted = _incremental_design(ctx, nodes, X; current_dimension)
    fitted === nothing && return nothing
    post = _posterior(fitted.B, Y, coefficient_precision, noise_shape, noise_rate)
    return _make_fit(nodes, ctx.terms, fitted.leaves, fitted.N, post)
end

function _fit_candidate(ctx::IncrementalContext, nodes, X, Y, pairs,
        coefficient_precision, noise_shape, noise_rate, current_dimension::Int)
    return _fit_refined(ctx, nodes, X, Y,
        coefficient_precision, noise_shape, noise_rate, current_dimension)
end

function _fit_candidate(::Nothing, nodes, X, Y, pairs,
        coefficient_precision, noise_shape, noise_rate, current_dimension::Int)
    # Keep full construction and the late dimension check as the reference path.
    return fit_fixed(nodes, X, Y; pairs, coefficient_precision, noise_shape, noise_rate)
end

function _proposals(nodes, X, thresholds; max_depth, min_leaf)
    routed = route.(Ref(nodes), eachrow(X))
    result = Tuple{Int,Int,Float64}[]
    for leaf in leafindices(nodes)
        node = nodes[leaf]
        node.depth >= max_depth && continue
        rows = findall(==(leaf), routed)
        length(rows) < 2min_leaf && continue
        for feature in axes(X, 2), threshold in thresholds[feature]
            node.lo[feature] < threshold < node.hi[feature] || continue
            left_count = count(i -> X[i, feature] <= threshold, rows)
            min(left_count, length(rows) - left_count) >= min_leaf &&
                push!(result, (leaf, feature, threshold))
        end
    end
    return result
end

function _coordinated_candidates(nodes, X, thresholds, graph;
        max_depth, min_leaf, prune_crosses, remaining)
    result = Tuple{Vector{TreeNode},Symbol,Int}[]
    singles = _proposals(nodes, X, thresholds; max_depth, min_leaf)

    if remaining >= 3
        for (leaf, feature, threshold) in singles
            if prune_crosses && !any(pair -> pair[1] == feature, graph)
                continue
            end
            first = grow(nodes, leaf, feature, threshold)
            left, right = first[leaf].left, first[leaf].right
            next = _proposals(first, X, thresholds; max_depth, min_leaf)
            for (candidate_leaf, other_feature, other_threshold) in next
                candidate_leaf == left && other_feature > feature || continue
                !prune_crosses || (feature, other_feature) in graph || continue
                (right, other_feature, other_threshold) in next || continue
                second = grow(first, left, other_feature, other_threshold)
                push!(result,
                    (grow(second, right, other_feature, other_threshold), :cross, 3))
            end
        end
    end

    # Sibling refinements are retained for every candidate-search policy.
    for parent in nodes
        parent.feature == 0 && continue
        left, right = parent.left, parent.right
        nodes[left].feature == 0 && nodes[right].feature == 0 || continue
        for (leaf, feature, threshold) in singles
            leaf == left && feature != parent.feature || continue
            (right, feature, threshold) in singles || continue
            first = grow(nodes, left, feature, threshold)
            push!(result, (grow(first, right, feature, threshold), :siblings, 2))
        end
    end
    return result
end

function _consider(best, best_score, best_move, current_dimension, trial, score, move)
    size(trial.N, 2) > current_dimension || return best, best_score, best_move, true
    score > best_score + 1e-8 || return best, best_score, best_move, false
    return trial, score, move, false
end

"Fit a continuous multilinear tree with shared geometry and independent outputs."
function fit(X::Matrix{Float64}, Y::Matrix{Float64};
        pairs::Vector{Tuple{Int,Int}}, candidate_search::Symbol = :full,
        max_splits::Int = 6, max_depth::Int = 4, min_leaf::Int = 8,
        thresholds::Vector{Vector{Float64}}, split_penalty::Float64 = 2.0,
        coefficient_precision::Float64 = 0.01, noise_shape::Float64 = 2.0,
        noise_rate::Float64 = 1.0, reuse_constraints::Bool = true)
    candidate_search in (:full, :graph_pruned) ||
        throw(ArgumentError("candidate_search must be :full or :graph_pruned"))
    terms = _terms(size(X, 2), pairs)
    nodes = root(size(X, 2))
    initial = _full_design(nodes, X, terms, :qr)
    post = _posterior(initial.B, Y, coefficient_precision, noise_shape, noise_rate)
    current = _make_fit(nodes, terms, initial.leaves, initial.N, post)
    score = post.score
    history = [score]
    moves = Tuple{Symbol,Int}[]
    evaluations = 1
    skipped_dimension = 0
    cache_builds = 0
    cache_hits = 0
    used = 0
    graph = Set(pairs)

    while used < max_splits
        context = reuse_constraints ? IncrementalContext(nodes, size(X, 2), terms) : nothing
        current_dimension = size(current.N, 2)
        best = current
        best_score = score
        best_move = (:single, 1)

        for (leaf, feature, threshold) in
                _proposals(nodes, X, thresholds; max_depth, min_leaf)
            candidate = grow(nodes, leaf, feature, threshold)
            trial = _fit_candidate(context, candidate, X, Y, pairs,
                coefficient_precision, noise_shape, noise_rate, current_dimension)
            evaluations += 1
            if trial === nothing
                skipped_dimension += 1
                continue
            end
            trial_score = trial.post.score - split_penalty * (used + 1)
            best, best_score, best_move, skipped =
                _consider(best, best_score, best_move, current_dimension, trial,
                    trial_score, (:single, 1))
            skipped_dimension += skipped
        end

        if used + 2 <= max_splits
            for (candidate, kind, added) in _coordinated_candidates(
                    nodes, X, thresholds, graph; max_depth, min_leaf,
                    prune_crosses = candidate_search === :graph_pruned,
                    remaining = max_splits - used)
                trial = _fit_candidate(context, candidate, X, Y, pairs,
                    coefficient_precision, noise_shape, noise_rate, current_dimension)
                evaluations += 1
                if trial === nothing
                    skipped_dimension += 1
                    continue
                end
                trial_score = trial.post.score - split_penalty * (used + added)
                best, best_score, best_move, skipped =
                    _consider(best, best_score, best_move, current_dimension, trial,
                        trial_score, (kind, added))
                skipped_dimension += skipped
            end
        end

        if context !== nothing
            cache_builds += context.builds
            cache_hits += context.hits
        end
        best === current && break
        current = best
        nodes = current.nodes
        used += best_move[2]
        score = best_score
        push!(history, score)
        push!(moves, best_move)
    end

    return _make_fit(current.nodes, current.terms, current.leaves, current.N, current.post;
        history, moves, evaluations, skipped_dimension, cache_builds, cache_hits)
end

"Return the routed projected design matrix for query rows."
function design(model::Fit, X::Matrix{Float64})
    size(X, 2) == length(model.nodes[1].lo) ||
        throw(DimensionMismatch("tree and predictor dimensions differ"))
    return _projected_design(model.nodes, model.slots, X, model.terms, model.N)
end

"Return routed posterior means without constructing the projected design."
function predict(model::Fit, X::Matrix{Float64})
    size(X, 2) == length(model.nodes[1].lo) ||
        throw(DimensionMismatch("tree and predictor dimensions differ"))
    output = Matrix{Float64}(undef, size(X, 1), size(model.rawcoef, 2))
    term_count = length(model.terms)
    for i in axes(X, 1)
        offset = (model.slots[route(model.nodes, view(X, i, :))] - 1) * term_count
        for response in axes(output, 2)
            value = 0.0
            for (column, term) in pairs(model.terms)
                contribution = model.rawcoef[offset + column, response]
                for feature in term
                    contribution *= X[i, feature]
                end
                value += contribution
            end
            output[i, response] = value
        end
    end
    return output
end

end
