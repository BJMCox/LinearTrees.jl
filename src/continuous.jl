using LinearAlgebra: norm

"""
    ContinuousTree

A continuous regression tree fitted by [`fit_continuous_tree`](@ref). Its
leaf polynomials agree on every shared face. Holding other predictors fixed
therefore gives joined, piecewise linear coordinate slices for each output.
Selected pair terms allow interactions, but not interactions of order three
or higher. Outputs share geometry and have independent conditional posteriors.
"""
struct ContinuousTree{Scalar}
    fit::Continuous.Fit
    xcenter::Vector{Float64}
    xscale::Vector{Float64}
    ycenter::Vector{Float64}
    yscale::Vector{Float64}
end

function _continuous_pairs(pairs, p)
    pairs === :none && return Tuple{Int,Int}[]
    pairs === :all && return [(j, k) for j in 1:p for k in j+1:p]
    pairs isa AbstractVector || throw(ArgumentError("pairs must be :none, :all, or a vector of index pairs"))
    selected = Tuple{Int,Int}[]
    for pair in pairs
        pair isa Tuple{Integer,Integer} || throw(ArgumentError("each pair must contain two integer feature indices"))
        j, k = minmax(pair...)
        1 <= j < k <= p || throw(ArgumentError("pair indices must be distinct and lie in 1:$p"))
        push!(selected, (j, k))
    end
    return sort!(unique!(selected))
end

function _continuous_data(A, name)
    Base.require_one_based_indexing(A)
    eltype(A) <: Real || throw(ArgumentError("$name must contain real numbers"))
    data = A isa AbstractVector ? Vector{Float64}(A) : Matrix{Float64}(A)
    all(isfinite, data) || throw(ArgumentError("$name must contain finite Float64 values"))
    return data
end

function _continuous_predictors(X)
    p = size(X, 2)
    center = zeros(p)
    scale = ones(p)
    for j in 1:p
        lo, hi = extrema(view(X, :, j))
        center[j] = lo / 2 + hi / 2
        lo == hi || (scale[j] = hi / 2 - lo / 2)
    end
    Z = (X .- transpose(center)) ./ transpose(scale)
    all(isfinite, Z) || throw(ArgumentError("predictor range cannot be normalized in Float64"))
    return Z, center, scale
end

function _continuous_responses(Y)
    n, outputs = size(Y)
    center = vec(sum(Y ./ n; dims=1))
    Z = Y .- transpose(center)
    scale = [norm(view(Z, :, k)) / sqrt(n) for k in 1:outputs]
    for k in 1:outputs
        if all(==(Y[1, k]), view(Y, :, k))
            center[k] = Y[1, k]
            Z[:, k] .= 0
            scale[k] = 1
        end
        iszero(scale[k]) && (scale[k] = 1.0)
        Z[:, k] ./= scale[k]
    end
    all(isfinite, scale) && all(isfinite, Z) ||
        throw(ArgumentError("response range cannot be normalized in Float64"))
    return Z, center, scale
end

function _continuous_thresholds(X, n_thresholds)
    return map(axes(X, 2)) do j
        values = sort!(unique(X[:, j]))
        length(values) == 1 && return Float64[]
        if n_thresholds === nothing
            cuts = [values[i] / 2 + values[i + 1] / 2 for i in 1:length(values)-1]
            return unique!(filter(t -> -1 < t < 1, cuts))
        end
        return collect(range(-1.0, 1.0; length=n_thresholds + 2))[2:end-1]
    end
end

"""
    fit_continuous_tree(X, y; pairs=:none, candidate_search=:full,
        max_splits=6, max_depth=4, min_leaf=8, n_thresholds=3,
        split_penalty=2.0, coefficient_precision=0.01,
        noise_shape=2.0, noise_rate=1.0)

Fit a [`ContinuousTree`](@ref) to finite numeric predictors, with observations
in rows. A vector target gives vector predictions. An `n × q` target matrix
gives matrix predictions with shared tree geometry and independent outputs.

Leaves use an intercept, all predictors, and the interactions selected by
`pairs`: `:none`, `:all`, or a vector such as `[(1, 2), (2, 3)]`. Polynomial
coefficients match across full leaf faces, including partial faces at
T-junctions. Derivatives may jump.

Predictors map their training ranges to `[-1, 1]`. Responses use their training
means and population standard deviations. Constant columns use scale one.
The coefficient prior is isotropic normal with precision
`coefficient_precision / σ²` on an orthonormal basis of continuous raw leaf
coefficients. Independently for each output, `σ²` has an inverse-gamma prior
with `noise_shape` and `noise_rate`. All three hyperparameters must be positive.
This prior depends on the tree geometry and the fitted normalization.

Greedy search compares single splits, paired sibling splits, and three-split
crosses. It sums output log marginal likelihoods and subtracts
`split_penalty` per split. `max_splits` bounds the total number of internal
nodes. `min_leaf` bounds the observed samples in each new leaf. By default,
each feature has `n_thresholds` equally spaced cuts in its training range.
Use `n_thresholds=nothing` for every distinct-value midpoint. `:full` searches
the whole candidate family at each step. `:graph_pruned` skips crosses outside
`pairs`, retaining all single and sibling moves. Neither search finds a
guaranteed global optimum.

Predictions extrapolate the routed leaf polynomial without clipping.
[`predictive`](@ref) gives Student-t marginals conditional on the chosen tree,
interaction graph and normalization, excluding model-selection uncertainty.
"""
function fit_continuous_tree(X::AbstractMatrix,
        y::Union{AbstractVector,AbstractMatrix}; pairs=:none,
        candidate_search=:full, max_splits=6, max_depth=4, min_leaf=8,
        n_thresholds=3, split_penalty=2.0, coefficient_precision=0.01,
        noise_shape=2.0, noise_rate=1.0)
    n, p = size(X)
    n > 0 && p > 0 || throw(ArgumentError("X must have at least one row and one feature"))
    size(y, 1) == n || throw(DimensionMismatch("X and y must have the same number of rows"))
    size(y, 2) > 0 || throw(ArgumentError("y must have at least one output"))
    for (name, value, minimum) in ((:max_splits, max_splits, 0),
            (:max_depth, max_depth, 0), (:min_leaf, min_leaf, 1))
        value isa Integer && value >= minimum ||
            throw(ArgumentError("$name must be an integer at least $minimum"))
    end
    n_thresholds === nothing || (n_thresholds isa Integer && n_thresholds > 0) ||
        throw(ArgumentError("n_thresholds must be a positive integer or nothing"))
    candidate_search in (:full, :graph_pruned) ||
        throw(ArgumentError("candidate_search must be :full or :graph_pruned"))
    for (name, value, positive) in ((:split_penalty, split_penalty, false),
            (:coefficient_precision, coefficient_precision, true),
            (:noise_shape, noise_shape, true), (:noise_rate, noise_rate, true))
        value isa Real && isfinite(value) && (positive ? value > 0 : value >= 0) ||
            throw(ArgumentError("$name must be finite and $(positive ? "positive" : "nonnegative")"))
        converted = Float64(value)
        isfinite(converted) && (positive ? converted > 0 : converted >= 0) ||
            throw(ArgumentError("$name must remain finite and $(positive ? "positive" : "nonnegative") in Float64"))
    end
    selected = _continuous_pairs(pairs, p)
    Z, xcenter, xscale = _continuous_predictors(_continuous_data(X, "X"))
    target = _continuous_data(y, "y")
    Y = y isa AbstractVector ? reshape(target, :, 1) : target
    response, ycenter, yscale = _continuous_responses(Y)
    thresholds = _continuous_thresholds(Z, n_thresholds)
    fitted = Continuous.fit(Z, response; pairs=selected, candidate_search,
        max_splits=Int(max_splits), max_depth=Int(max_depth), min_leaf=Int(min_leaf),
        thresholds, split_penalty=Float64(split_penalty),
        coefficient_precision=Float64(coefficient_precision),
        noise_shape=Float64(noise_shape), noise_rate=Float64(noise_rate))
    return ContinuousTree{y isa AbstractVector}(fitted, xcenter, xscale, ycenter, yscale)
end

function _continuous_input(model::ContinuousTree, X)
    size(X, 2) == length(model.xcenter) || throw(DimensionMismatch("X has the wrong number of features"))
    Z = (_continuous_data(X, "X") .- transpose(model.xcenter)) ./ transpose(model.xscale)
    all(isfinite, Z) || throw(ArgumentError("predictor range cannot be normalized in Float64"))
    return Z
end

_continuous_shape(::ContinuousTree{true}, values) = vec(values)
_continuous_shape(::ContinuousTree{false}, values) = values

"""
    predict(model::ContinuousTree, X)

Return posterior mean predictions. The output is a vector for a vector target
and a matrix for a matrix target, with observations in rows. Outside the
training range, evaluate the routed leaf polynomial without clipping.
"""
function predict(model::ContinuousTree, X::AbstractMatrix)
    values = Continuous.predict(model.fit, _continuous_input(model, X))
    values .*= transpose(model.yscale)
    values .+= transpose(model.ycenter)
    return _continuous_shape(model, values)
end

"""
    predict!(out, model::ContinuousTree, X)

Write posterior mean predictions to `out`, which must match the target rank
and the number of prediction rows. This method uses a temporary prediction
array and supports aliasing between `out` and `X`.
"""
function predict!(out::AbstractArray, model::ContinuousTree, X::AbstractMatrix)
    expected = model isa ContinuousTree{true} ? (size(X, 1),) : (size(X, 1), length(model.ycenter))
    size(out) == expected || throw(DimensionMismatch("out must have size $expected"))
    copyto!(out, predict(model, X))
    return out
end

"""
    predictive(model::ContinuousTree, X; observation=true)

Return `(location, scale2, dof)` for the conditional Student-t predictive
marginals. `observation=true` includes observation noise. Use `false` for
latent-function uncertainty. `scale2` is the Student-t scale squared, not its
variance. When `dof > 2`, variance is `scale2 * dof / (dof - 2)`.

`location` and `scale2` follow the target rank. `dof` is a scalar for a vector
target and one value per output for a matrix target. Marginals condition on the
selected tree, interaction graph and training normalization. They exclude
model-selection uncertainty and do not describe independent draws across rows.
"""
function predictive(model::ContinuousTree{Scalar}, X::AbstractMatrix;
        observation::Bool=true) where {Scalar}
    B = Continuous.design(model.fit, _continuous_input(model, X))
    post = model.fit.post
    location = (B * post.coef) .* transpose(model.yscale) .+ transpose(model.ycenter)
    projected = post.precision.L \ transpose(B)
    leverage = vec(sum(abs2, projected; dims=1)) .+ observation
    scale2 = leverage * transpose((post.rate ./ post.shape) .* model.yscale.^2)
    dof = Scalar ? 2post.shape : fill(2post.shape, length(model.ycenter))
    return (; location=_continuous_shape(model, location),
        scale2=_continuous_shape(model, scale2), dof)
end

function Base.show(io::IO, model::ContinuousTree)
    print(io, "ContinuousTree(", length(model.fit.leaves), " leaves, ",
        length(model.xcenter), " features, ", length(model.ycenter), " outputs)")
end
