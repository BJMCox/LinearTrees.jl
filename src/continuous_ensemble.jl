"""
    ContinuousEnsemble

A finite mixture of continuous trees refitted on one common training set.
`components` hold the conditional trees; `weights` are their posterior model
probabilities under the supplied `logprior`.
"""
struct ContinuousEnsemble{Scalar}
    components::Vector{ContinuousTree{Scalar}}
    weights::Vector{Float64}
    logprior::Vector{Float64}
end

const _ContinuousGeometryLeafKey = Tuple{Vector{Float64},Vector{Float64}}
const _ContinuousGeometryKey =
    Tuple{Vector{_ContinuousGeometryLeafKey},Vector{Vector{Int}}}

function _ensemble_positive(name, value)
    value isa Real && isfinite(value) && value > 0 ||
        throw(ArgumentError("$name must be finite and positive"))
    converted = Float64(value)
    isfinite(converted) && converted > 0 ||
        throw(ArgumentError("$name must remain finite and positive in Float64"))
    return converted
end

function _continuous_vector_less(a::Vector{T}, b::Vector{T}) where {T}
    for j in 1:min(length(a), length(b))
        a[j] == b[j] || return isless(a[j], b[j])
    end
    return length(a) < length(b)
end

function _continuous_box_less(a::_ContinuousGeometryLeafKey,
        b::_ContinuousGeometryLeafKey)
    a[1] == b[1] || return _continuous_vector_less(a[1], b[1])
    return _continuous_vector_less(a[2], b[2])
end

function _continuous_geometry_key(nodes::Vector{Continuous.TreeNode},
        terms::Vector{Vector{Int}})
    boxes = _ContinuousGeometryLeafKey[
        (map(x -> iszero(x) ? 0.0 : x, node.lo),
         map(x -> iszero(x) ? 0.0 : x, node.hi))
        for node in nodes if node.feature == 0]
    sort!(boxes; lt=_continuous_box_less)
    basis = [copy(term) for term in terms]
    sort!(basis; lt=_continuous_vector_less)
    return (boxes, basis)
end

"""
    fit_continuous_ensemble(X, y, geometries; logprior=zeros(length(geometries)),
        coefficient_precision=0.01, noise_shape=2.0, noise_rate=1.0)

Refit each supplied continuous-tree geometry on common normalized `X` and `y`,
then normalize prior-times-evidence weights across this finite model space.
Entries with identical tree geometry and pair-term basis are rejected. Every
geometry must use the normalization computed from `X`.

The posterior is exact conditional on this fixed, supplied model space.
Geometries selected using `y` make it a restricted, adaptive approximation to
a posterior over all continuous trees.
"""
function fit_continuous_ensemble(X::AbstractMatrix,
        y::Union{AbstractVector,AbstractMatrix},
        geometries::AbstractVector{<:ContinuousTree};
        logprior=zeros(length(geometries)), coefficient_precision=0.01,
        noise_shape=2.0, noise_rate=1.0)
    n, p = size(X)
    n > 0 && p > 0 || throw(ArgumentError("X must have at least one row and one feature"))
    size(y, 1) == n || throw(DimensionMismatch("X and y must have the same number of rows"))
    size(y, 2) > 0 || throw(ArgumentError("y must have at least one output"))
    isempty(geometries) && throw(ArgumentError("geometries must be nonempty"))
    λ = _ensemble_positive(:coefficient_precision, coefficient_precision)
    a = _ensemble_positive(:noise_shape, noise_shape)
    b = _ensemble_positive(:noise_rate, noise_rate)
    logprior isa AbstractVector && length(logprior) == length(geometries) ||
        throw(DimensionMismatch("logprior must have one entry per geometry"))
    Base.require_one_based_indexing(logprior)
    all(value -> value isa Real && (isfinite(value) || value == -Inf), logprior) ||
        throw(ArgumentError("logprior entries must be finite or -Inf"))
    priors = Float64.(logprior)
    all(value -> isfinite(value) || value == -Inf, priors) ||
        throw(ArgumentError("logprior entries must remain finite or -Inf in Float64"))

    Z, xcenter, xscale = _continuous_predictors(_continuous_data(X, "X"))
    target = _continuous_data(y, "y")
    Y = y isa AbstractVector ? reshape(target, :, 1) : target
    response, ycenter, yscale = _continuous_responses(Y)
    scalar = y isa AbstractVector
    components = ContinuousTree{scalar}[]
    seen = _ContinuousGeometryKey[]
    scores = Vector{Float64}(undef, length(geometries))
    for (index, source) in enumerate(geometries)
        source.xcenter == xcenter && source.xscale == xscale ||
            throw(ArgumentError("geometry $index uses different predictor normalization"))
        nodes = deepcopy(source.fit.nodes)
        key = _continuous_geometry_key(nodes, source.fit.terms)
        any(previous -> isequal(previous, key), seen) &&
            throw(ArgumentError("duplicate continuous geometry and basis"))
        push!(seen, key)
        selected_pairs = Tuple{Int,Int}[(term[1], term[2]) for term in source.fit.terms
            if length(term) == 2]
        fitted = Continuous.fit_fixed(nodes, Z, response; pairs=selected_pairs,
            coefficient_precision=λ, noise_shape=a, noise_rate=b)
        push!(components, ContinuousTree{scalar}(fitted, copy(xcenter), copy(xscale),
            copy(ycenter), copy(yscale)))
        scores[index] = fitted.post.score + priors[index]
    end
    maximum_score = maximum(scores)
    isfinite(maximum_score) ||
        throw(ArgumentError("at least one geometry must have finite posterior log mass"))
    weights = exp.(scores .- maximum_score)
    weights ./= sum(weights)
    return ContinuousEnsemble{scalar}(components, weights, priors)
end

"Return the posterior-model-weighted conditional means."
function predict(model::ContinuousEnsemble, X::AbstractMatrix; batch_size::Int=1024)
    n, outputs = size(X, 1), length(model.components[1].ycenter)
    location = zeros(Float64, n, outputs)
    for (weight, component) in zip(model.weights, model.components)
        iszero(weight) && continue
        location .+= weight .* reshape(predict(component, X; batch_size), n, outputs)
    end
    return _continuous_shape(model.components[1], location)
end

"Write ensemble mean predictions after reading all of `X`, preserving aliasing."
function predict!(out::AbstractArray, model::ContinuousEnsemble, X::AbstractMatrix;
        batch_size::Int=1024)
    expected = model isa ContinuousEnsemble{true} ?
        (size(X, 1),) : (size(X, 1), length(model.components[1].ycenter))
    size(out) == expected || throw(DimensionMismatch("out must have size $expected"))
    copyto!(out, predict(model, X; batch_size))
    return out
end

"""
    predictive(model::ContinuousEnsemble, X; observation=true, batch_size=1024)

Return mixture `location`, `variance`, `within_variance`, `between_variance`,
posterior `weights`, and conditional Student-t `components`. The mixture is
generally not Student-t. Its variance is infinite when a positive-weight
component has at most two degrees of freedom.
"""
function predictive(model::ContinuousEnsemble{Scalar}, X::AbstractMatrix;
        observation::Bool=true, batch_size::Int=1024) where {Scalar}
    n, outputs = size(X, 1), length(model.components[1].ycenter)
    weights = copy(model.weights)
    components = [predictive(component, X; observation, batch_size)
        for component in model.components]
    location = zeros(Float64, n, outputs)
    within = zeros(Float64, n, outputs)
    between = zeros(Float64, n, outputs)
    for (weight, component) in zip(weights, components)
        iszero(weight) && continue
        location .+= weight .* reshape(component.location, n, outputs)
    end
    for (weight, component) in zip(weights, components)
        iszero(weight) && continue
        member_location = reshape(component.location, n, outputs)
        member_scale2 = reshape(component.scale2, n, outputs)
        for output in 1:outputs
            degrees = Scalar ? component.dof : component.dof[output]
            variance_factor = degrees > 2 ? degrees / (degrees - 2) : Inf
            for row in 1:n
                difference = member_location[row, output] - location[row, output]
                within[row, output] += degrees > 2 ?
                    weight * member_scale2[row, output] * variance_factor : Inf
                between[row, output] += weight * difference^2
            end
        end
    end
    return (; location=_continuous_shape(model.components[1], location),
        variance=_continuous_shape(model.components[1], within + between),
        within_variance=_continuous_shape(model.components[1], within),
        between_variance=_continuous_shape(model.components[1], between),
        weights, components)
end

function Base.show(io::IO, model::ContinuousEnsemble)
    print(io, "ContinuousEnsemble(", length(model.components), " components, ",
        length(model.components[1].ycenter), " outputs)")
end
