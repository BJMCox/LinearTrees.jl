# StatsAPI interface implementation and Tables.jl input.

using Tables, DataAPI

"Coefficient count for `n`, for `dof`. `V <: SVector` multiplies by its length (Softmax)."
ncoef(n::Node{T,V}) where {T,V} = (n.model == CON ? 1 : n.model == LIN ? 2 : n.model == PCON ? 2 :
    n.model == BLIN ? 3 : 4) * (V <: SVector ? length(V) : 1)

"""
    TableEncoder

Maps a matrix or a Tables.jl table to the `Float64` matrix `fit_tree` needs.
`names` is the column order; `categorical` lists the 1-based column indices
backed by a `DataAPI.refpool` (e.g. a `CategoricalArray`); `levels[j]` is the
sorted level list for a categorical column `j` (empty otherwise), sorted by
`string` so that two columns with the same values in a different pool order
encode identically. `unseen` is `:error` (default) or `:right`, applied at
`encode` time to a level absent from `levels[j]`.
"""
struct TableEncoder
    names::Vector{Symbol}
    categorical::Vector{Int}
    levels::Vector{Vector{Any}}
    unseen::Symbol
end

"Pass-through encoder for a plain matrix: no categorical columns, names `x1, x2, ...`."
function TableEncoder(X::AbstractMatrix, unseen)
    p = size(X, 2)
    TableEncoder([Symbol("x", j) for j in 1:p], Int[], [Any[] for _ in 1:p], unseen)
end

function TableEncoder(table, unseen)
    Tables.istable(table) || throw(ArgumentError("X must be a matrix or a Tables.jl table"))
    cols = Tables.columns(table)
    names = collect(Tables.columnnames(cols))
    categorical = Int[]
    levels = Vector{Vector{Any}}()
    for (j, nm) in enumerate(names)
        col = Tables.getcolumn(cols, nm)
        pool = DataAPI.refpool(col)
        if pool === nothing
            push!(levels, Any[])
        else
            push!(categorical, j)
            # `DataAPI.levels`, not the pool's own ref order, so two arrays with the
            # same values but different pool orders (e.g. a re-leveled CategoricalArray)
            # produce the same codes; sorted by `string` for a total order across types.
            push!(levels, sort!(Any[v for v in DataAPI.levels(col) if v !== missing]; by = string))
        end
    end
    return TableEncoder(names, categorical, levels, unseen)
end

function encode(::TableEncoder, X::AbstractMatrix)
    any(ismissing, X) && throw(ArgumentError("missing values are not supported"))
    return Matrix{Float64}(X)
end

"""
    encode(enc, table) -> Matrix{Float64}

Apply `enc`'s stored level maps to `table`. Columns are independent, so the
column loop threads when there are at least `PARALLEL_MIN_ROWS` rows; each
column writes only its own slice of the output.
"""
function encode(enc::TableEncoder, table)
    cols = Tables.columns(table)
    n = length(Tables.getcolumn(cols, enc.names[1]))
    out = Matrix{Float64}(undef, n, length(enc.names))
    if n >= PARALLEL_MIN_ROWS
        Threads.@threads for j in eachindex(enc.names)
            encode_column!(out, enc, cols, j)
        end
    else
        for j in eachindex(enc.names)
            encode_column!(out, enc, cols, j)
        end
    end
    return out
end

function encode_column!(out::Matrix{Float64}, enc::TableEncoder, cols, j)
    nm = enc.names[j]
    col = Tables.getcolumn(cols, nm)
    any(ismissing, col) && throw(ArgumentError("column $nm has missing values"))
    if j in enc.categorical
        lv = enc.levels[j]
        code = Dict(v => i for (i, v) in enumerate(lv))
        for i in eachindex(col)
            v = col[i]
            c = get(code, v, 0)
            if c == 0
                enc.unseen == :right || throw(ArgumentError("unseen level $v in column $nm"))
                c = length(lv) + 1   # beyond every node's mask, so it routes right like an unseen level at predict time
            end
            out[i, j] = c
        end
    else
        for i in eachindex(col)
            out[i, j] = Float64(col[i])
        end
    end
    return out
end

"Turn a single feature row `x` into what `encode(enc, ...)` expects: a `1 × p` matrix for a pass-through encoder, a one-row table otherwise."
# `isempty(enc.categorical)` stands in for "pass-through or table without
# categorical columns": both encode a matrix identically, since the matrix
# method never reads names or levels. Revisit if a third encoder kind appears.
reshape_row(enc::TableEncoder, x::AbstractVector) = isempty(enc.categorical) ? reshape(collect(x), 1, length(x)) :
    NamedTuple{Tuple(enc.names)}(Tuple(Any[v] for v in x))

"""
    LinearTreeRegressorFit{Tr}

A [`fit_tree`](@ref) result wrapped with its input encoder and training data
as a `StatsAPI.RegressionModel`. Build with
`fit(LinearTreeRegressorFit, X, y; loss=MSE(), weights=nothing, unseen=:error, kwargs...)`,
where `X` is a matrix or a Tables.jl table and `kwargs` forward to `fit_tree`.
"""
struct LinearTreeRegressorFit{Tr<:LinearTree} <: StatsAPI.RegressionModel
    tree::Tr
    encoder::TableEncoder
    X::Matrix{Float64}
    y::Vector{Float64}
    w::Vector{Float64}
end

"""
    LinearTreeClassifierFit{Tr,C}

A [`fit_tree`](@ref) result for a categorical target, wrapped as a
`StatsAPI.StatisticalModel`. Build with
`fit(LinearTreeClassifierFit, X, y; weights=nothing, unseen=:error, kwargs...)`.
`classes = sort(unique(y))` fixes both the label-to-code map and the column
order of `predict`'s probability matrix. Two classes fit a single
[`Logistic`](@ref) tree with `classes[1]` (the first sorted class) as the
positive outcome; `predict` returns `hcat(p1, 1 .- p1)`, so column `k` is
still `P(classes[k])`. Three or more classes fit a [`Softmax`](@ref) tree
with `classes[k]` at coordinate `k`.
"""
struct LinearTreeClassifierFit{Tr<:LinearTree,C} <: StatsAPI.StatisticalModel
    tree::Tr
    encoder::TableEncoder
    X::Matrix{Float64}
    y::Vector{Int}
    w::Vector{Float64}
    classes::Vector{C}
end

"""
    fit(LinearTreeRegressorFit, X, y; loss=MSE(), weights=nothing, unseen=:error, kwargs...)
    fit(LinearTreeClassifierFit, X, y; weights=nothing, unseen=:error, kwargs...)

Fit a [`LinearTreeRegressorFit`](@ref) or [`LinearTreeClassifierFit`](@ref)
from a matrix or Tables.jl table `X` and a target `y`. `kwargs` forward to
[`fit_tree`](@ref). `unseen` is `:error` (default) or `:right`, applied to a
categorical level absent from training at predict time.
"""
function StatsAPI.fit(::Type{LinearTreeRegressorFit}, X, y; loss::Loss = MSE(), weights = nothing, unseen = :error, kwargs...)
    enc = TableEncoder(X, unseen)
    Xm = encode(enc, X)
    w = weights === nothing ? ones(length(y)) : Vector{Float64}(weights)
    tree = fit_tree(Xm, y, loss; weights = w, categorical = enc.categorical, kwargs...)
    return LinearTreeRegressorFit(tree, enc, Xm, Vector{Float64}(y), w)
end

function StatsAPI.fit(::Type{LinearTreeClassifierFit}, X, y; weights = nothing, unseen = :error, kwargs...)
    enc = TableEncoder(X, unseen)
    Xm = encode(enc, X)
    sorted = sort(unique(y))
    # `Vector{eltype(sorted)}`, not `collect`: a `CategoricalArray`'s own `sort`/`unique`
    # stay `CategoricalArray`-typed (not a `Vector`, so it can't match the struct's
    # `classes::Vector{C}` field), but converting element-by-element to a `Vector`
    # keeps each `CategoricalValue` (and the pool it carries) intact.
    classes = Vector{eltype(sorted)}(sorted)
    K = length(classes)
    code = Dict(c => i for (i, c) in enumerate(classes))
    yi = [code[v] for v in y]
    w = weights === nothing ? ones(length(y)) : Vector{Float64}(weights)
    loss = K == 2 ? Logistic() : Softmax(K)
    ytarget = K == 2 ? Float64.(yi .== 1) : yi
    tree = fit_tree(Xm, ytarget, loss; weights = w, categorical = enc.categorical, kwargs...)
    return LinearTreeClassifierFit(tree, enc, Xm, yi, w, classes)
end

StatsAPI.predict(m::LinearTreeRegressorFit, X) = predict(m.tree, encode(m.encoder, X))

function StatsAPI.predict(m::LinearTreeClassifierFit, X)
    Xm = encode(m.encoder, X)
    if m.tree.loss isa Logistic
        p1 = predict(m.tree, Xm)
        return hcat(p1, 1 .- p1)
    end
    return predict(m.tree, Xm)
end

const AnyFit = Union{LinearTreeRegressorFit,LinearTreeClassifierFit}

"Number of rows passed to `fit`, zero-weight rows included."
StatsAPI.nobs(m::AnyFit) = length(m.y)

"Training row weights, as stored (ones when `fit` was called with no `weights`)."
StatsAPI.weights(m::AnyFit) = m.w

"""
    dof(m)

Count of stored coefficients across every node: 1 for `CON`, 2 for `LIN` or
`PCON`, 3 for `BLIN`, 4 for `PLIN`, times `K-1` for a `Softmax` fit. A
parameter count, not an effective degrees of freedom, and unrelated to
`BIC`'s per-kind selection penalty of the same name.
"""
StatsAPI.dof(m::AnyFit) = sum(ncoef(n) for n in m.tree.nodes)

"Training residuals `y - predict(tree, X)`, on the response scale."
StatsAPI.residuals(m::LinearTreeRegressorFit) = m.y .- predict(m.tree, m.X)
StatsAPI.deviance(m::LinearTreeRegressorFit) = deviance(m.tree.loss, m.y, score(m.tree, m.X), m.w)
StatsAPI.coeftable(m::AnyFit, x) = coeftable(m.tree, vec(encode(m.encoder, reshape_row(m.encoder, x))))
feature_importance(m::AnyFit) = feature_importance(m.tree)
