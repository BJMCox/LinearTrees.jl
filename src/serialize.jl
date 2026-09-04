# Model serialization to/from disk.

"""
    jsonnum(x)

`x` on a JSON-safe encoding: a finite `Real` as `Float64`, a non-finite
`Real` (`Inf`, `-Inf`, `NaN`, none of which have a JSON literal) as the
matching string, and an `SVector` mapped elementwise into a `Vector{Any}`.
"""
jsonnum(x::Real) = isfinite(x) ? Float64(x) : string(x)
jsonnum(x::SVector) = Any[jsonnum(c) for c in x]

"""
    fromjsonnum(T, x)

Inverse of the scalar case of [`jsonnum`](@ref): a number converts to `T`
directly, and one of the strings `"Inf"`, `"-Inf"`, `"NaN"` parses back to
the matching non-finite `T`. Broadcast over an `SVector`-encoded field.
"""
fromjsonnum(::Type{T}, x::Real) where {T} = T(x)
fromjsonnum(::Type{T}, x::AbstractString) where {T} =
    x == "Inf" ? T(Inf) : x == "-Inf" ? T(-Inf) : x == "NaN" ? T(NaN) :
    throw(ArgumentError("unrecognised numeric string \"$x\""))

"""
    to_dict(tree)

JSON-friendly form: nodes as vectors of field dictionaries, loss as a name
plus parameters. Three encodings keep every field JSON-safe:
- Every `Real`-valued field (`threshold, lcoef, lintercept, rcoef,
  rintercept, xmin, xmax, cover, xmean, gain, lo, hi, base`) goes through
  [`jsonnum`](@ref): finite values become `Float64`, non-finite values
  become the string `"Inf"`, `"-Inf"`, or `"NaN"`, and an `SVector` field
  maps elementwise into a `Vector{Any}`.
- Integer fields (`feature, left, right, catstart, catwords, model`) store
  as `Int`, restored to their struct field type (`Int32` or `ModelKind`) by
  `from_dict`.
- `catmasks` (`UInt64`, may use the top bit) store as lowercase hex strings
  via `string(m; base = 16)`, since a plain integer would overflow `Int64`
  and lose exactness in a JSON parser; restored with `parse(UInt64, s; base
  = 16)`.
"""
function to_dict(t::LinearTree{T,V}) where {T,V}
    nodefield(n, f) = f in (:feature, :left, :right, :catstart, :catwords, :model) ?
        Int(getfield(n, f)) : jsonnum(getfield(n, f))
    nodes = [Dict{String,Any}(String(f) => nodefield(n, f) for f in fieldnames(Node)) for n in t.nodes]
    return Dict{String,Any}("format" => 1, "T" => string(T), "K" => V <: SVector ? length(V) + 1 : 1,
        "nodes" => nodes, "catmasks" => [string(m; base = 16) for m in t.catmasks], "loss" => lossdict(t.loss),
        "lo" => jsonnum(t.lo), "hi" => jsonnum(t.hi), "base" => jsonnum(t.base),
        "nfeatures" => t.nfeatures, "truncate" => t.truncate)
end

lossdict(l::Loss) = Dict{String,Any}("name" => string(nameof(typeof(l))),
    "params" => Dict{String,Any}(String(f) => getfield(l, f) for f in fieldnames(typeof(l))))
lossdict(l::AdaptedLoss) = throw(ArgumentError(
    "AdaptedLoss wraps a LossFunctions.jl loss and cannot round-trip through to_dict"))

"""
    lossfromdict(d)

Rebuild a `Loss` from a [`lossdict`](@ref) dict. Dispatches on the loss name
as a `Val` rather than looking a `DataType` up and reflecting on it, so each
loss's constructor call is concretely typed and JET-clean.
"""
lossfromdict(d) = _lossfromname(Val(Symbol(d["name"])), d["params"])

_lossfromname(::Val{:MSE}, ps) = MSE()
_lossfromname(::Val{:Huber}, ps) = Huber(ps["δ"])
_lossfromname(::Val{:Quantile}, ps) = Quantile(ps["τ"])
_lossfromname(::Val{:MAD}, ps) = MAD()
_lossfromname(::Val{:Logistic}, ps) = Logistic()
_lossfromname(::Val{:Poisson}, ps) = Poisson()
_lossfromname(::Val{:NegBin}, ps) = NegBin(ps["θ"])
_lossfromname(::Val{:Gamma}, ps) = Gamma()
_lossfromname(::Val{:Tweedie}, ps) = Tweedie(ps["ρ"])
_lossfromname(::Val{:Softmax}, ps) = Softmax(ps["K"])
_lossfromname(::Val{S}, ps) where {S} = throw(ArgumentError("unknown loss name \"$S\""))

"""
    from_dict(d) -> LinearTree

Inverse of [`to_dict`](@ref). Throws `ArgumentError` for an unrecognised
format version or loss name.
"""
function from_dict(d::AbstractDict)
    d["format"] == 1 || throw(ArgumentError("unknown LinearTrees dict format $(d["format"])"))
    T = d["T"] == "Float32" ? Float32 : Float64
    K = d["K"]
    V = K == 1 ? T : SVector{K - 1,T}
    convV(x) = V <: SVector ? V(fromjsonnum.(T, x)) : fromjsonnum(V, x)
    nodes = [Node{T,V}(Int32(n["feature"]), fromjsonnum(T, n["threshold"]),
        Int32(n["left"]), Int32(n["right"]),
        convV(n["lcoef"]), convV(n["lintercept"]), convV(n["rcoef"]), convV(n["rintercept"]),
        fromjsonnum(T, n["xmin"]), fromjsonnum(T, n["xmax"]), fromjsonnum(T, n["cover"]),
        fromjsonnum(T, n["xmean"]), fromjsonnum(T, n["gain"]),
        Int32(n["catstart"]), Int32(n["catwords"]), ModelKind(Int(n["model"]))) for n in d["nodes"]]
    loss = lossfromdict(d["loss"])
    catmasks = [parse(UInt64, s; base = 16) for s in d["catmasks"]]
    return LinearTree{T,V,typeof(loss)}(nodes, catmasks, loss, convV(d["lo"]), convV(d["hi"]),
        convV(d["base"]), Int(d["nfeatures"]), Bool(d["truncate"]))
end
