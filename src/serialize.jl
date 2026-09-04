# Model serialization to/from disk.

"""
    to_dict(tree)

JSON-friendly form: nodes as vectors of field dictionaries, masks as
integers, loss as a name plus parameters, bounds and base as numbers or
vectors. A `LIN` or leaf node's `threshold` is `NaN` on the struct; since
`NaN` has no JSON literal, it is stored as `nothing` and restored to `NaN`
by `from_dict`.
"""
function to_dict(t::LinearTree{T,V}) where {T,V}
    vec_or_num(v) = v isa SVector ? collect(v) : v
    nodefield(n, f) = f == :model ? Int(getfield(n, f)) :
        f == :threshold ? (isnan(getfield(n, f)) ? nothing : getfield(n, f)) :
        vec_or_num(getfield(n, f))
    nodes = [Dict{String,Any}(String(f) => nodefield(n, f) for f in fieldnames(Node)) for n in t.nodes]
    return Dict{String,Any}("format" => 1, "T" => string(T), "K" => V <: SVector ? length(V) + 1 : 1,
        "nodes" => nodes, "catmasks" => collect(t.catmasks), "loss" => lossdict(t.loss),
        "lo" => vec_or_num(t.lo), "hi" => vec_or_num(t.hi), "base" => vec_or_num(t.base),
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
    conv(v) = V(v)
    nodes = [Node{T,V}(Int32(n["feature"]), n["threshold"] === nothing ? T(NaN) : T(n["threshold"]),
        Int32(n["left"]), Int32(n["right"]),
        conv(n["lcoef"]), conv(n["lintercept"]), conv(n["rcoef"]), conv(n["rintercept"]),
        T(n["xmin"]), T(n["xmax"]), T(n["cover"]), T(n["xmean"]), T(n["gain"]),
        Int32(n["catstart"]), Int32(n["catwords"]), ModelKind(n["model"])) for n in d["nodes"]]
    loss = lossfromdict(d["loss"])
    return LinearTree{T,V,typeof(loss)}(nodes, UInt64.(d["catmasks"]), loss, conv(d["lo"]), conv(d["hi"]),
        conv(d["base"]), Int(d["nfeatures"]), Bool(d["truncate"]))
end
