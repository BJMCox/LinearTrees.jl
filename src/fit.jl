# Tree growth: node splitting and stopping rules.

"""
One worker's per-level buffers for `scan_categorical`, one entry per level of
the categorical column in hand. They sit inside the worker's `Scratch`, so the
task that owns that set is their only writer, and they grow to the widest
categorical column that worker meets.
"""
struct LevelSums{V}
    sz::Vector{V}           # Σ h·z per level
    sw::Vector{V}           # Σ h per level
    counts::Vector{Int}     # rows per level, offset by one so it doubles as the counting sort's histogram
    cumoffset::Vector{Int}  # where each level's bucket starts
    cursor::Vector{Int}     # the bucket write positions, as the counting sort fills them
    rank::Vector{Int32}     # each level's place in the current order
    order::Vector{Int}      # the present levels, sorted by mean working response
    present::Vector{Int}    # the levels the node's rows actually use
end
LevelSums{V}() where {V} = LevelSums{V}(V[], V[], Int[], Int[], Int[], Int32[], Int[], Int[])

"""
One worker's buffers. A worker index (`tid`) picks the set a task owns, and no
two concurrent tasks are ever given the same one, so nothing here needs a
lock. Each buffer is reused for several unrelated purposes over a node's life;
the docstring of every user says why its own use is free at that point.

Every buffer starts empty and `ensure_len!` grows it to the largest node it is
used on. A set the fit only ever gives to small subtrees therefore never
reaches length `n`, which is what keeps `SCRATCH_PER_THREAD` sets per thread
affordable. `ids` is the worker-id list one node's `best_split` and
`partition!` are called with; it belongs to the set because the task that owns
the set is its only writer.
"""
struct Scratch{T,V,B}
    xs::Vector{T}
    zs::Vector{V}
    hs::Vector{V}
    ws::Vector{T}
    perm::Vector{Int32}
    ids::Vector{Int}
    levels::LevelSums{V}
    search::B
end
function Scratch{T,V}(search::SplitSearch) where {T,V}
    buffer = search_workspace(search, T, V)
    return Scratch{T,V,typeof(buffer)}(T[], V[], V[], T[], Int32[], Int[], LevelSums{V}(), buffer)
end
Scratch{T,V}() where {T,V} = Scratch{T,V}(ExactSearch())

"""
Scratch buffers are sized on demand, never shrunk, and only ever by the one
task that owns the set, so this is the whole of their allocation policy.
Returns `v`, which `partition!` uses to size and name a buffer in one step.
"""
@inline function ensure_len!(v::Vector, m::Integer)
    length(v) < m && resize!(v, m)
    return v
end

"Grow one worker's level buffers to `L` levels. `counts` and the two offsets derived from it are indexed `1:L+1`."
function ensure_levels!(lv::LevelSums, L::Integer)
    ensure_len!(lv.sz, L); ensure_len!(lv.sw, L); ensure_len!(lv.rank, L)
    ensure_len!(lv.order, L); ensure_len!(lv.present, L)
    ensure_len!(lv.counts, L + 1); ensure_len!(lv.cumoffset, L + 1); ensure_len!(lv.cursor, L + 1)
    return lv
end

"The empty left-level list every non-categorical and non-`pcon` scan returns. Shared, because no caller writes to it."
const NO_LEVELS = Int[]

"""
The scratch ids no task owns, guarded by a lock. `trytake!` never blocks: a
task that finds the pool empty grows both children inline instead of waiting,
so no task ever waits on a buffer that a task below it has to release, and
growth cannot deadlock.
"""
struct ScratchPool
    lock::ReentrantLock
    free::Vector{Int}
end
ScratchPool(ids) = ScratchPool(ReentrantLock(), collect(Int, ids))

"Take one free scratch id, or `0` when none is free. Never blocks."
function trytake!(p::ScratchPool)
    lock(p.lock)
    try
        return isempty(p.free) ? 0 : pop!(p.free)
    finally
        unlock(p.lock)
    end
end

"Return one id taken by `trytake!`. The caller must no longer use that set."
function give!(p::ScratchPool, id::Int)
    lock(p.lock)
    try
        push!(p.free, id)
    finally
        unlock(p.lock)
    end
    return nothing
end

"""
Append up to `k` free ids to `ids` and return it. Fewer than `k` is normal:
the caller then threads over what it got. `ids` is the caller's own buffer
(`st.scratch[tid].ids`), so a node that borrows nothing allocates nothing.
"""
function borrow!(p::ScratchPool, ids::Vector{Int}, k::Integer)
    k >= 1 || return ids
    lock(p.lock)
    try
        for _ in 1:k
            isempty(p.free) && break
            push!(ids, pop!(p.free))
        end
    finally
        unlock(p.lock)
    end
    return ids
end

"""
Return every id in `ids` past the first, which is the caller's own and stays.
Paired with `worker_ids!` in a `try`/`finally`, so an exception inside the
threaded call it fed strands no id.
"""
function giveback!(p::ScratchPool, ids::Vector{Int})
    length(ids) > 1 || return nothing
    lock(p.lock)
    try
        for k in 2:length(ids)
            push!(p.free, ids[k])
        end
    finally
        unlock(p.lock)
    end
    resize!(ids, 1)
    return nothing
end

"""
This task's own scratch id first, then up to `k` borrowed ones, in the owner's
own `ids` buffer. Owner first because every threaded consumer falls back to
`first(ids)` when it decides not to thread.
"""
function worker_ids!(sc::Scratch, p::ScratchPool, tid::Int, k::Integer)
    ids = sc.ids
    empty!(ids)
    push!(ids, tid)
    return borrow!(p, ids, k)
end

"Scratch sets per worker. Two, because a task blocked in `wait` still owns its set and would otherwise deny a runnable task any buffers."
const SCRATCH_PER_THREAD = 2

"Partition indices and worker buffers reused across sequential boosting rounds of the same size."
struct TreeWorkspace{T,V,B,I}
    idx::Matrix{Int32}
    scratch::Vector{Scratch{T,V,B}}
    sampled_ids::I
end

function TreeWorkspace{T,V}(n, p, nthreads, search::SplitSearch) where {T,V}
    nsets = nthreads == 1 ? 1 : SCRATCH_PER_THREAD * nthreads
    scratch = [Scratch{T,V}(search) for _ in 1:nsets]
    return TreeWorkspace(index_workspace(search, n, p), scratch,
        sampled_index_workspace(search, n, p))
end
TreeWorkspace{T,V}(n, p, nthreads) where {T,V} = TreeWorkspace{T,V}(n, p, nthreads, ExactSearch())

"Row-count gate for sibling-subtree parallelism: a split whose right child has at least this many rows may run as its own task."
const SUBTREE_MIN_ROWS = 256

"""
A node's rows: the slice of `st.roworder` that `partition!` left it. Spelled out
rather than left as an `AbstractVector`, because every node in every fit has
exactly this type and an abstract one costs `grow_subtree` its concrete inference.
"""
const RowView = SubArray{Int32,1,Vector{Int32},Tuple{UnitRange{Int}},true}

"""
Everything one `fit_tree` call carries through growth. Built by keyword
(`Base.@kwdef`) because the positional form is twenty arguments wide and a
field reorder in it would corrupt a fit silently.
"""
Base.@kwdef mutable struct FitState{T,V,Y,L<:Loss,R<:SelectionRule,S<:SplitSearch,B}
    X::Matrix{T}
    y::Vector{Y}
    w::Vector{T}
    f::Vector{V}
    g::Vector{V}
    h::Vector{V}
    z::Vector{V}
    idx::Matrix{Int32}          # n × p, selected columns hold feature-sorted row orders, partitioned node by node
    roworder::Vector{Int32}     # n, the same rows in the order every node's sums run over, partitioned alongside `idx`
    isleft::Vector{Bool}        # n, per-row left marker used by `partition!`; each node touches only its own rows
    scratch::Vector{Scratch{T,V,B}}   # SCRATCH_PER_THREAD per worker, one when nthreads == 1
    pool::ScratchPool = ScratchPool(Int[])   # the ids of `scratch` no task owns, so the two are built and live together
    nodes::Vector{Node{T,V}}
    catmasks::Vector{UInt64}
    iscat::Vector{Bool}         # length p, true for the columns `fit_tree` was given as categorical
    nlevels::Vector{Int}
    features::Vector{Int}       # sorted unique feature indices considered for split scans
    loss::L
    rule::R
    split_search::S
    lo::V
    hi::V
    max_depth::Int
    min_fit::T
    min_leaf::T
    min_sum_hessian::T
    max_lin_chain::Int
    truncate::Bool
    nthreads::Int
    niter::Int
    # `h` is exactly one on every row: unweighted MSE or a Frozen target whose
    # supplied Hessian and frequency weight have that effective value. The split
    # scan then reads `hs` as `UnitHessians` and accumulates without a multiply.
    unith::Bool
end

function FitState{T,V,Y,L,R}(; scratch, split_search::SplitSearch = ExactSearch(), kwargs...) where {T,V,Y,L<:Loss,R<:SelectionRule}
    B = eltype(scratch).parameters[3]
    return FitState{T,V,Y,L,R,typeof(split_search),B}(; scratch, split_search, kwargs...)
end

"""
    fit_tree(X, y, loss=MSE(); kwargs...) -> LinearTree

Fit a linear model tree to the rows of numeric matrix `X` and target vector `y`.
Columns are features. Targets must satisfy the domain of `loss`; feature values
must be finite. Use [`predict`](@ref) for response-scale predictions and
[`score`](@ref) for scores on the loss scale.

# Keywords

- `weights=nothing`: nonnegative frequency weights, with a positive total.
  The default gives every row weight one. Zero-weight rows are excluded.
- `categorical=Int[]`: columns containing positive integer category codes.
- `rule=BIC()`: node selection rule. See [`BIC`](@ref) and [`GainRule`](@ref).
- `split_search=ExactSearch()`: numeric threshold search. [`BinnedSearch`](@ref)
  and [`HybridSearch`](@ref) offer approximate search for scalar-score losses.
- `max_depth=12`: maximum branching depth. Unsplit `LIN` nodes add no depth.
- `min_fit=10`: minimum total row weight for another model-selection step.
- `min_leaf=5`: minimum total row weight on each side of a split.
- `min_sum_hessian=1.0`: stop below this total Hessian, summed over score coordinates.
- `max_lin_chain=10`: maximum number of consecutive unsplit linear nodes.
- `truncate=true`: bound score accumulation and each node's feature extrapolation.
- `truncation_factor=3`: parameter passed to [`scorebound`](@ref). Must be at least one.
- `features=1:size(X, 2)`: feature columns available for fitting. Prediction
  still expects the original matrix layout.
- `nthreads=Threads.nthreads()`: maximum number of available Julia threads to use.
- `niter=5`: node-refit iterations for non-smooth losses such as `MAD` and
  `Quantile`. Smooth losses ignore this keyword.
- `presort=nothing`: optional `Int32` matrix of stable sorted row indices,
  one column per original feature. Used by exact and local-bin search.

Nodes add score increments along a path. Model selection uses a quadratic
surrogate for nonquadratic losses. Logistic updates also check true training
loss before accepting a full step. No pruning pass follows growth.

For table inputs and stored encoders, use [`LinearTreeRegressorFit`](@ref) or
[`LinearTreeClassifierFit`](@ref) through [`fit`](@ref).
"""
function fit_tree(X::AbstractMatrix, y::AbstractVector, loss::Loss = MSE();
        weights = nothing, categorical = Int[], rule::SelectionRule = BIC(),
        split_search::SplitSearch = ExactSearch(),
        max_depth = 12, min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
        max_lin_chain = 10, truncate = true, truncation_factor = 3,
        features = 1:size(X, 2), presort::Union{Nothing,AbstractMatrix{Int32}} = nothing,
        nthreads = Threads.nthreads(), niter = 5)
    return _fit_tree(X, y, loss, nothing; weights, categorical, rule, max_depth, min_fit, min_leaf,
        min_sum_hessian, max_lin_chain, truncate, truncation_factor, features, presort, nthreads, niter, split_search)
end

function _fit_tree(X::AbstractMatrix, y::AbstractVector, loss::Loss, workspace;
        weights = nothing, categorical = Int[], rule::SelectionRule = BIC(),
        split_search::SplitSearch = ExactSearch(),
        max_depth = 12, min_fit = 10, min_leaf = 5, min_sum_hessian = 1.0,
        max_lin_chain = 10, truncate = true, truncation_factor = 3,
        features = 1:size(X, 2), presort::Union{Nothing,AbstractMatrix{Int32}} = nothing,
        nthreads = Threads.nthreads(), niter = 5)
    # `niter` reaches an `Int` field, so a non-integer would surface as an
    # `InexactError` from deep inside the fit rather than as a rejected argument
    isinteger(niter) && niter >= 1 || throw(ArgumentError("niter must be an integer >= 1, got $niter"))
    isinteger(max_depth) && max_depth >= 0 || throw(ArgumentError("max_depth must be an integer >= 0, got $max_depth"))
    isinteger(max_lin_chain) && max_lin_chain >= 1 || throw(ArgumentError("max_lin_chain must be an integer >= 1, got $max_lin_chain"))
    min_fit >= 1 || throw(ArgumentError("min_fit must be >= 1, got $min_fit"))
    min_leaf >= 1 || throw(ArgumentError("min_leaf must be >= 1, got $min_leaf"))
    min_sum_hessian >= 0 || throw(ArgumentError("min_sum_hessian must be >= 0, got $min_sum_hessian"))
    # below 1 the padding term goes negative, so the clamp band closes inside the
    # observed range of `y` and every extreme score is pulled toward the middle
    truncation_factor >= 1 || throw(ArgumentError("truncation_factor must be >= 1, got $truncation_factor"))
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    T = float(promote_type(eltype(X), target_eltype(loss, y)))
    V = coeftype(loss, T)
    V <: Real || split_search isa ExactSearch ||
        throw(ArgumentError("$(typeof(split_search)) requires scalar coefficients; use ExactSearch() with $(typeof(loss))"))
    n, p = size(X)
    feats = sort!(unique!(collect(Int, features)))
    isempty(feats) && throw(ArgumentError("features must not be empty"))
    all(j -> 1 <= j <= p, feats) || throw(ArgumentError("features must lie in 1:$p"))
    presort === nothing || size(presort) == (n, p) ||
        throw(DimensionMismatch("presort is $(size(presort)), X is $n × $p"))
    length(y) == n || throw(DimensionMismatch("X has $n rows, y has $(length(y))"))
    validate_target(loss, y)
    w = weights === nothing ? ones(T, n) : Vector{T}(weights)
    all(v -> isfinite(v) && v >= 0, w) || throw(ArgumentError("weights must be finite and non-negative"))
    keep = findall(>(0), w)
    isempty(keep) && throw(ArgumentError("total weight must be positive"))
    # No row is dropped and the caller already has the working element type, so
    # the copy would be pure cost. `st.X` is read-only for the whole fit, so the
    # tree holds a reference to the caller's matrix only until `fit_tree` returns.
    Xm = length(keep) == n && X isa Matrix{T} ? X : Matrix{T}(view(X, keep, :))
    yv = prepare_target(loss, y, keep, T); w = w[keep]
    all(isfinite, Xm) || throw(ArgumentError("X contains NaN or Inf"))
    nlevels = zeros(Int, p)
    iscat = zeros(Bool, p)
    for j in categorical
        1 <= j <= p || throw(ArgumentError("categorical column $j is outside 1:$p"))
        col = view(Xm, :, j)
        all(x -> isfinite(x) && x >= 1 && x == round(x), col) ||
            throw(ArgumentError("categorical column $j must hold integer codes >= 1"))
        nlevels[j] = Int(maximum(col))
        iscat[j] = true
    end
    prepared_search = prepare_search(split_search, Xm, feats, iscat, keep, workspace, nthreads)
    return _fit_tree(Xm, yv, w, loss, rule, V, iscat, nlevels, keep, feats, presort, workspace;
        max_depth, min_fit, min_leaf, min_sum_hessian, max_lin_chain, truncate, truncation_factor,
        nthreads, niter, split_search = prepared_search)
end

"""
Unbounded score interval in `V`. `V(-Inf)` is not a constructor for an
`SVector` score, so the interval is built from `zero(V)` for every score type.
"""
function infbounds(::Type{V}) where {V}
    z = zero(V)
    return (oftype(z, z .- Inf), oftype(z, z .+ Inf))
end

"""
The body of `fit_tree` once every argument is concrete: `Xm::Matrix{T}`,
`yv::Vector{Y}`, `w::Vector{T}`, a concrete loss and rule, and coefficient type `V`.
A function barrier, so growth specializes on those types instead of
re-dispatching on them at run time in every node. `fit_tree` stays the
validating front end.
"""
function _fit_tree(Xm::Matrix{T}, yv::Vector{Y}, w::Vector{T}, loss::L, rule::R, ::Type{V},
        iscat::Vector{Bool}, nlevels::Vector{Int}, keep::Vector{Int}, features::Vector{Int},
        presort::Union{Nothing,AbstractMatrix{Int32}}, workspace;
        max_depth, min_fit, min_leaf, min_sum_hessian, max_lin_chain, truncate, truncation_factor,
        nthreads, niter, split_search::S) where {T,V,Y,L<:Loss,R<:SelectionRule,S<:SplitSearch}
    V <: Real || split_search isa ExactSearch ||
        throw(ArgumentError("$(typeof(split_search)) requires scalar coefficients; use ExactSearch() with $(typeof(loss))"))
    n, p = size(Xm)
    f0 = V(initscore(loss, yv, w))
    # every `scorebound` method returns bounds in its own working type (often
    # `Float64`, regardless of `V`), so convert here rather than trust each method
    lo, hi = truncate ? map(V, scorebound(loss, yv; truncation_factor)) : infbounds(V)
    f = fill(clampscore(f0, lo, hi), n)
    workspace === nothing && (workspace = TreeWorkspace{T,V}(n, p, nthreads, split_search))
    idx = workspace.idx
    initialize_index!(idx, Xm, presort, keep, features, nthreads, split_search)
    # Models own their nodes and masks. Only fully overwritten work buffers are
    # shared across rounds; a fresh pool starts each fit with no borrowed ids.
    nsets = length(workspace.scratch)
    st = FitState{T,V,Y,L,R,S,eltype(workspace.scratch).parameters[3]}(; X = Xm, y = yv, w, f,
        g = zeros(V, n), h = zeros(V, n), z = zeros(V, n),
        idx, roworder = collect(Int32(1):Int32(n)), isleft = zeros(Bool, n),
        scratch = workspace.scratch,
        pool = ScratchPool(nsets:-1:2),   # id 1 is the root task's own and never enters the pool
        nodes = Node{T,V}[], catmasks = UInt64[], iscat, nlevels, features, loss, rule, split_search, lo, hi,
        max_depth = Int(max_depth), min_fit = T(min_fit), min_leaf = T(min_leaf),
        min_sum_hessian = T(min_sum_hessian), max_lin_chain = Int(max_lin_chain),
        truncate = Bool(truncate), nthreads = Int(nthreads), niter = Int(niter),
        unith = unit_hessian(loss) && all(isone, w))
    rows = view(st.roworder, 1:n)
    refresh!(st, rows, 1)
    loss isa Frozen && (st.unith = unit_hessians(st.h))
    grow_subtree(st, rows, 1:n, 0, 0, 1, st.nodes, st.catmasks)
    # Fold the clamped start score into the root node: training accumulates it in
    # st.f before any node is fit, but prediction starts score_row at zero, so the
    # root's own intercepts must carry it. Exact for every loss, clamped or not.
    f0c = clampscore(f0, lo, hi)
    st.nodes[1] = Node{T,V}(st.nodes[1]; lintercept = st.nodes[1].lintercept + f0c,
        rintercept = st.nodes[1].rintercept + f0c)
    # `expected_score` only reads `nodes`/`catmasks` (see shap.jl), so build once with
    # a placeholder base to get the real one, the empty-coalition value (spec line 669-670)
    prelim = LinearTree{T,V,L}(st.nodes, st.catmasks, loss, lo, hi, zero(V), p, truncate)
    base = expected_score(prelim)
    return LinearTree{T,V,L}(st.nodes, st.catmasks, loss, lo, hi, base, p, truncate)
end

"""
Run `f(cols, t)` over `nthreads` contiguous column blocks of `1:p`, threaded
only when a column is long enough (`n` rows) to pay for the tasks. `t` is the
block's index in `1:nthreads`, for callers that keep one buffer per worker.
The row-shaped analogue is `row_blocks`.
"""
function column_blocks(f, p::Integer, n::Integer, nthreads::Integer)
    nthreads = clamp(nthreads, 1, Threads.nthreads())
    if nthreads == 1 || n < PARALLEL_MIN_ROWS || p == 1
        f(1:p, 1)
    else
        chunk = cld(p, nthreads)
        Threads.@threads for t in 1:nthreads
            lo = (t - 1) * chunk + 1
            hi = min(t * chunk, p)
            lo <= hi && f(lo:hi, t)
        end
    end
    return nothing
end

"Unsigned integer as wide as `T`, or `nothing` for an element type with no radix key, which then keeps the comparison sort."
radix_uint(::Type{Float16}) = UInt16
radix_uint(::Type{Float32}) = UInt32
radix_uint(::Type{Float64}) = UInt64
radix_uint(::Type) = nothing

"""
Order-preserving map from a float to an unsigned integer of the same width:
flip the sign bit of a non-negative value, every bit of a negative one. The
result orders as `isless` does over finite values, `-0.0` below `0.0`
included. `fit_tree` rejects a non-finite `X`, so the one case this does not
reproduce -- `isless` puts a negative `NaN` last, the key puts it first -- is
unreachable from a fit.
"""
@inline function radix_key(x::T) where {T<:Union{Float16,Float32,Float64}}
    u = reinterpret(radix_uint(T), x)
    top = one(u) << (8 * sizeof(u) - 1)
    return ifelse(u & top == 0, u | top, ~u)
end

"""
One worker's radix-sort buffers: the two key arrays and the two index arrays
its passes alternate between, the per-byte histogram, and the list of bytes
worth a pass. Built once per column block and reused for every column in it,
which is the whole point -- `sortperm` allocated a permutation and a merge
buffer per column.
"""
struct RadixBuffers{U<:Unsigned}
    keys::Vector{U}
    kbuf::Vector{U}
    ibuf::Vector{Int32}
    jbuf::Vector{Int32}
    hist::Matrix{Int}
    active::Vector{Int}
end
RadixBuffers(::Type{U}, n::Integer) where {U<:Unsigned} = RadixBuffers{U}(
    Vector{U}(undef, n), Vector{U}(undef, n), Vector{Int32}(undef, n), Vector{Int32}(undef, n),
    Matrix{Int}(undef, 256, sizeof(U)), Vector{Int}(undef, sizeof(U)))

"""
Write into `out` the permutation that sorts `x` stably, by a least-significant
byte first radix sort over `radix_key`. Each pass is a counting sort, so equal
keys keep the order they arrived in and, since the first pass starts from
`1:n`, equal values keep ascending row index -- the order `st.idx` promises the
scan. A byte that takes one value on every row cannot reorder anything and is
skipped, which on real data removes most of the exponent's high bytes.
"""
function radix_sortperm!(out::AbstractVector{Int32}, x::AbstractVector, buf::RadixBuffers{U}) where {U}
    n = length(x)
    nbytes = sizeof(U)
    kv = buf.keys; hist = buf.hist
    fill!(hist, 0)
    # one pass builds every byte's histogram, so the skip test below is free
    for i in 1:n
        k = radix_key(x[i])
        kv[i] = k
        for b in 1:nbytes
            hist[Int((k >> (8 * (b - 1))) & 0xff) + 1, b] += 1
        end
    end
    npass = 0
    for b in 1:nbytes
        # a byte that takes one value on every row cannot reorder anything
        constant = false
        for c in 1:256
            if hist[c, b] == n
                constant = true
                break
            end
        end
        if !constant
            npass += 1
            buf.active[npass] = b
        end
    end
    ksrc = kv; kdst = buf.kbuf
    isrc = buf.ibuf; idst = buf.jbuf
    for i in 1:n
        isrc[i] = Int32(i)
    end
    npass == 0 && return copyto!(out, 1, isrc, 1, n)
    for pass in 1:npass
        b = buf.active[pass]
        sh = 8 * (b - 1)
        cursor = 1
        for c in 1:256
            cnt = hist[c, b]
            hist[c, b] = cursor      # the histogram column doubles as the write cursor
            cursor += cnt
        end
        for i in 1:n
            k = ksrc[i]
            c = Int((k >> sh) & 0xff) + 1
            t = hist[c, b]; hist[c, b] = t + 1
            kdst[t] = k
            idst[t] = isrc[i]
        end
        ksrc, kdst = kdst, ksrc
        isrc, idst = idst, isrc
    end
    # `out` is a column of `idx`, so it is not one of the two arrays the passes
    # alternate between: keeping it out of them keeps their element type concrete
    return copyto!(out, 1, isrc, 1, n)
end

"""
Fewest rows a radix sort is worth. Each pass reads and writes a 256-counter
table whatever `n` is, so below the measured crossover the comparison sort
wins; see the presort section of `bench/PROFILE.md` for the sweep.
"""
const RADIX_MIN_ROWS = 640

"""
Stable per-feature sort orders. Features are independent, so this threads over
columns. Both sorts give the same permutation; the row count and the element
type pick which one runs.
"""
function presort!(idx::Matrix{Int32}, X::Matrix{T}, nthreads) where {T<:Real}
    # two call sites, not one on a `Union`: each stays a static dispatch
    size(X, 1) < RADIX_MIN_ROWS && return presort!(idx, X, nthreads, nothing)
    return presort!(idx, X, nthreads, radix_uint(T))
end

"Comparison sort, for a short column or an element type with no radix key."
function presort!(idx::Matrix{Int32}, X::Matrix, nthreads, ::Nothing)
    column_blocks(size(X, 2), size(X, 1), nthreads) do cols, _
        for j in cols
            idx[:, j] .= Int32.(sortperm(view(X, :, j); alg = MergeSort))
        end
    end
    return idx
end

"Radix sort, with one buffer set per column block rather than a permutation and a merge buffer per column."
function presort!(idx::Matrix{Int32}, X::Matrix, nthreads, ::Type{U}) where {U<:Unsigned}
    n = size(X, 1)
    column_blocks(size(X, 2), n, nthreads) do cols, _
        buf = RadixBuffers(U, n)
        for j in cols
            radix_sortperm!(view(idx, :, j), view(X, :, j), buf)
        end
    end
    return idx
end

"""
Stable filter of a full-data presort to the rows in `keep`, renumbered to
`1:length(keep)`. `O(n length(features))`, threaded over selected columns by
`column_blocks` exactly as `presort!` is. Used by `fit_boost`, which presorts once and fits many trees on
row subsets. The caller's `presort` is read only, never permuted. Unselected
columns are untouched and are never read during this tree's growth.
"""
function filter_presort!(idx::Matrix{Int32}, presort::AbstractMatrix{Int32}, keep::Vector{Int}, nthreads,
        features = axes(idx, 2))
    pos = zeros(Int32, size(presort, 1))
    for (k, i) in enumerate(keep)
        pos[i] = Int32(k)
    end
    m0 = size(idx, 1)
    column_blocks(length(features), m0, nthreads) do cols, _
        for posj in cols
            j = features[posj]
            m = 0
            for i in view(presort, :, j)
                k = pos[i]
                k == 0 && continue
                m += 1
                idx[m, j] = k
            end
            # corruption guard, not a user path: from a threaded block this
            # surfaces wrapped in a TaskFailedException
            m == m0 || throw(ArgumentError("presort column $j is not a permutation of the rows"))
        end
    end
    return idx
end

"""
Recompute `g`, `h`, `z` for `rows` at the current score. Floor first, then
weight. Rows are independent, so large row sets are processed in chunks on
separate threads; there is no cross-row reduction, so the result is identical
to the serial pass.
"""
function refresh!(st::FitState, rows, tid)
    ε = node_epsilon(st, rows, tid)
    if st.nthreads == 1 || length(rows) < PARALLEL_MIN_ROWS
        refresh_chunk!(st, rows, ε)
    else
        # plain index vectors, not views of `rows`: a view-of-a-view into `st.y` makes
        # Base's alias check recurse during inference and leaves dynamic dispatch behind
        chunks = [rows[r] for r in Iterators.partition(eachindex(rows), cld(length(rows), st.nthreads))]
        Threads.@threads for ch in chunks
            refresh_chunk!(st, ch, ε)
        end
    end
    return st
end

"""
`ε` for IRLS, computed once over the whole `rows` set so a chunked pass
matches the serial one exactly; smooth losses ignore it. Runs on worker
`tid`'s scratch: the residual goes in `.ws`, `median_abs!`'s abs-value buffer
is `.xs` and its index buffer is `.perm`. All three are free here for the
same reason they are free in `irls_refit` -- `best_split` has returned and
`partition!` has not yet run -- so the node's ε costs no allocation.
"""
function node_epsilon(st::FitState{T}, rows, tid) where {T}
    issmooth(st.loss) && return zero(T)
    sc = st.scratch[tid]
    m = length(rows)
    ensure_len!(sc.ws, m); ensure_len!(sc.xs, m); ensure_len!(sc.perm, m)
    resid = view(sc.ws, 1:m)
    for (k, i) in enumerate(rows)
        resid[k] = st.y[i] - st.f[i]
    end
    return max(irls_epsilon!(sc.xs, sc.perm, resid, view(st.w, rows)),
        sqrt(eps(T)) * max(maximum(abs, view(st.y, rows)), one(T)))
end

function refresh_chunk!(st::FitState, rows, ε)
    yv = view(st.y, rows); fv = view(st.f, rows)
    gv = view(st.g, rows); hv = view(st.h, rows)
    gradhess!(gv, hv, st.loss, yv, fv)
    issmooth(st.loss) || irls_weights!(hv, st.loss, yv, fv; ε)
    for i in rows
        st.g[i] *= st.w[i]
        st.h[i] *= st.w[i]
        st.z[i] = working_response(st.loss, st.y[i], st.f[i], st.g[i], st.h[i])
    end
    return st
end

"""
Gather the node rows, `st.idx[span, j]` in feature-`j` order, into one
worker's scratch. `O(length(span))`. On the unit-hessian path `sc.hs` is not
written: `scan_gathered` hands the scan a `UnitHessians` instead, and the
branch is hoisted out of the row loop so neither loop tests it per row.
"""
function gather!(st::FitState, sc::Scratch, span::UnitRange{Int}, j)
    len = length(span)
    ensure_len!(sc.xs, len); ensure_len!(sc.zs, len); ensure_len!(sc.ws, len)
    st.unith || ensure_len!(sc.hs, len)
    m = 0
    if st.unith
        for k in span
            i = st.idx[k, j]
            m += 1
            sc.xs[m] = st.X[i, j]; sc.zs[m] = st.z[i]; sc.ws[m] = st.w[i]
        end
    else
        for k in span
            i = st.idx[k, j]
            m += 1
            sc.xs[m] = st.X[i, j]; sc.zs[m] = st.z[i]; sc.hs[m] = st.h[i]; sc.ws[m] = st.w[i]
        end
    end
    return m
end

"""
Scan the first `m` rows of `sc` under `rule`. Two calls, not one on a
`Union` of `hs` types: on the unit-hessian path the scan gets `UnitHessians`,
which drops the multiply from `addrow` and `subrow`, and each branch stays a
call with a concrete argument type. Both give the same sums to the bit, since
the multiply dropped is by exactly one.
"""
@inline function scan_gathered(st::FitState{T,V}, sc::Scratch{T,V}, m, rule, dmin) where {T,V}
    xs = view(sc.xs, 1:m); zs = view(sc.zs, 1:m); ws = view(sc.ws, 1:m)
    search = rule isa PconOnly ? ExactSearch() : st.split_search
    buffer = rule isa PconOnly ? nothing : sc.search
    st.unith && return scan_feature(xs, zs, UnitHessians{V}(m), ws, rule, st.min_leaf, dmin, search, buffer)
    return scan_feature(xs, zs, view(sc.hs, 1:m), ws, rule, st.min_leaf, dmin, search, buffer)
end

"""
Stable partition of `idx[span]` into rows with `isleft[i]` true, then the
rest, each side in its original order. `perm` is a buffer of at least
`length(span)`. Returns the left count. Positions outside `span` are untouched.
"""
function partition_column!(idx::AbstractVector{Int32}, span::UnitRange{Int}, isleft::Vector{Bool}, perm::Vector{Int32})
    nleft = 0
    for k in span
        nleft += isleft[idx[k]]
    end
    return partition_column!(idx, span, isleft, perm, nleft)
end

# Every feature column has the same row set, so partition! counts it once.
function partition_column!(idx::AbstractVector{Int32}, span::UnitRange{Int}, isleft::Vector{Bool}, perm::Vector{Int32}, nleft::Int)
    a = 0; b = nleft
    for k in span
        i = idx[k]
        if isleft[i]
            perm[a += 1] = i
        else
            perm[b += 1] = i
        end
    end
    copyto!(idx, first(span), perm, 1, length(span))
    return nleft
end

"""
Partition selected columns of `idx`, and `roworder` with them, over `span` so that
the rows marked in `isleft` come first, in place and stable. Each child's rows
then stay sorted by every selected feature, and `roworder[span]` keeps them in the order
the parent's sums ran over. Columns are independent: with at least
`PARALLEL_MIN_ROWS` rows they split across the scratch ids in `ids`, each task
using its own `scratch[id].perm` buffer. The caller owns the marks -- it sets
them for this node's rows and clears them afterwards, so concurrent sibling
subtrees, which own disjoint rows, never see each other's. Returns the left
count.
"""
function partition!(idx::Matrix{Int32}, roworder::Vector{Int32}, span::UnitRange{Int}, isleft::Vector{Bool},
        scratch, ids::AbstractVector{Int}, features = axes(idx, 2))
    # every column returns the same left count, so take it here rather than from
    # whichever task happens to finish last
    nleft = 0
    for k in span
        nleft += isleft[idx[k, first(features)]]
    end
    let nleft = nleft
        column_blocks(length(features), length(span), length(ids)) do cols, t
            perm = ensure_len!(scratch[ids[t]].perm, length(span))
            for k in cols
                j = features[k]
                partition_column!(view(idx, :, j), span, isleft, perm, nleft)
            end
        end
    end
    # `roworder` is not a feature column, so it goes outside the threaded block,
    # on the calling task's own buffer
    partition_column!(roworder, span, isleft, ensure_len!(scratch[first(ids)].perm, length(span)))
    return nleft
end

partition_node!(st, span, ids) =
    partition!(st.idx, st.roworder, span, st.isleft, st.scratch, ids, st.features)

"Wrap a rule so only `con` and `pcon` are offered, for categorical scans."
struct PconOnly{R<:SelectionRule} <: SelectionRule
    inner::R
end
allowed(r::PconOnly, k::ModelKind) = (k == CON || k == PCON) && allowed(r.inner, k)
score_logn(r::PconOnly, n) = score_logn(r.inner, n)
devkey(r::PconOnly, surrogate, dmin) = devkey(r.inner, surrogate, dmin)
selection_score(r::PconOnly, k, s, n, dmin, ncoord::Integer = 1, logn = score_logn(r, n)) =
    allowed(r, k) ? selection_score(r.inner, k, s, n, dmin, ncoord, logn) : Inf
ridge(r::PconOnly) = ridge(r.inner)
@inline fit_con(s::MomentSums, r::PconOnly) = fit_con(s, r.inner)
@inline fit_lin(s::MomentSums, r::PconOnly; tol = SINGULAR_TOL) = fit_lin(s, r.inner; tol)

"""
Order the node's levels by weighted mean working response, scan a `pcon`
split over the ranks, and return the candidate plus the left level codes.
Runs on the worker's own scratch `sc`, so it is safe inside the threaded
feature search: no shared mutable state.

For a vector `V` (`Softmax`), one working response per coordinate gives one
candidate order per coordinate `k`; each is scanned in turn and the
lowest-scoring candidate wins. A scalar `V` has exactly one coordinate, so
this reduces to the original single-order scan.
"""
function scan_categorical(st::FitState{T,V}, sc::Scratch{T,V}, rows, j, dmin) where {T,V}
    L = st.nlevels[j]
    nr = length(rows)
    ensure_len!(sc.xs, nr); ensure_len!(sc.zs, nr); ensure_len!(sc.hs, nr)
    ensure_len!(sc.ws, nr); ensure_len!(sc.perm, nr)
    lv = ensure_levels!(sc.levels, L)
    sz = lv.sz; sw = lv.sw; counts = lv.counts
    # `counts` is offset by one so it doubles as the counting sort's histogram
    for c in 1:L
        sz[c] = zero(V); sw[c] = zero(V); counts[c + 1] = 0
    end
    for i in rows
        c = Int(st.X[i, j])
        sz[c] += st.h[i] .* st.z[i]
        sw[c] += st.h[i]
        counts[c + 1] += 1
    end
    present = lv.present
    np = 0
    for c in 1:L
        if counts[c + 1] > 0
            np += 1
            present[np] = c
        end
    end
    np < 2 && return nocandidate(T, V), NO_LEVELS
    # Bucket rows by level in one O(L + |rows|) counting-sort pass, then walk
    # the buckets in rank order to fill the scratch. No O(L · |rows|) rescans.
    # The bucketing itself doesn't depend on level order, so it is built once
    # and reused for every coordinate's ordering below. `sc.perm` is free until
    # `partition!` runs, well after the scan.
    cumoffset = lv.cumoffset; cursor = lv.cursor
    cumoffset[1] = 0
    for c in 1:L
        cumoffset[c + 1] = cumoffset[c] + counts[c + 1]
        cursor[c] = cumoffset[c]
    end
    bucketed = sc.perm
    for i in rows
        c = Int(st.X[i, j])
        cursor[c] += 1
        bucketed[cursor[c]] = i
    end
    rule = PconOnly(st.rule)      # a categorical column never carries a linear piece, whatever the rule allows
    Km = length(zero(V))
    rank = lv.rank
    order = view(lv.order, 1:np)
    best = nocandidate(T, V); bestleft = NO_LEVELS
    for k in 1:Km
        copyto!(order, 1, present, 1, np)
        # stable, so levels whose mean working response ties keep ascending code
        sort!(order; by = c -> sz[c][k] / sw[c][k])
        # every present level is ranked here, and no other level is ever read back
        for (r, lc) in enumerate(order)
            rank[lc] = r
        end
        m = 0
        for lc in order
            for b in (cumoffset[lc] + 1):cumoffset[lc + 1]
                i = bucketed[b]
                m += 1
                sc.xs[m] = T(rank[lc]); sc.zs[m] = st.z[i]; sc.hs[m] = st.h[i]; sc.ws[m] = st.w[i]
            end
        end
        cand = scan_gathered(st, sc, m, rule, dmin)
        if cand.score < best.score
            best = cand
            # the winner's codes outlive the scratch set, so this one stays a fresh vector
            bestleft = cand.kind == PCON ? [lc for lc in order if rank[lc] <= cand.threshold] : NO_LEVELS
        end
    end
    return best, bestleft
end

"""
Append a packed left-level mask to the mask pool `masks` the growing subtree
writes into, and return its `(catstart, catwords)` in that pool.
"""
function push_mask!(masks::Vector{UInt64}, leftcodes, L)
    words = (L + 63) >> 6
    start = length(masks) + 1
    append!(masks, zeros(UInt64, words))
    for c in leftcodes
        masks[start + ((c - 1) >> 6)] |= UInt64(1) << ((c - 1) & 63)
    end
    return Int32(start), Int32(words)
end

"Serial search over `features` using scratch set `tid`. Returns the best candidate, its feature, and (for a categorical winner) its left level codes."
function best_split_serial(st::FitState{T,V}, rows, span::UnitRange{Int}, dmin, features, tid) where {T,V}
    sc = st.scratch[tid]
    best = nocandidate(T, V); bestj = 0; bestleft = NO_LEVELS
    for j in features
        if st.iscat[j]
            c, leftcodes = scan_categorical(st, sc, rows, j, dmin)
        else
            m = gather!(st, sc, span, j)
            c = scan_gathered(st, sc, m, st.rule, dmin)
            leftcodes = NO_LEVELS
        end
        if c.score < best.score
            best = c; bestj = j; bestleft = leftcodes
        end
    end
    return best, bestj, bestleft
end

"""
Search all features on the scratch sets named by `ids`, which the caller owns
for the duration of the call. Large nodes split the feature range across
`length(ids)` tasks, each with its own scratch. The reduction takes the lowest
score and, on ties, the lowest feature index, so the result equals the serial
search regardless of how many ids turned up or of task completion order.
"""
function best_split(st::FitState{T,V}, rows, span::UnitRange{Int}, dmin, ids::AbstractVector{Int}) where {T,V}
    feats = st.features
    p = length(feats)
    if length(rows) < PARALLEL_MIN_ROWS || length(ids) == 1 || p == 1
        return best_split_serial(st, rows, span, dmin, feats, first(ids))
    end
    chunks = collect(Iterators.partition(feats, cld(p, length(ids))))
    # `tasks` is iterated in creation order below, over ascending chunks, so the
    # reduction sees the chunks in feature order however the tasks finish. A
    # comprehension over `zip` would infer a 0-dimensional `similar` here, since
    # `ids` is abstractly typed
    tasks = Vector{Task}(undef, length(chunks))
    for k in eachindex(chunks)
        # bound outside the task: `@spawn` closes over its arguments rather than
        # copying them, and `ids` is the caller's buffer, returned once we are done
        ch = chunks[k]; id = ids[k]
        tasks[k] = Threads.@spawn best_split_serial(st, rows, span, dmin, ch, id)
    end
    # join every task before returning, including on the error path: the caller
    # gives the borrowed ids back the moment this returns, and an id must not
    # reach another task while the one holding that scratch set still runs
    failed = nothing
    for t in tasks
        try
            wait(t)
        catch e
            failed === nothing && (failed = e)
        end
    end
    # `throw`, not `rethrow`: the catch block above has exited by here, and
    # `rethrow` outside one raises an `ErrorException` of its own instead. Nothing
    # is lost -- `wait` already wrapped the chunk's exception in a
    # `TaskFailedException`, whose backtrace is the chunk's own.
    failed === nothing || throw(failed)
    best = nocandidate(T, V); bestj = 0; bestleft = NO_LEVELS
    for t in tasks
        # `fetch` infers `Any`; without this the winning candidate stays boxed and
        # `grow_subtree`'s whole body dispatches on it once per node
        c, j, lc = fetch(t)::Tuple{Candidate{T,V},Int,Vector{Int}}
        if c.score < best.score || (c.score == best.score && j != 0 && (bestj == 0 || j < bestj))
            best = c; bestj = j; bestleft = lc
        end
    end
    return best, bestj, bestleft
end

"`nw` is the node's total weight, which `grow_subtree` already holds; recomputing it here would be a second O(|rows|) pass."
function dmin_for(st::FitState{T,V}, rows, nw) where {T,V}
    s = zero(T)
    for i in rows
        s += sum(st.h[i] .* st.z[i] .^ 2)   # sum over coordinates for vector V; a no-op for scalar V
    end
    return eps(T) * max(s, nw)
end

leafnode(st::FitState{T,V}, rows, b) where {T,V} =
    Node{T,V}(; lintercept = b, cover = sum(view(st.w, rows)))

"""
Add `idxoff` to every non-zero child index and `maskoff` to every non-zero
`catstart` in `nodes`. Used once per spawned sibling subtree, to relocate the
local node vector and mask words it grew concurrently into the parent's.
"""
function shift_subtree(nodes::Vector{Node{T,V}}, idxoff::Int32, maskoff::Int32) where {T,V}
    return [Node{T,V}(n; left = n.left == 0 ? Int32(0) : n.left + idxoff,
                        right = n.right == 0 ? Int32(0) : n.right + idxoff,
                        catstart = n.catwords == 0 ? Int32(0) : n.catstart + maskoff) for n in nodes]
end

"""
Grow the subtree for `rows`, appending its nodes to `nodes` in preorder and
its level masks to `masks`. The subtree root lands at `length(nodes) + 1` and
child indices and `catstart` values are absolute in those two vectors, so the
caller has nothing to splice: a node is written once and moved never.

`span` is the range of `st.idx` this node owns: `st.idx[span, j]` holds
exactly `rows`, sorted by feature `j`, for every `j`. A split partitions the
span in place, so children own disjoint sub-ranges and no other node reads or
writes them. `linchain` counts consecutive `lin` fits in this node position.

`tid` is the one `st.scratch` set this task owns. It leaves `st.pool` through
`trytake!`, for a spawned sibling, or `borrow!`, for one threaded call on this
node, and each of those two returns sits in a `finally` below.

Only the calling task appends to `nodes` and `masks`. A spawned sibling grows
into its own pair and is shifted and appended once when it returns, so that
subtree is the one and only case where a node is copied.
"""
function grow_subtree(st::FitState{T,V}, rows::RowView, span::UnitRange{Int}, depth::Int, linchain::Int,
        tid::Int, nodes::Vector{Node{T,V}}, masks::Vector{UInt64}) where {T,V}
    mine = length(nodes) + 1
    nw = sum(view(st.w, rows))
    sumh = sum(sum(h) for h in view(st.h, rows))   # sum over coordinates too, for vector V
    if nw < st.min_fit || depth >= st.max_depth || sumh < st.min_sum_hessian || linchain >= st.max_lin_chain
        b = fit_con(node_sums(st, rows), st.rule)[1]
        me = leafnode(st, rows, b)
        me = refit_node(st, me, rows, tid)
        update_score!(st, rows, me)
        push!(nodes, me)
        return nothing
    end
    dmin = dmin_for(st, rows, nw)
    # borrow only when the callee would actually thread: under the row gate both
    # `best_split` and `partition!` run on `tid` alone, and the borrow would be a
    # lock round trip for nothing at every one of the tree's small nodes
    nborrow = length(rows) < PARALLEL_MIN_ROWS ? 0 : min(length(st.features), st.nthreads) - 1
    sc = st.scratch[tid]
    ids = worker_ids!(sc, st.pool, tid, nborrow)
    best, bestj, leftcodes = try
        best_split(st, rows, span, dmin, ids)
    finally
        giveback!(st.pool, ids)
    end
    con_intercept, con_surrogate = fit_con(node_sums(st, rows), st.rule)
    gain = T(sum(con_surrogate) - sum(best.surrogate))
    if best.kind == CON || bestj == 0
        me = leafnode(st, rows, best.kind == CON ? best.lintercept : con_intercept)
        me = refit_node(st, me, rows, tid)
        update_score!(st, rows, me)
        push!(nodes, me)
        return nothing
    end
    iscat = st.iscat[bestj]
    if iscat
        xmin = xmax = xmean = zero(T)
    else
        xj = view(st.X, rows, bestj)
        xmin, xmax = extrema(xj)
        xmean = sum(st.w[i] * st.X[i, bestj] for i in rows) / nw
    end
    if best.kind == LIN
        # A LIN node stores `threshold = NaN`, so `x <= n.threshold` is false for
        # every row and the routing falls to the right branch. That is harmless
        # only because both branches are the same fit: `rcoef == lcoef`,
        # `rintercept == lintercept` and, below, `right == left`. `score_row`
        # (predict.jl) and `coeftable`'s walk (importance.jl) both rely on it.
        me = Node{T,V}(; feature = bestj, threshold = T(NaN), lcoef = best.lcoef, lintercept = best.lintercept,
            rcoef = best.lcoef, rintercept = best.lintercept, xmin, xmax, cover = nw, xmean, gain, model = LIN)
        me = refit_node(st, me, rows, tid)
        update_score!(st, rows, me); refresh!(st, rows, tid)
        push!(nodes, Node{T,V}(me; left = Int32(mine + 1), right = Int32(mine + 1)))
        grow_subtree(st, rows, span, depth, linchain + 1, tid, nodes, masks)
        return nothing
    end
    catstart = Int32(0); catwords = Int32(0); threshold = best.threshold
    if iscat
        catstart, catwords = push_mask!(masks, leftcodes, st.nlevels[bestj])
        threshold = T(NaN)
    end
    me = Node{T,V}(; feature = bestj, threshold, lcoef = best.lcoef, lintercept = best.lintercept,
        rcoef = best.rcoef, rintercept = best.rintercept, xmin, xmax, cover = nw, xmean, gain, model = best.kind,
        catstart, catwords)
    me = refit_node(st, me, rows, tid, masks)
    update_score!(st, rows, me, masks)
    refresh!(st, rows, tid)
    # one `goes_left` pass, straight into the markers `partition!` reads
    for i in rows
        st.isleft[i] = goes_left(st, me, i, masks)
    end
    ids = worker_ids!(sc, st.pool, tid, nborrow)
    nl = try
        partition_node!(st, span, ids)
    finally
        giveback!(st.pool, ids)
    end
    lspan = first(span):(first(span) + nl - 1); rspan = (first(span) + nl):last(span)
    for k in lspan
        st.isleft[st.roworder[k]] = false   # after the partition these are exactly the rows marked above
    end
    # `roworder[span]` held this node's rows in the order its own sums ran over,
    # and the partition was stable, so the two halves are the child row sets in
    # that same order -- what the two fresh `Vector{Int32}`s used to hold. Each
    # child owns its half alone: the sibling task writes only the other one, and
    # a child reads `rows` only before its own `partition!` reorders it.
    leftrows = view(st.roworder, lspan); rightrows = view(st.roworder, rspan)
    push!(nodes, me)   # placeholder: the child indices are only known once both children have grown
    # a spare set means a spare worker: run the right subtree as its own task and
    # grow the left one inline. An empty pool is not a reason to wait, so both
    # children then grow inline and the whole recursion stays deadlock-free.
    rtid = length(rightrows) >= SUBTREE_MIN_ROWS ? trytake!(st.pool) : 0
    if rtid == 0
        grow_subtree(st, leftrows, lspan, depth + 1, 0, tid, nodes, masks)
        rightidx = length(nodes) + 1
        grow_subtree(st, rightrows, rspan, depth + 1, 0, tid, nodes, masks)
    else
        rnodes = Node{T,V}[]; rmasks = UInt64[]
        task = Threads.@spawn grow_subtree(st, rightrows, rspan, depth + 1, 0, rtid, rnodes, rmasks)
        try
            grow_subtree(st, leftrows, lspan, depth + 1, 0, tid, nodes, masks)
            fetch(task)   # `fetch`, not `wait`: a failed sibling must raise here, not vanish
        finally
            # the sibling owns `rtid` until it stops running, so the id goes back only
            # after it does -- including when the left side is the one that threw
            try
                wait(task)
            catch
                # a sibling failure is already propagating, from `fetch` or from the left side
            end
            give!(st.pool, rtid)
        end
        # the sibling grew local indices from 1, so shift by what now precedes it
        rightidx = length(nodes) + 1
        append!(nodes, shift_subtree(rnodes, Int32(length(nodes)), Int32(length(masks))))
        append!(masks, rmasks)
    end
    nodes[mine] = Node{T,V}(me; left = Int32(mine + 1), right = Int32(rightidx))
    return nothing
end

"""
True when row `i` is routed left by node `n`. Categorical nodes route by
mask; others by threshold. `masks` must be the mask pool `n.catstart`
indexes into: during growth that is the pool the subtree is appending to
(`st.catmasks` on the main chain, a local vector inside a spawned sibling);
after fitting it is `tree.catmasks`. Non-categorical nodes never touch `masks`,
so the empty default is safe wherever the node cannot be categorical.
"""
function goes_left(st::FitState, n::Node, i, masks::Vector{UInt64} = UInt64[])
    iscategorical(n) && return category_is_left(masks, n, Int(st.X[i, n.feature]))
    return n.model == LIN || st.X[i, n.feature] <= n.threshold
end

"""
IRLS refit of one node's own coefficients on its own rows. `niter` passes:
each recomputes the residual against the node's current fit, the scale-aware
`ε`, and the L1-majorizer weight, then re-solves the node's model kind.
Serial: this is one node's `MomentSums` reduction, not worth threading.
Takes and returns a `Node` value so it works on a node still local to a
growing subtree, before it has a place in `st.nodes`. `masks` is the pool
`n.catstart` indexes into (see `goes_left`); irrelevant, so omitted, for a
node that cannot be categorical. `tid` is this task's own worker index:
`st.scratch[tid].zs` (the residual buffer)
and `.xs` (`median_abs`'s abs-value buffer) and `.perm` (its sort
permutation) are all free at this point -- `best_split` has already returned
its winning candidate, and `partition!` has not yet run for this node -- so
reusing them here avoids a fresh allocation on every call.
"""
function irls_refit(st::FitState{T,V}, n::Node{T,V}, rows, tid, niter, masks::Vector{UInt64} = UInt64[]) where {T,V}
    j = n.feature
    m = length(rows)
    sc = st.scratch[tid]
    ensure_len!(sc.zs, m); ensure_len!(sc.xs, m); ensure_len!(sc.perm, m)
    resid = view(sc.zs, 1:m)
    permbuf = sc.perm
    yscale = max(maximum(abs, view(st.y, rows)), one(T))
    for _ in 1:niter
        for (k, i) in enumerate(rows)
            pred = if n.model == CON
                n.lintercept
            else
                x = st.X[i, j]
                goleft = goes_left(st, n, i, masks)
                goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
            end
            resid[k] = st.z[i] - pred
        end
        ε = max(irls_epsilon!(sc.xs, permbuf, resid, view(st.w, rows)), sqrt(eps(T)) * yscale)
        left = zero(MomentSums{V}); right = zero(MomentSums{V})
        for (k, i) in enumerate(rows)
            r = resid[k]
            # floor the unweighted pseudo-hessian at HMIN before scaling by st.w[i],
            # the same order irls_weights! applies it in, so a zero-weight row can't
            # be the only thing keeping a non-smooth refit's Hessian away from zero
            hi = st.w[i] * max(l1weight(st.loss, r) / max(abs(r), ε), oftype(r, HMIN))
            if n.model == CON
                left = addrow(left, zero(T), st.z[i], hi)
            else
                x = st.X[i, j]
                goleft = goes_left(st, n, i, masks)
                if n.model != LIN && !goleft
                    right = addrow(right, x, st.z[i], hi)
                else
                    left = addrow(left, x, st.z[i], hi)
                end
            end
        end
        if n.model == CON
            b = fit_con(left, st.rule)[1]
            n = Node{T,V}(n; lintercept = b, rintercept = b)
        elseif n.model == LIN
            r = fit_lin(left, st.rule); r === nothing && break
            a, b, _ = r
            n = Node{T,V}(n; lcoef = a, lintercept = b, rcoef = a, rintercept = b)
        elseif n.model == PCON
            bl = fit_con(left, st.rule)[1]; br = fit_con(right, st.rule)[1]
            n = Node{T,V}(n; lintercept = bl, rintercept = br)
        elseif n.model == PLIN
            rl = fit_lin(left, st.rule); rr = fit_lin(right, st.rule)
            (rl === nothing || rr === nothing) && break
            n = Node{T,V}(n; lcoef = rl[1], lintercept = rl[2], rcoef = rr[1], rintercept = rr[2])
        else # BLIN
            r = fit_blin(left, right, n.threshold); r === nothing && break
            n = Node{T,V}(n; lcoef = r[1], lintercept = r[2], rcoef = r[3], rintercept = r[4])
        end
    end
    return n
end

function node_sums(st::FitState{T,V}, rows) where {T,V}
    s = zero(MomentSums{V})
    for i in rows
        s = addrow(s, zero(T), st.z[i], st.h[i])
    end
    return s
end

@inline function node_increment(st::FitState, n::Node, i, masks)
    isleaf(n) && return n.lintercept
    goleft = goes_left(st, n, i, masks)
    iscategorical(n) && return goleft ? n.lintercept : n.rintercept
    x = st.X[i, n.feature]
    x = st.truncate ? min(max(x, n.xmin), n.xmax) : x
    return goleft ? n.lcoef * x + n.lintercept : n.rcoef * x + n.rintercept
end

# Small logistic Hessians can turn a useful Newton direction into a disastrous
# full step. Keep a descending full step; otherwise halve it until the actual
# weighted loss descends, including the same truncation used during prediction.
function logistic_backtrack(st::FitState{T,V}, n::Node{T,V}, rows, tid, masks) where {T,V}
    # Search has finished and partition has not started, so this worker's
    # response buffer is free, as it is during the non-smooth IRLS refit.
    inc = st.scratch[tid].zs
    ensure_len!(inc, length(rows))
    baseline = zero(T)
    for (k, i) in enumerate(rows)
        inc[k] = node_increment(st, n, i, masks)
        baseline += st.w[i] * pointloss(st.loss, st.y[i], st.f[i])
    end
    scale = one(T)
    while scale >= eps(T)
        candidate = zero(T)
        for (k, i) in enumerate(rows)
            f = st.f[i] + scale * inc[k]
            f = st.truncate ? clampscore(f, st.lo, st.hi) : f
            candidate += st.w[i] * pointloss(st.loss, st.y[i], f)
        end
        if isfinite(candidate) && candidate <= baseline
            scale == one(T) && return n
            return Node{T,V}(n; lcoef = scale * n.lcoef, lintercept = scale * n.lintercept,
                rcoef = scale * n.rcoef, rintercept = scale * n.rintercept)
        end
        scale /= 2
    end
    return Node{T,V}(n; lcoef = zero(V), lintercept = zero(V),
        rcoef = zero(V), rintercept = zero(V))
end

"Add node `n`'s piece to the score of `rows`, then clamp. `masks` is the pool `n.catstart` indexes into (see `goes_left`)."
function update_score!(st::FitState, rows, n::Node, masks::Vector{UInt64} = UInt64[])
    for i in rows
        s = st.f[i] + node_increment(st, n, i, masks)
        st.f[i] = st.truncate ? clampscore(s, st.lo, st.hi) : s
    end
    return st
end
