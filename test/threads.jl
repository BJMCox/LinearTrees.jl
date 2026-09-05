using StableRNGs

@testset "subtree-parallel fit equals serial fit" begin
    rng = StableRNG(34)
    X = rand(rng, 6_000, 5); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 6_000)
    t1 = fit_tree(X, y; nthreads = 1)
    tn = fit_tree(X, y)
    @test t1.nodes == tn.nodes
    # categorical and softmax paths
    lvl = Float64.(rand(rng, 1:6, 6_000))
    Xc = hcat(lvl, X)
    yc = [X[i, 1] > 0.5 ? 1 : lvl[i] <= 3 ? 2 : 3 for i in 1:6_000]
    tc1 = fit_tree(Xc, yc, Softmax(3); categorical = [1], nthreads = 1)
    tcn = fit_tree(Xc, yc, Softmax(3); categorical = [1])
    @test tc1.nodes == tcn.nodes
    @test tc1.catmasks == tcn.catmasks
end

# Every thread count, three times each: the number of ids `borrow!` finds free
# varies run to run at fixed `nthreads`, so one run per count would not see a
# reduction that had become borrow-count dependent.
@testset "odd scratch splits and nested row threading" begin
    rng = StableRNG(35)
    X = rand(rng, 20_000, 4); y = sin.(3 .* X[:, 1]) .+ X[:, 2] .* X[:, 3] .+ 0.1 .* randn(rng, 20_000)
    lvl = Float64.(rand(rng, 1:7, 20_000))
    Xc = hcat(lvl, X)
    t1 = fit_tree(Xc, y; categorical = [1], nthreads = 1, max_depth = 5)
    for k in 1:Threads.nthreads(), _ in 1:3
        tk = fit_tree(Xc, y; categorical = [1], nthreads = k, max_depth = 5)
        @test tk.nodes == t1.nodes
        @test tk.catmasks == t1.catmasks
    end
end

# Fails if `trytake!` ever hands one id to two callers, if `give!` loses one, or
# if `borrow!` returns more than it was asked for.
@testset "ScratchPool hands out every free id at most once" begin
    pool = LinearTrees.ScratchPool(4:-1:2)
    out = [LinearTrees.trytake!(pool) for _ in 1:3]
    @test sort(out) == [2, 3, 4]
    @test LinearTrees.trytake!(pool) == 0
    @test isempty(pool.free)
    LinearTrees.give!(pool, out[2])
    @test LinearTrees.trytake!(pool) == out[2]
    @test LinearTrees.trytake!(pool) == 0
    @test LinearTrees.ScratchPool(1:-1:2).free == Int[]   # the nthreads == 1 pool

    pool = LinearTrees.ScratchPool(5:-1:2)
    ids = LinearTrees.borrow!(pool, [1], 2)
    @test length(ids) == 3 && ids[1] == 1 && allunique(ids)
    @test length(pool.free) == 2
    ids = LinearTrees.borrow!(pool, [1], 9)   # more than the pool holds
    @test length(ids) == 3 && isempty(pool.free)
    LinearTrees.giveback!(pool, ids)
    @test ids == [1] && length(pool.free) == 2
    @test LinearTrees.borrow!(pool, [1], 0) == [1] && length(pool.free) == 2
end

# The pool race: ties are everywhere (two-digit features), a nine-level
# categorical exercises `scan_categorical`'s own buffers, and `MAD` adds the
# IRLS refit as a second scratch user. Fails if the cross-feature reduction
# stops resolving ties by lowest feature index, if two tasks reach one scratch
# set, or if `node_epsilon` becomes chunk-dependent.
@testset "pool stress: every thread count reproduces the serial tree" begin
    rng = StableRNG(36)
    n = 40_000
    Xn = round.(rand(rng, n, 6); digits = 2)
    lvl = Float64.(rand(rng, 1:9, n))
    X = hcat(Xn, lvl)
    yr = sum(floor.(3 .* Xn[:, j]) for j in 1:3) .+ Xn[:, 4] .* Xn[:, 5] .+ 0.3 .* lvl .+ 0.1 .* randn(rng, n)
    w = rand(rng, n) .+ 0.5
    ycls = Float64.([Xn[i, 1] > 0.5 ? 1 : (lvl[i] <= 4 ? 2 : 3) for i in 1:n])
    designs = (("MSE", yr, MSE(), (;)),
        ("weighted MSE", yr, MSE(), (; weights = w)),
        ("MAD", yr, MAD(), (; niter = 5)),
        ("Softmax(3)", ycls, Softmax(3), (;)))
    for (_, y, loss, kw) in designs
        ref = fit_tree(X, y, loss; categorical = [7], max_depth = 8, nthreads = 1, kw...)
        for k in 1:Threads.nthreads(), _ in 1:3
            t = fit_tree(X, y, loss; categorical = [7], max_depth = 8, nthreads = k, kw...)
            @test t.nodes == ref.nodes
            @test t.catmasks == ref.catmasks
        end
    end
end

"""
A loss that throws from inside `refresh!` once a node is smaller than
`minrows`, so the throw lands while sibling subtrees are still growing. Only
`gradhess!` is needed: `issmooth` defaults to true, so nothing else in growth
touches the loss.
"""
struct ThrowSmall <: LinearTrees.Loss
    minrows::Int
end

function LinearTrees.gradhess!(g::AbstractVector, h::AbstractVector, l::ThrowSmall,
        y::AbstractVector, f::AbstractVector)
    length(y) < l.minrows && error("ThrowSmall fired at $(length(y)) rows")
    for i in eachindex(g, h, y, f)
        g[i] = f[i] - y[i]
        h[i] = one(eltype(h))
    end
    return g
end

"""
A rule that throws from `allowed` on its `nth` call and every call after it.
`allowed` is reached only from `scan_feature`, so the throw lands inside
`best_split`'s spawned scan, at the root, while the node holds every id it
borrowed.
"""
struct ThrowNth <: LinearTrees.SelectionRule
    inner::BIC
    nth::Int
    calls::Threads.Atomic{Int}
end
ThrowNth(nth::Int) = ThrowNth(BIC(), nth, Threads.Atomic{Int}(0))

function LinearTrees.allowed(r::ThrowNth, k::LinearTrees.ModelKind)
    Threads.atomic_add!(r.calls, 1) + 1 >= r.nth && error("ThrowNth fired")
    return LinearTrees.allowed(r.inner, k)
end
LinearTrees.score_logn(r::ThrowNth, n) = LinearTrees.score_logn(r.inner, n)
LinearTrees.devkey(r::ThrowNth, surrogate, dmin) = LinearTrees.devkey(r.inner, surrogate, dmin)
LinearTrees.selection_score(r::ThrowNth, k, s, n, dmin, ncoord::Integer = 1,
    logn = LinearTrees.score_logn(r, n)) = LinearTrees.selection_score(r.inner, k, s, n, dmin, ncoord, logn)

"""
A rule that throws from the `LIN` score of the one feature that fits `y`
exactly, and blocks every other feature's `LIN` score on an `Event`. That
feature is column 1, so the chunk that throws is always `tasks[1]` and the
chunks still running are always the later ones, whatever order the scheduler
starts them in.
"""
struct BlockOthers <: LinearTrees.SelectionRule
    inner::BIC
    thrown::Threads.Event
    release::Threads.Event
end
BlockOthers() = BlockOthers(BIC(), Threads.Event(), Threads.Event())

function LinearTrees.selection_score(r::BlockOthers, kind::LinearTrees.ModelKind, surrogate, n, dmin,
        ncoord::Integer = 1, logn = LinearTrees.score_logn(r, n))
    if kind == LinearTrees.LIN
        # only column 1 fits `z` exactly, so only its residual sum reaches zero
        if surrogate < 1e-6
            notify(r.thrown)
            error("BlockOthers fired on chunk 1")
        end
        wait(r.release)
    end
    return LinearTrees.selection_score(r.inner, kind, surrogate, n, dmin, ncoord, logn)
end
LinearTrees.allowed(r::BlockOthers, k::LinearTrees.ModelKind) = LinearTrees.allowed(r.inner, k)
LinearTrees.score_logn(r::BlockOthers, n) = LinearTrees.score_logn(r.inner, n)
LinearTrees.devkey(r::BlockOthers, surrogate, dmin) = LinearTrees.devkey(r.inner, surrogate, dmin)

"20_000 x 4, a step target so BIC keeps splitting down to nodes of a few hundred rows."
function stepdata()
    rng = StableRNG(37)
    X = rand(rng, 20_000, 4)
    return X, sum(floor.(4 .* X[:, j]) for j in 1:3) .+ 0.1 .* randn(rng, 20_000)
end

"20_000 x 4 where column 1 is `y` itself, so its linear fit is the only exact one."
function linedata()
    rng = StableRNG(38)
    X = rand(rng, 20_000, 4)
    return X, copy(X[:, 1])
end

"""
A `FitState` over `data` with `nt` workers and `SCRATCH_PER_THREAD * nt` scratch
sets, ready for `grow_subtree`. Returns it with its row vector and the set
count. `grow_subtree` is the only way to reach the pool guards, and no public
entry point that reaches them also hands back `st`.
"""
function poolstate(loss, rule, nt; data = stepdata())
    X, y = data
    n, p = size(X)
    nsets = nt == 1 ? 1 : LinearTrees.SCRATCH_PER_THREAD * nt
    idx = Matrix{Int32}(undef, n, p)
    LinearTrees.presort!(idx, X, nt)
    st = LinearTrees.FitState{Float64,Float64,typeof(loss),typeof(rule)}(; X, y, w = ones(n),
        f = zeros(n), g = zeros(n), h = zeros(n), z = zeros(n), idx, isleft = zeros(Bool, n),
        scratch = [LinearTrees.Scratch{Float64,Float64}() for _ in 1:nsets],
        pool = LinearTrees.ScratchPool(nsets:-1:2),
        nodes = LinearTrees.Node{Float64,Float64}[], catmasks = UInt64[],
        iscat = zeros(Bool, p), nlevels = zeros(Int, p), loss, rule, lo = -Inf, hi = Inf,
        max_depth = 12, min_fit = 10.0, min_leaf = 5.0, min_sum_hessian = 1.0, max_lin_chain = 10,
        truncate = false, nthreads = nt, niter = 5, unith = false)
    rows = collect(Int32(1):Int32(n))
    LinearTrees.refresh!(st, rows, 1)   # 20_000 rows in chunks of 5_000, all above any throw below
    return st, rows, nsets
end

# Fails if the spawned sibling's `give!` leaves its `finally`: the ids held by
# the tasks unwinding would stay out of the pool. Also fails if the sibling's
# error is swallowed rather than raised by `fetch`. The throw fires from
# `refresh!` at a node below `PARALLEL_MIN_ROWS`, where nothing is borrowed, so
# this covers the spawned-sibling guard alone.
@testset "an exception in a spawned sibling returns its scratch id" begin
    st, rows, nsets = poolstate(ThrowSmall(2_000), BIC(), min(4, Threads.nthreads()))
    @test_throws Exception LinearTrees.grow_subtree(st, rows, 1:20_000, 0, 0, 1, st.nodes, st.catmasks)
    @test sort(st.pool.free) == collect(2:nsets)
end

# Fails if `giveback!` leaves its `finally` at the `best_split` call site: the
# throw fires inside the root's threaded scan, which is the only node deep
# enough in the fit to have borrowed anything.
@testset "an exception in the threaded split search returns the borrowed ids" begin
    # at four workers the root borrows min(p, nt) - 1 = 3 of the 7 free ids
    st, rows, nsets = poolstate(MSE(), ThrowNth(1), min(4, Threads.nthreads()))
    @test_throws Exception LinearTrees.grow_subtree(st, rows, 1:20_000, 0, 0, 1, st.nodes, st.catmasks)
    @test sort(st.pool.free) == collect(2:nsets)
end

# Fails if `best_split` drops its join loop and returns on the first chunk's
# failure: the caller's `giveback!` then puts ids back that the later chunks are
# still scanning on. The window is opened by an `Event`, not by a sleep, and
# closed by `notify`, so the only bound in the test is on the negative check --
# with the join, `t` can never finish while chunk 2 and up are blocked, and
# without it `t` unwinds in microseconds.
@testset "the threaded split search does not return before its own chunks" begin
    nt = min(4, Threads.nthreads())
    if nt > 1
        rule = BlockOthers()
        st, rows, nsets = poolstate(MSE(), rule, nt; data = linedata())
        t = Threads.@spawn LinearTrees.grow_subtree(st, rows, 1:20_000, 0, 0, 1, st.nodes, st.catmasks)
        wait(rule.thrown)
        @test timedwait(() -> istaskdone(t), 1.0; pollint = 0.02) === :timed_out
        notify(rule.release)
        @test_throws TaskFailedException wait(t)
        @test sort(st.pool.free) == collect(2:nsets)
    end
end
