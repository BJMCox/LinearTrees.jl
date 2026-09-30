# Profiling pass over tree, boosting, continuous, and prediction cases.
#
#     julia --project=bench -t auto bench/profile.jl        # threaded pass
#     julia --project=bench -t 1    bench/profile.jl        # serial pass
#
# Case data lives in `bench/cases.jl`, all from `StableRNG` seeds, so two runs
# on one machine profile the same trees.
#
# Each case gets a warm-up call, a `BenchmarkTools` median, a CPU profile
# (flat text plus a PProf flame graph), and an allocation profile. PProf output
# goes to a per-case log so progress bars do not flood the terminal. Files go
# to `bench/profiles/`, suffixed `-t1` or `-tN` by `Threads.nthreads()`, so a
# serial and a threaded run do not overwrite each other.
#
# Two things this script does not do, because they need their own process:
#
#   julia --project=bench -t 1 --track-allocation=user bench/trackalloc.jl <case>
#   julia --project=bench bench/typecheck.jl        # JET + @code_warntype
#
# Set `LT_PROFILE_CASES=7,8` to run only boosting and continuous fits.
# Set `LT_PROFILE_CASES=9` to profile exact, refined, and coarse model-tree fits.
# Set `LT_PROFILE_OUT` to write artifacts outside the checkout.

using Profile, PProf, BenchmarkTools, Printf, LinearAlgebra
using LinearTrees, StaticArrays

include(joinpath(@__DIR__, "cases.jl"))

const OUT = get(ENV, "LT_PROFILE_OUT", joinpath(@__DIR__, "profiles"))
const NT = Threads.nthreads()
const TAG = "t$NT"
mkpath(OUT)
BLAS.set_num_threads(1)

"Flat listing of the current profile buffer, as a string."
function flat_print(mincount, sortedby)
    io = IOBuffer()
    Profile.print(IOContext(io, :displaysize => (24, 220)); format = :flat, sortedby, mincount)
    return String(take!(io))
end

"Sample count from the `Total snapshots: N` trailer of a flat listing."
function snapshot_count(flat)
    m = match(r"Total snapshots: (\d+)", flat)
    return m === nothing ? 0 : parse(Int, m[1])
end

"""
Profile repeated calls of `f` for approximately one second, write two flat
text listings and a PProf flame graph, and return `(nsamples, flat text)`.
`mincount` is 2% of the samples, as the original profiling brief asks.
The second listing sorts by `:overhead`, which is self time; the `Count`
column of the first is cumulative, so it ranks callers, not hot loops.
"""
function cpu_profile(name, f, median_ns)
    Profile.clear()
    Profile.init(; n = 10^8, delay = 0.001)
    repetitions = clamp(ceil(Int, 1e9 / median_ns), 1, 10_000)
    Profile.@profile for _ in 1:repetitions
        f()
    end
    ns = snapshot_count(flat_print(typemax(Int), :count))
    mc = max(1, round(Int, 0.02 * ns))
    flat = flat_print(mc, :count)
    open(joinpath(OUT, "$name-$TAG.flat.txt"), "w") do fh
        println(fh, "# $name  threads=$NT  repetitions=$repetitions  samples=$ns  mincount=$mc (2%), sorted by cumulative count")
        print(fh, flat)
        println(fh, "\n\n# same profile, sorted by self time (Overhead)")
        print(fh, flat_print(mc, :overhead))
    end
    open(joinpath(OUT, "$name-$TAG.pprof.log"), "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                PProf.pprof(; web = false, out = joinpath(OUT, "$name-$TAG.pb.gz"))
            end
        end
    end
    return ns, flat
end

"""
Allocation-profile `f` at a 1% sample rate and write both a PProf graph and a
per-line text summary. Returns the sampled byte total (scaled back up by the
sample rate is *not* applied: the raw sampled total is what the file reports).
"""
function alloc_profile(name, f)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 0.01 f()
    res = Profile.Allocs.fetch()
    tot = sum(a.size for a in res.allocs; init = 0)
    byline = Dict{String,Tuple{Int,Int}}()
    source_root = dirname(pathof(LinearTrees))
    for a in res.allocs
        st = a.stacktrace
        # first frame inside LinearTrees, else the innermost frame
        k = findfirst(fr -> startswith(string(fr.file), source_root), st)
        fr = k === nothing ? (isempty(st) ? nothing : st[1]) : st[k]
        key = fr === nothing ? "?" : "$(basename(string(fr.file))):$(fr.line) $(fr.func)"
        c, b = get(byline, key, (0, 0))
        byline[key] = (c + 1, b + a.size)
    end
    open(joinpath(OUT, "$name-$TAG.allocs.txt"), "w") do fh
        println(fh, "# $name  threads=$NT  sampled allocs=$(length(res.allocs))  sampled bytes=$tot  rate=0.01")
        for (k, (c, b)) in sort!(collect(byline); by = kv -> -kv[2][2])[1:min(25, length(byline))]
            @printf(fh, "%12d B  %8d allocs  %s\n", b, c, k)
        end
    end
    open(joinpath(OUT, "$name-$TAG.alloc.pprof.log"), "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                PProf.Allocs.pprof(res; web = false, out = joinpath(OUT, "$name-$TAG.alloc.pb.gz"))
            end
        end
    end
    return tot, length(res.allocs)
end

"Median time, allocation count and bytes from one `@benchmark` run."
function bench(f)
    b = @benchmark $f() samples = 5 seconds = 60 evals = 1
    return (t = median(b).time, allocs = median(b).allocs, bytes = median(b).memory)
end

const SUMMARY = String[]

function run_case(name, f; label = name)
    f()   # warm-up
    m = bench(f)
    ns, _ = cpu_profile(name, f, m.t)
    ab, an = alloc_profile(name, f)
    line = @sprintf("%-10s threads=%2d  median=%9.3f ms  allocs=%9d  bytes=%12d  cpu_samples=%6d  sampled_alloc_bytes=%d",
        label, NT, m.t / 1e6, m.allocs, m.bytes, ns, ab)
    println(line)
    push!(SUMMARY, line)
    return nothing
end

wanted = haskey(ENV, "LT_PROFILE_CASES") ? parse.(Int, split(ENV["LT_PROFILE_CASES"], ",")) : collect(1:9)

println("Julia ", VERSION, "  threads=", NT, "  BLAS=", BLAS.get_num_threads(),
    "  CPU=", Sys.cpu_info()[1].model)

if 1 in wanted
    X1, y1 = case1_data()
    run_case("case1-mse", () -> fit_tree(X1, y1; max_depth = 12))
end
if 2 in wanted
    X2, y2 = case2_data()
    run_case("case2-softmax", () -> fit_tree(X2, y2, Softmax(3); categorical = [10], max_depth = 12))
end
if 3 in wanted
    X3, y3 = case3_data()
    run_case("case3-mad", () -> fit_tree(X3, y3, MAD(); max_depth = 12, niter = 5))
end
if 4 in wanted
    df4, y4 = case4_data()
    run_case("case4-table", () -> fit(LinearTreeRegressorFit, df4, y4; max_depth = 12))
end
if 5 in wanted || 6 in wanted
    tree, Xp = case56_data()
    if 5 in wanted
        out = Vector{Float64}(undef, size(Xp, 1))
        run_case("case5-predict", () -> LinearTrees.predict(tree, Xp))
        run_case("case5-predict!", () -> LinearTrees.predict!(out, tree, Xp))
    end
    if 6 in wanted
        Xs = Xp[1:10_000, :]
        run_case("case6-shap", () -> shap(tree, Xs))
    end
end
if 7 in wanted
    X7, y7 = case7_data()
    run_case("case7-boost", () -> fit_boost(X7, y7;
        nrounds = 30, max_depth = 3, nthreads = 1, rng = StableRNG(7)))
end
if 8 in wanted
    X8, y8 = case8_data()
    run_case("case8-continuous", () -> fit_continuous_tree(X8, y8;
        pairs = [(1, 2)], max_splits = 4))
end
if 9 in wanted
    X9, y9 = case9_data()
    for (name, search) in (("exact", ExactSearch()),
            ("refined", BinnedSearch(nbins = 24)),
            ("coarse", BinnedSearch(nbins = 24, refine = false)))
        run_case("case9-modeltree-$name", () -> fit_model_tree(X9, y9;
            max_depth = 4, min_leaf = 12, lambda = 1.0, split_penalty = 0.0,
            split_search = search))
    end
end

open(joinpath(OUT, "summary-$TAG.txt"), "w") do fh
    println(fh, "Julia ", VERSION, "  threads=", NT, "  BLAS=", BLAS.get_num_threads(),
        "  CPU=", Sys.cpu_info()[1].model)
    foreach(l -> println(fh, l), SUMMARY)
end
println("\nwrote ", OUT)
