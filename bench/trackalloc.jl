# Line-level allocation counts for one case under Julia's malloc log.
#
#     julia --project=bench --track-allocation=user bench/trackalloc.jl 1
#
# The case runs once as a warm-up, `Profile.clear_malloc_data()` drops
# everything the compiler allocated, then the case runs again. Julia writes
# `src/*.jl.mem` at exit; this script copies the LinearTrees ones into
# `bench/profiles/mem-case<N>/` and prints the top lines.
#
# One process per case: the `.mem` files are per source file, not per call, so
# two cases in one process would be summed together.

using Profile, Printf
include(joinpath(@__DIR__, "cases.jl"))

const CASE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const SRC = normpath(joinpath(@__DIR__, "..", "src"))
const DST = joinpath(@__DIR__, "profiles", "mem-case$CASE")

work = if CASE == 1
    X, y = case1_data()
    () -> fit_tree(X, y; max_depth = 12)
elseif CASE == 2
    X, y = case2_data()
    () -> fit_tree(X, y, Softmax(3); categorical = [10], max_depth = 12)
elseif CASE == 3
    X, y = case3_data()
    () -> fit_tree(X, y, MAD(); max_depth = 12, niter = 5)
elseif CASE == 4
    df, y = case4_data()
    () -> fit(LinearTreeRegressorFit, df, y; max_depth = 12)
elseif CASE == 5
    tree, Xp = case56_data()
    () -> LinearTrees.predict(tree, Xp)
elseif CASE == 6
    tree, Xp = case56_data()
    Xs = Xp[1:10_000, :]
    () -> shap(tree, Xs)
else
    error("unknown case $CASE")
end

work()
Profile.clear_malloc_data()
work()

atexit() do
    mkpath(DST)
    rows = Tuple{Int,String}[]
    for f in readdir(SRC; join = true)
        endswith(f, ".jl.mem") || continue
        cp(f, joinpath(DST, basename(f)); force = true)
        base = basename(f)
        for (ln, line) in enumerate(eachline(f))
            m = match(r"^\s*(\d+)\s(.*)$", line)
            m === nothing && continue
            b = parse(Int, m[1])
            b > 0 && push!(rows, (b, @sprintf("%-16s:%-5d %s", base, ln, strip(m[2]))))
        end
    end
    sort!(rows; by = first, rev = true)
    open(joinpath(DST, "top.txt"), "w") do io
        println(io, "# case $CASE, --track-allocation=user, one call after clear_malloc_data()")
        for (b, s) in rows[1:min(30, length(rows))]
            @printf(io, "%14d B  %s\n", b, s)
        end
    end
    println("\n== case $CASE top allocating lines ==")
    for (b, s) in rows[1:min(15, length(rows))]
        @printf("%14d B  %s\n", b, s)
    end
end
