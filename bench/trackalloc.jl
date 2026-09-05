# Line-level allocation counts for one case under Julia's malloc log.
#
#     julia --project=bench -t 1 --track-allocation=user bench/trackalloc.jl 1
#     julia --project=bench bench/trackalloc.jl 1 report
#
# The first command runs the case once as a warm-up, calls
# `Profile.clear_malloc_data()` to drop everything the compiler allocated, then
# runs it again. Julia writes `src/*.jl.mem` when that process exits. The
# second command reads those files, copies them into
# `bench/profiles/mem-case<N>/` and prints the top lines; it is a separate
# process because the malloc log is only complete once the first one is gone.
# Julia names the log files `<source>.<pid>.mem`, so the run phase records its
# own pid and the report phase reads only that run's files.
#
# Run it on one thread: the per-line counters are not synchronised, so a
# threaded run attributes bytes to whichever thread got there last.
#
# One process per case: the `.mem` files are per source file, not per call, so
# two cases in one process would be summed together.

using Printf

const CASE = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
const REPORT = length(ARGS) >= 2 && ARGS[2] == "report"
const SRC = normpath(joinpath(@__DIR__, "..", "src"))
const DST = joinpath(@__DIR__, "profiles", "mem-case$CASE")

if REPORT
    pid = strip(read(joinpath(DST, "pid.txt"), String))
    rows = Tuple{Int,String}[]
    for f in readdir(SRC; join = true)
        endswith(f, ".$pid.mem") || continue
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
    println("== case $CASE top allocating lines ==")
    for (b, s) in rows[1:min(15, length(rows))]
        @printf("%14d B  %s\n", b, s)
    end
    exit(0)
end

using Profile
include(joinpath(@__DIR__, "cases.jl"))

# The malloc log costs a counter at every allocation site and runs one to two
# orders of magnitude slower than the profiled build, so this pass takes the
# first `1/LT_TRACKALLOC_DIV` of each case's rows. It answers *which lines*
# allocate; the absolute byte totals per call come from `bench/profile.jl`.
const DIV = parse(Int, get(ENV, "LT_TRACKALLOC_DIV", "10"))
head(A::AbstractMatrix) = A[1:(size(A, 1) ÷ DIV), :]
head(v::AbstractVector) = v[1:(length(v) ÷ DIV)]

work = if CASE == 1
    X, y = case1_data()
    X, y = head(X), head(y)
    () -> fit_tree(X, y; max_depth = 12)
elseif CASE == 2
    X, y = case2_data()
    X, y = head(X), head(y)
    () -> fit_tree(X, y, Softmax(3); categorical = [10], max_depth = 12)
elseif CASE == 3
    X, y = case3_data()
    X, y = head(X), head(y)
    () -> fit_tree(X, y, MAD(); max_depth = 12, niter = 5)
elseif CASE == 4
    df, y = case4_data()
    df, y = df[1:(nrow(df) ÷ DIV), :], head(y)
    () -> fit(LinearTreeRegressorFit, df, y; max_depth = 12)
elseif CASE == 5
    tree, Xp = case56_data()
    Xp = head(Xp)
    () -> LinearTrees.predict(tree, Xp)
elseif CASE == 6
    tree, Xp = case56_data()
    Xs = Xp[1:(10_000 ÷ DIV), :]
    () -> shap(tree, Xs)
else
    error("unknown case $CASE")
end

mkpath(DST)
write(joinpath(DST, "pid.txt"), string(getpid()))
work()
Profile.clear_malloc_data()
work()
println("case $CASE done; now run: julia --project=bench bench/trackalloc.jl $CASE report")
