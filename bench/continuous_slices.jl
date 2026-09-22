# Run with the bench environment, then render with continuous_slices.py.
using LinearTrees, StableRNGs, DelimitedFiles

function continuous_slices(path)
    grid = collect(range(-1.0, 1.0; length=19))
    X = reduce(vcat, ([x y] for x in grid for y in grid))
    interaction = max.(X[:, 1], 0) .* max.(X[:, 2], 0)
    Y = hcat(1 .+ abs.(X[:, 1]) .+ 0.5X[:, 2] .+ 1.5interaction,
        -0.4 .+ 0.3X[:, 1] .- 1.2abs.(X[:, 2]) .+ 0.8interaction)
    Y .+= 0.03randn(StableRNG(922), size(Y))
    model = fit_continuous_tree(X, Y; pairs=[(1, 2)], max_splits=3,
        n_thresholds=1, min_leaf=10)
    rows = Vector{Float64}[]
    for feature in 1:2, fixed in (-0.7, 0.0, 0.7)
        x = collect(range(-1.2, 1.2; length=241))
        query = fill(fixed, length(x), 2)
        query[:, feature] .= x
        values = predict(model, query)
        for output in 1:2, i in eachindex(x)
            push!(rows, [output, feature, fixed, x[i], values[i, output]])
        end
    end
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "output,feature,fixed,x,mean")
        writedlm(io, reduce(hcat, rows)', ',')
    end
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    continuous_slices(isempty(ARGS) ? joinpath(tempdir(), "continuous-slices.csv") : only(ARGS))
end
