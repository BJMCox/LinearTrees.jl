import Distributions
using Statistics

@testset "Fixed-geometry continuous ensemble" begin
    x = collect(range(-1.0, 1.0; length=31))
    X = reshape(x, :, 1)
    y = abs.(x) .+ 0.04sin.(5x)
    root = fit_continuous_tree(X, y; max_splits=0, coefficient_precision=0.4)
    split = fit_continuous_tree(X, y; max_splits=1, n_thresholds=1,
        min_leaf=4, split_penalty=0.0, coefficient_precision=0.4)
    @test length(split.fit.leaves) == 2

    prior = [log(2.0), 0.0]
    ensemble = fit_continuous_ensemble(X, y, [root, split]; logprior=prior,
        coefficient_precision=0.03)
    @test all(component.fit.post.score != source.fit.post.score
        for (component, source) in zip(ensemble.components, (root, split)))

    equal_prior = fit_continuous_ensemble(X, y, [root, split])
    shifted = fit_continuous_ensemble(X, y, [root, split]; logprior=fill(1e20, 2))
    @test shifted.weights ≈ equal_prior.weights
    excluded = fit_continuous_ensemble(X, y, [root, split]; logprior=[1e20, -Inf])
    @test excluded.weights == [1.0, 0.0]
    @test predict(excluded, X) == predict(excluded.components[1], X)

    query = reshape([-0.9, -0.3, 0.2, 0.8], :, 1)
    posterior = predictive(ensemble, query; batch_size=2)
    members = [predictive(component, query; batch_size=2)
        for component in ensemble.components]
    expected_mean = sum(ensemble.weights[j] .* members[j].location for j in 1:2)
    expected_within = sum(ensemble.weights[j] .* members[j].scale2 .*
        (members[j].dof / (members[j].dof - 2)) for j in 1:2)
    expected_between = sum(ensemble.weights[j] .*
        (members[j].location .- expected_mean).^2 for j in 1:2)
    @test predict(ensemble, query; batch_size=2) ≈ expected_mean
    @test posterior.location ≈ expected_mean
    @test posterior.within_variance ≈ expected_within
    @test posterior.between_variance ≈ expected_between
    @test posterior.variance ≈ expected_within .+ expected_between
    @test posterior.weights == ensemble.weights

    row = 2
    distributions = [member.location[row] +
        sqrt(member.scale2[row]) * Distributions.TDist(member.dof) for member in members]
    mixture = Distributions.MixtureModel(distributions,
        Distributions.Categorical(posterior.weights))
    @test mean(mixture) ≈ posterior.location[row]
    @test var(mixture) ≈ posterior.variance[row]
    @test Distributions.pdf(mixture, y[row]) ≈
        sum(posterior.weights[j] * Distributions.pdf(distributions[j], y[row]) for j in 1:2)
    lower = Distributions.quantile(mixture, 0.05)
    upper = Distributions.quantile(mixture, 0.95)
    @test Distributions.cdf(mixture, lower) ≈ 0.05 atol=1e-8
    @test Distributions.cdf(mixture, upper) ≈ 0.95 atol=1e-8

    copied = copy(query)
    expected = predict(ensemble, copied; batch_size=2)
    predict!(view(copied, :, 1), ensemble, copied; batch_size=2)
    @test copied[:, 1] ≈ expected
    Y = hcat(y, 2y .+ 0.1sin.(3x))
    multi = fit_continuous_ensemble(X, Y, [root, split]; logprior=prior)
    multi_posterior = predictive(multi, query; batch_size=2)
    @test size(multi_posterior.location) == (length(query), 2)
    @test size(multi_posterior.variance) == (length(query), 2)
    @test multi_posterior.location ≈ predict(multi, query; batch_size=2)
    for output in 1:2
        expected_output = sum(multi.weights[j] .*
            multi_posterior.components[j].location[:, output] for j in 1:2)
        @test multi_posterior.location[:, output] ≈ expected_output
    end

    before = predict(ensemble, query)
    root.fit.nodes[1].lo[1] = -0.5
    @test predict(ensemble, query) == before

    @test_throws ArgumentError fit_continuous_ensemble(X, y, [split, deepcopy(split)])
    grid = collect(range(-1.0, 1.0; length=7))
    crossed_X = reduce(vcat, ([u v] for u in grid for v in grid))
    crossed_y = crossed_X[:, 1] .+ crossed_X[:, 2] .+
        abs.(crossed_X[:, 1]) .* abs.(crossed_X[:, 2])
    reference = fit_continuous_tree(crossed_X, crossed_y; max_splits=0)
    normalized = LinearTrees._continuous_input(reference, crossed_X)
    response = reshape((crossed_y .- only(reference.ycenter)) ./
        only(reference.yscale), :, 1)
    core = LinearTrees.Continuous
    first_order = core.grow(core.root(2), 1, 1, 0.0)
    first_order = core.grow(first_order, 2, 2, 0.0)
    first_order = core.grow(first_order, 3, 2, 0.0)
    reverse_order = core.grow(core.root(2), 1, 2, 0.0)
    reverse_order = core.grow(reverse_order, 2, 1, 0.0)
    reverse_order = core.grow(reverse_order, 3, 1, 0.0)
    function crossed_source(nodes)
        fitted = core.fit_fixed(nodes, normalized, response; pairs=Tuple{Int,Int}[])
        ContinuousTree{true}(fitted, copy(reference.xcenter), copy(reference.xscale),
            copy(reference.ycenter), copy(reference.yscale))
    end
    @test_throws ArgumentError fit_continuous_ensemble(crossed_X, crossed_y,
        [crossed_source(first_order), crossed_source(reverse_order)])

    pair_X = hcat(x, x.^2)
    no_pairs = fit_continuous_tree(pair_X, y; max_splits=0, pairs=:none)
    pair = fit_continuous_tree(pair_X, y; max_splits=0, pairs=:all)
    @test length(fit_continuous_ensemble(pair_X, y, [no_pairs, pair]).components) == 2

    single_X = reshape([0.0], :, 1)
    single_y = [1.0]
    single = fit_continuous_tree(single_X, single_y; max_splits=0)
    heavy = fit_continuous_ensemble(single_X, single_y, [single]; noise_shape=0.2,
        response_normalization=(center=0.0, scale=1.0))
    @test predictive(heavy, single_X).variance == [Inf]
end

@testset "Continuous ensemble matches a sample-space continuity oracle" begin
    x = collect(range(-1.0, 1.0; length=25))
    X = reshape(x, :, 1)
    proposal = max.(x .- 1 / 3, 0)
    root = fit_continuous_tree(X, proposal; max_splits=0)
    split = fit_continuous_tree(X, proposal; max_splits=1, n_thresholds=2,
        min_leaf=3, split_penalty=0.0, coefficient_precision=0.2)
    t = split.fit.nodes[1].threshold
    Y = hcat(sin.(2x) .+ 0.2cos.(9x), 2 .+ 0.5x .+ 0.3sin.(5x))
    center, scale = [0.7, -1.0], [1.4, 2.5]
    λ, a0, b0 = 0.7, 2.5, 0.8
    prior = log.([2.0, 1.0])
    ensemble = fit_continuous_ensemble(X, Y, [root, split]; logprior=prior,
        response_normalization=(; center, scale), coefficient_precision=λ,
        noise_shape=a0, noise_rate=b0)
    query = [-1.4, -0.65, t, 0.55, 1.3]
    z = (Y .- center') ./ scale'
    a = a0 + size(Y, 1) / 2
    # Raw leaf coefficients satisfy c'θ=0. This explicit projector avoids the
    # production nullspace, projected design, and coefficient-space ridge solve.
    c = [1.0, t, -1.0, -t]
    projector = Matrix{Float64}(I, 4, 4) - c * c' / dot(c, c)
    routed(u) = u <= t ? [1.0, u, 0.0, 0.0] : [0.0, 0.0, 1.0, u]
    Dsplit = reduce(vcat, transpose.(routed.(x)))
    Qsplit = reduce(vcat, transpose.(routed.(query)))
    designs = ((hcat(ones(length(x)), x), hcat(ones(length(query)), query),
        Matrix{Float64}(I, 2, 2)), (Dsplit, Qsplit, projector))
    oracles = map(designs) do (D, Q, P)
        V = I + D * P * D' / λ
        k = Q * P * D' / λ
        b = [b0 + dot(z[:, output], V \ z[:, output]) / 2
            for output in axes(Y, 2)]
        location = center' .+ scale' .* (k * (V \ z))
        latent = diag(Q * P * Q' / λ - k * (V \ k'))
        scale2 = (1 .+ latent) .* (scale.^2 .* b ./ a)'
        logmass = -size(Y, 2) * logdet(Symmetric(V)) / 2 - a * sum(log, b)
        (;location, scale2, logmass)
    end
    logmass = [oracles[j].logmass + prior[j] for j in 1:2]
    weights = exp.(logmass .- maximum(logmass))
    weights ./= sum(weights)
    @test ensemble.weights ≈ weights rtol=1e-10
    posterior = predictive(ensemble, reshape(query, :, 1))
    for j in 1:2
        @test posterior.components[j].location ≈ oracles[j].location rtol=1e-10
        @test posterior.components[j].scale2 ≈ oracles[j].scale2 rtol=1e-10
        @test posterior.components[j].dof == fill(2a, size(Y, 2))
    end
    for output in axes(Y, 2), i in eachindex(query)
        expected = Distributions.MixtureModel([
            Distributions.LocationScale(oracles[j].location[i, output],
                sqrt(oracles[j].scale2[i, output]),
                Distributions.TDist(2a)) for j in 1:2], weights)
        actual = Distributions.MixtureModel([
            Distributions.LocationScale(part.location[i, output],
                sqrt(part.scale2[i, output]), Distributions.TDist(part.dof[output]))
            for part in posterior.components], posterior.weights)
        @test Distributions.logpdf(actual, sin(2query[i])) ≈
            Distributions.logpdf(expected, sin(2query[i])) rtol=1e-10
        @test Distributions.cdf(actual, sin(2query[i])) ≈
            Distributions.cdf(expected, sin(2query[i])) rtol=1e-10
    end

    shift, factor = [3.0, -4.0], [2.0, -0.5]
    changed = fit_continuous_ensemble(X, shift' .+ Y .* factor', [root, split];
        logprior=prior, coefficient_precision=λ, noise_shape=a0, noise_rate=b0,
        response_normalization=(center=shift .+ factor .* center,
            scale=abs.(factor) .* scale))
    changed_posterior = predictive(changed, reshape(query, :, 1))
    @test changed_posterior.weights ≈ weights rtol=1e-10
    @test changed_posterior.location ≈ shift' .+ posterior.location .* factor'
    @test changed_posterior.within_variance ≈ posterior.within_variance .* (factor.^2)'
    @test changed_posterior.between_variance ≈ posterior.between_variance .* (factor.^2)'
    @test changed_posterior.variance ≈ posterior.variance .* (factor.^2)'

    center .= 100
    scale .= 200
    unchanged = predictive(ensemble, reshape(query, :, 1))
    @test predict(ensemble, reshape(query, :, 1)) ≈ posterior.location
    @test unchanged.variance ≈ posterior.variance
    @test unchanged.weights == posterior.weights

end
