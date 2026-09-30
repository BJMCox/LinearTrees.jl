using Statistics
using StableRNGs

@testset "Continuous trees" begin
    @testset "A root agrees with conjugate linear regression" begin
        X = reshape(collect(range(-2.0, 2.0; length=41)), :, 1)
        y = @. 3.0 + 1.5 * X[:, 1] + 0.1 * cos(3 * X[:, 1])
        model = fit_continuous_tree(X, y; max_splits=0,
            coefficient_precision=0.2, noise_shape=3.0, noise_rate=0.7)
        B = hcat(ones(length(y)), X[:, 1] / 2)
        center, scale = mean(y), std(y; corrected=false)
        z = (y .- center) ./ scale
        H = B' * B + 0.2I
        coef = H \ (B' * z)
        shape = 3.0 + length(y) / 2
        rate = 0.7 + (sum(abs2, z - B * coef) + 0.2sum(abs2, coef)) / 2
        μ = center .+ scale .* (B * coef)
        latent = scale^2 * rate / shape .* diag(B * (H \ B'))
        @test predict(model, X) ≈ μ
        posterior = predictive(model, X; observation=false)
        @test posterior.location ≈ μ
        @test posterior.scale2 ≈ latent
        @test posterior.dof == 2shape
        observed = predictive(model, X)
        @test observed.scale2 ≈ latent .+ scale^2 * rate / shape
        @test predict!(similar(y), model, X) ≈ μ
        @test predict(model, X; batch_size=7) ≈ μ
        @test predictive(model, X; observation=false, batch_size=7).scale2 ≈ latent
    end

    @testset "Fixed response units agree with independent output posteriors" begin
        x = collect(range(-1.0, 1.0; length=9))
        X = reshape(x, :, 1)
        Y = hcat(2 .+ 1.5x .+ 0.1cos.(3x), fill(7.0, length(x)))
        center, scale = [1.0, -2.0], [2.0, 3.0]
        λ, a0, b0 = 0.4, 2.5, 0.8
        model = fit_continuous_tree(X, Y; max_splits=0,
            response_normalization=(; center, scale), coefficient_precision=λ,
            noise_shape=a0, noise_rate=b0)
        B = hcat(ones(length(x)), x)
        z = (Y .- center') ./ scale'
        H = B'B + λ * I
        coef = H \ (B'z)
        shape = a0 + length(x) / 2
        rate = [b0 + (sum(abs2, z[:, k] - B * coef[:, k]) +
            λ * sum(abs2, coef[:, k])) / 2 for k in axes(Y, 2)]
        query = [-1.3, -0.2, 0.8, 1.4]
        Q = hcat(ones(length(query)), query)
        expected = center' .+ (Q * coef) .* scale'
        latent = diag(Q * (H \ Q')) .* (scale.^2 .* rate ./ shape)'
        inputs = reshape(query, :, 1)
        posterior = predictive(model, inputs; observation=false, batch_size=2)
        observed = predictive(model, inputs; batch_size=2)
        @test predict(model, inputs; batch_size=2) ≈ expected
        @test posterior.location ≈ expected
        @test posterior.scale2 ≈ latent
        @test posterior.dof == fill(2shape, 2)
        @test observed.scale2 ≈ latent .+ (scale.^2 .* rate ./ shape)'

        # Normalization argument shape must not determine target rank.
        scalar = fit_continuous_tree(X, Y[:, 1]; max_splits=0,
            response_normalization=(scale=scale[1:1], center=center[1:1]),
            coefficient_precision=λ, noise_shape=a0, noise_rate=b0)
        singleton = fit_continuous_tree(X, Y[:, 1:1]; max_splits=0,
            response_normalization=(center=center[1], scale=scale[1]),
            coefficient_precision=λ, noise_shape=a0, noise_rate=b0)
        @test predict(scalar, inputs) ≈ expected[:, 1]
        @test predict(singleton, inputs) ≈ expected[:, 1:1]
        @test predictive(scalar, inputs).scale2 ≈ observed.scale2[:, 1]
        @test predictive(singleton, inputs).scale2 ≈ observed.scale2[:, 1:1]

        center .= 100
        scale .= 200
        @test predict(model, inputs) ≈ expected
        @test predictive(model, inputs; observation=false).scale2 ≈ latent
    end

    @testset "Repeated fits preserve earlier predictive distributions" begin
        X = 2rand(StableRNG(731), 96, 2) .- 1
        Y = hcat(1 .+ 2abs.(X[:, 1]) .+ 0.4X[:, 2],
            -0.5 .+ X[:, 1] .- abs.(X[:, 2]))
        model = fit_continuous_tree(X, Y; max_splits=3, n_thresholds=1)
        point = predict(model, X)
        expected = predictive(model, X)
        fit_continuous_tree(X, reverse(Y; dims=1); max_splits=3, n_thresholds=1,
            pairs=:all)
        repeated = predictive(model, X)
        @test predict(model, X) == point
        @test repeated.location == expected.location
        @test repeated.scale2 == expected.scale2
    end

    @testset "Collinear root retains a positive coefficient prior" begin
        for (n, λ) in ((4, 1e-20), (41, 1e-20), (41, 0.2))
            x = collect(range(-1.0, 1.0; length=n))
            X = hcat(x, x)
            y = sin.(x)
            model = fit_continuous_tree(X, y; max_splits=0,
                coefficient_precision=λ)
            query = [1.0 -1.0]
            marginal = predictive(model, query; observation=false)
            expected = setprecision(256) do
                B = BigFloat.(hcat(ones(n), x, x))
                z = BigFloat.((y .- model.ycenter[1]) ./ model.yscale[1])
                ridge = BigFloat(λ)
                precision = B'B + ridge * I
                coef = precision \ (B'z)
                rate = 1 + (sum(abs2, z - B * coef) + ridge * sum(abs2, coef)) / 2
                shape = 2 + n / 2
                row = BigFloat[1, 1, -1]
                location = model.ycenter[1] + model.yscale[1] * dot(row, coef)
                scale2 = dot(row, precision \ row) * rate / shape * model.yscale[1]^2
                score = size(B, 2) / 2 * log(ridge) - logdet(precision) / 2 -
                    shape * log(rate)
                (; location=Float64(location), scale2=Float64(scale2),
                    score=Float64(score))
            end
            @test only(marginal.location) ≈ expected.location rtol=1e-7 atol=1e-9
            @test only(marginal.scale2) ≈ expected.scale2 rtol=1e-7
            @test model.fit.post.score ≈ expected.score rtol=1e-7 atol=1e-8
        end
    end

    @testset "Shared geometry keeps output posteriors separate" begin
        x = collect(range(-1.0, 1.0; length=13))
        points = [(u, v) for u in x for v in x]
        X = [point[j] for point in points, j in 1:2]
        Y = hcat(abs.(X[:, 1]) .+ 0.3X[:, 2],
            20 .* abs.(X[:, 2]) .- 3X[:, 1])
        model = fit_continuous_tree(X, Y; n_thresholds=1, max_splits=3,
            min_leaf=4, coefficient_precision=1e-4, split_penalty=0.0)
        values = predict(model, X)
        @test size(values) == size(Y)
        @test maximum(abs, (values - Y) ./ std(Y; dims=1)) < 1e-3
        @test predict!(similar(Y), model, X) ≈ values
        @test size(predict(model, zeros(0, 2))) == (0, 2)
        marginal = predictive(model, X)
        @test marginal.location ≈ values
        @test size(marginal.scale2) == size(Y)
        @test predict(model, X; batch_size=7) ≈ values
        @test predictive(model, X; batch_size=7).scale2 ≈ marginal.scale2
        @test marginal.dof == fill(size(Y, 1) + 4, 2)
        @test all(marginal.scale2 .> predictive(model, X; observation=false).scale2)
        @test count(node -> node.feature != 0, model.fit.nodes) <= 3
        @test all(diff(model.fit.history) .> 0)

        # With fixed geometry, each column has exactly its scalar posterior.
        Z = (X .- model.xcenter') ./ model.xscale'
        for output in axes(Y, 2)
            z = reshape((Y[:, output] .- model.ycenter[output]) ./ model.yscale[output], :, 1)
            reference = LinearTrees.Continuous.fit_fixed(model.fit.nodes, Z, z;
                pairs=Tuple{Int,Int}[], coefficient_precision=1e-4, solver=:svd)
            expected = vec(LinearTrees.Continuous.predict(reference, Z)) .* model.yscale[output] .+ model.ycenter[output]
            @test values[:, output] ≈ expected atol=1e-10
            @test model.fit.post.rate[output] ≈ reference.post.rate[1]
        end
        singleton = fit_continuous_tree(X, Y[:, 1:1]; max_splits=0)
        @test size(predict(singleton, X)) == (size(X, 1), 1)
        @test predictive(singleton, X).dof isa Vector{Float64}
    end

    @testset "Wide leaf basis matches dense regression" begin
        X = 2rand(StableRNG(205), 80, 4) .- 1
        Y = hcat(X[:, 1] .* X[:, 2] .+ X[:, 3], X[:, 2] .* X[:, 4])
        model = fit_continuous_tree(X, Y; pairs=:all, max_splits=0)
        Z = (X .- model.xcenter') ./ model.xscale'
        B = hcat(ones(size(X, 1)), Z, [Z[:, j] .* Z[:, k] for j in 1:4 for k in j+1:4]...)
        response = (Y .- model.ycenter') ./ model.yscale'
        H = B' * B + 0.01I
        coefficients = H \ (B' * response)
        expected = (B * coefficients) .* model.yscale' .+ model.ycenter'
        @test predict(model, X) ≈ expected
        @test predictive(model, X).location ≈ expected
        @test model.fit.post.coef ≈ coefficients
        @test size(predictive(model, zeros(0, 4)).scale2) == (0, 2)
    end

    @testset "Coordinate slices join and extrapolate" begin
        x = collect(range(-1.0, 1.0; length=17))
        points = [(u, v) for u in x for v in x]
        X = [point[j] for point in points, j in 1:2]
        y = max.(X[:, 1], 0) .* max.(X[:, 2], 0)
        model = fit_continuous_tree(X, y; pairs=[(1, 2)], n_thresholds=1,
            max_splits=3, min_leaf=8, coefficient_precision=1e-5)
        @test any(move -> first(move) == :cross, model.fit.moves)
        @test maximum(abs, predict(model, X) - y) < 1e-3
        for feature in 1:2, fixed in (-0.7, 0.3, 2.0)
            slice = fill(fixed, 7, 2)
            slice[:, feature] .= [-2, -1, -1e-8, 0, 1e-8, 1, 2]
            values = predict(model, slice)
            @test abs(values[5] - values[3]) < 1e-6
            @test values[2] ≈ (values[1] + values[4]) / 2 atol=1e-10
            @test values[6] ≈ (values[4] + values[7]) / 2 atol=1e-10
        end
        @test predict(model, [2.0 2.0])[1] > 3.9
        duplicate = fit_continuous_tree(X, y; pairs=[(2, 1), (1, 2)], max_splits=0)
        @test predict(duplicate, X) ≈ predict(fit_continuous_tree(X, y; pairs=:all, max_splits=0), X)
    end

    @testset "Training scaling, constant columns and target rank" begin
        rng = StableRNG(911)
        X = hcat(rand(rng, 80) .* 2 .- 1, fill(3.0, 80))
        y = @. abs(X[:, 1]) + 0.01sin(3X[:, 1])
        model = fit_continuous_tree(X, y; max_splits=1)
        changed = fit_continuous_tree(X .* [1e6 7.0] .+ [12.0 -200.0],
            100 .+ 9y; max_splits=1)
        Q = [-3.0 3.0; -0.2 3.0; 0.4 3.0; 3.0 3.0]
        @test predict(changed, Q .* [1e6 7.0] .+ [12.0 -200.0]) ≈ 100 .+ 9predict(model, Q)
        @test predictive(changed, Q .* [1e6 7.0] .+ [12.0 -200.0]).scale2 ≈ 81predictive(model, Q).scale2
        fixed = fit_continuous_tree(X, y; max_splits=1,
            response_normalization=(center=0.25, scale=2.0))
        fixed_changed = fit_continuous_tree(X, 100 .+ 9y; max_splits=1,
            response_normalization=(center=102.25, scale=18.0))
        @test predict(fixed_changed, Q) ≈ 100 .+ 9predict(fixed, Q)
        @test predictive(fixed_changed, Q).scale2 ≈ 81predictive(fixed, Q).scale2
        @test predict(model, Q[2:2, :])[1] == predict(model, Q)[2]
        @test isempty(predict(model, zeros(0, 2)))
        @test isempty(predictive(model, zeros(0, 2)).scale2)
        constant = fit_continuous_tree(fill(0.1, 37, 2), fill(0.1, 37))
        @test constant.yscale == [1.0]
        @test predict(constant, fill(0.1, 4, 2)) == fill(0.1, 4)
        @test length(constant.fit.leaves) == 1
        numeric = fit_continuous_tree(Float32.(X), Float32.(y);
            max_splits=Int32(0), coefficient_precision=1, noise_shape=2, noise_rate=1)
        @test predict(numeric, X) isa Vector{Float64}
        copied = copy(Q)
        expected = predict(model, copied)
        predict!(view(copied, :, 1), model, copied; batch_size=2)
        @test copied[:, 1] == expected
        allcuts = fit_continuous_tree(reshape([-2.0, -1, 1, 2], :, 1), [-2.0, -1, -1, -2];
            n_thresholds=nothing, max_splits=1, min_leaf=1)
        @test all(isfinite, predict(allcuts, reshape([-3.0, 3.0], :, 1)))
    end

    @testset "Reject unsupported inputs" begin
        X = reshape(collect(1.0:12.0), :, 1)
        y = X[:, 1]
        for options in ((pairs=[(1, 1)],), (pairs=[(1, 2)],), (pairs=:auto,),
                (candidate_search=:unknown,), (max_splits=-1,), (max_depth=1.5,),
                (min_leaf=0,), (n_thresholds=0,), (coefficient_precision=0.0,),
                (noise_shape=Inf,), (noise_rate=-1.0,), (split_penalty=-1.0,),
                (coefficient_precision=big"1e-10000",), (noise_shape=big"1e-10000",),
                (noise_rate=big"1e-10000",))
            @test_throws ArgumentError fit_continuous_tree(X, y; options...)
        end
        @test_throws ArgumentError fit_continuous_tree(zeros(0, 1), Float64[])
        @test_throws ArgumentError fit_continuous_tree(zeros(12, 0), y)
        @test_throws ArgumentError fit_continuous_tree(X, zeros(12, 0))
        @test_throws DimensionMismatch fit_continuous_tree(X, y[1:11])
        @test_throws ArgumentError fit_continuous_tree(fill(NaN, 12, 1), y)
        @test_throws ArgumentError fit_continuous_tree(X, fill(Inf, 12))
        model = fit_continuous_tree(X, y; max_splits=0)
        @test_throws DimensionMismatch predict(model, zeros(2, 2))
        @test_throws DimensionMismatch predict!(zeros(12, 1), model, X)
        @test_throws ArgumentError predict(model, fill(NaN, 1, 1))
        @test_throws ArgumentError predictive(model, fill(Inf, 1, 1))
    end
end
