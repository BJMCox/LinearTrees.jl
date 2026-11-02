using LinearTrees
using Test
using Aqua
using JET
using LinearAlgebra

@testset "LinearTrees.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LinearTrees)
    end
    # JET's verdicts track the compiler: on 1.10 it reports nine false positives inside
    # Base broadcast and vcat that 1.12 infers cleanly, so the lint gate runs on 1.12+.
    if VERSION >= v"1.12"
        @testset "Code linting (JET.jl)" begin
            # Abstract package scans misreport valid Base.supertype/getproperty
            # calls on Julia 1.14 prereleases. Keep the broad scan on stable Julia
            # and check concrete public calls on every compiler tested here.
            if isempty(VERSION.prerelease)
                # Julia 1.13's generic QR/norm analysis reaches impossible Union{}
                # views and iterate(::Nothing) inside these two stdlib functions.
                # Standalone QR/norm calls reproduce them. Check concrete refits below.
                ignored =
                    v"1.13" <= VERSION < v"1.14" ?
                    (
                        JET.AnyFrameMethod(LinearAlgebra.generic_norm2),
                        JET.AnyFrameMethod(LinearAlgebra.norm_recursive_check),
                    ) : ()
                JET.test_package(LinearTrees; ignored_modules = ignored)
            end
            for T in (Float32, Float64)
                JET.test_call(fit_tree, (Matrix{T}, Vector{T}))
                JET.test_call(fit_boost, (Matrix{T}, Vector{T}))
                JET.test_call(predict, (LinearTree{T,T,MSE}, Matrix{T}))
                JET.test_opt(
                    refit_leaves,
                    (LinearTree{T,T,MSE}, Matrix{T}, Vector{T});
                    target_modules = (LinearTrees,),
                )
            end
        end
    end
    include("loss.jl")
    include("predict.jl")
    include("continuous.jl")
    include("continuous_ensemble.jl")
    include("continuous_solver.jl")
    include("accumulate.jl")
    include("select.jl")
    include("boost_rule.jl")
    include("boost_frozen.jl")
    include("boost_fit.jl")
    include("boost_safeguard.jl")
    include("boost_shap.jl")
    include("boost_interfaces.jl")
    include("scan.jl")
    include("search.jl")
    include("hybrid.jl")
    include("fit.jl")
    include("centered.jl")
    include("refit.jl")
    include("prune.jl")
    include("modeltree.jl")
    include("fit_losses.jl")
    include("review_extremes.jl")
    include("pilot_reference.jl")
    include("categorical.jl")
    include("softmax.jl")
    include("threads.jl")
    include("partition.jl")
    include("importance.jl")
    include("shap.jl")
    include("lossfunctions.jl")
    include("show.jl")
    include("serialize.jl")
    include("statsapi.jl")
    include("mlj.jl")
    include("allocations.jl")
end
