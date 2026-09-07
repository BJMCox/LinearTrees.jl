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
            JET.test_package(LinearTrees)
        end
    end
    include("loss.jl")
    include("predict.jl")
    include("accumulate.jl")
    include("select.jl")
    include("boost_rule.jl")
    include("boost_frozen.jl")
    include("boost_fit.jl")
    include("scan.jl")
    include("fit.jl")
    include("fit_losses.jl")
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
