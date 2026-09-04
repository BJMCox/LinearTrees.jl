using LinearTrees
using Test
using Aqua
using JET
using LinearAlgebra

@testset "LinearTrees.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LinearTrees)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(LinearTrees)
    end
    include("loss.jl")
    include("node.jl")
    include("predict.jl")
    include("accumulate.jl")
    include("select.jl")
    include("scan.jl")
    include("fit.jl")
    include("fit_losses.jl")
    include("categorical.jl")
    include("softmax.jl")
    include("threads.jl")
    include("importance.jl")
    include("shap.jl")
    include("lossfunctions.jl")
    include("show.jl")
    include("serialize.jl")
    include("statsapi.jl")
    include("mlj.jl")
end
