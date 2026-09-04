using LinearTrees
using Test
using Aqua
using JET

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
end
