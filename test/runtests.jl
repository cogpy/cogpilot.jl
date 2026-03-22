using Test

@testset "CogPilot.jl" begin
    # Unit Tests
    @testset "ReservoirComputing" begin
        include("unit/test_reservoir_computing.jl")
    end

    @testset "BSeries" begin
        include("unit/test_bseries.jl")
    end

    @testset "RootedTrees" begin
        include("unit/test_rooted_trees.jl")
    end

    @testset "PSystems" begin
        include("unit/test_psystems.jl")
    end

    # End-to-End Tests
    @testset "Integration" begin
        include("e2e/test_integration.jl")
    end
end
