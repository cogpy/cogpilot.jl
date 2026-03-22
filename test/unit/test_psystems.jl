using Test
using PSystems

@testset "PSystems.jl Basic Simulation" begin
    # Define a simple P-system for string rewriting
    rules = Dict("a" => "ab", "b" => "a")
    initial_objects = ["a"]
    psystem = PSystem(rules, initial_objects)

    # Run the simulation for a few steps
    @test step(psystem) == ["a", "b"]
    @test step(psystem, 2) == ["a", "b", "a"]
    @test step(psystem, 3) == ["a", "b", "a", "a", "b"]

    # Test with a different initial state
    psystem2 = PSystem(rules, ["b"])
    @test step(psystem2) == ["a"]
end
