using Test
using RootedTrees

@testset "RootedTrees.jl Enumeration" begin
    # Test the number of rooted trees with a given number of nodes
    # OEIS A000081
    @test number_of_rooted_trees(1) == 1
    @test number_of_rooted_trees(2) == 1
    @test number_of_rooted_trees(3) == 2
    @test number_of_rooted_trees(4) == 4
    @test number_of_rooted_trees(5) == 9
    @test number_of_rooted_trees(6) == 20

    # Test specific tree properties
    t = rooted_tree("[[[]]]")
    @test order(t) == 4
    @test σ(t) == 2
    @test γ(t) == 8
    @test α(t) == 1.0 / (σ(t) * γ(t))
end
