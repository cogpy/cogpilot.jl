using Test
using BSeries

@testset "BSeries.jl Composition" begin
    # Define two simple B-series
    a = BSeries.exp(1.0)
    b = BSeries.log(1.0)

    # Compose them
    c = a * b

    # The composition of exp and log should be the identity B-series
    # which has a coefficient of 1 for the empty tree and 0 for all others.
    # We'll check the first few coefficients.
    @test c.coefficients[RootedTrees.RootedTree([])] ≈ 1.0 atol=1e-12
    @test c.coefficients[RootedTrees.RootedTree([1])] ≈ 0.0 atol=1e-12
    @test c.coefficients[RootedTrees.RootedTree([1, 1])] ≈ 0.0 atol=1e-12
    @test c.coefficients[RootedTrees.RootedTree([2])] ≈ 0.0 atol=1e-12
end
