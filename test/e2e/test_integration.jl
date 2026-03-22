using Test
using ModelingToolkit
using DifferentialEquations
using ReservoirComputing
using RootedTrees

# This is a conceptual integration test. A real-world scenario would be more complex.
# We will model a simple chaotic system, use a reservoir to learn its dynamics,
# and then use rooted trees to analyze the structure of the learned model.

@testset "E2E: Lorenz System -> ESN -> Tree Analysis" begin

    # 1. Define the Lorenz System using ModelingToolkit
    @parameters t σ=10.0 ρ=28.0 β=8/3
    @variables x(t)=1.0 y(t)=0.0 z(t)=0.0
    D = Differential(t)

    eqs = [D(x) ~ σ*(y-x),
           D(y) ~ x*(ρ-z)-y,
           D(z) ~ x*y - β*z]
    @named lorenz_system = ODESystem(eqs)

    # 2. Generate data from the Lorenz system
    prob = ODEProblem(lorenz_system, [], (0.0, 20.0))
    sol = solve(prob, Tsit5())
    data = sol(0.0:0.01:20.0)
    training_data = data[1, 1:1500]'
    testing_data = data[1, 1501:2000]'

    # 3. Train an ESN on the Lorenz data
    res_size = 100
    esn = ESN(1, res_size, training_data, tanh, 1.2, 3, 0.1, alpha=0.8)
    W_out = ESNtrain(esn, 1e-4)
    @test size(W_out) == (1, res_size + 1)

    # 4. Predict with the ESN
    output = ESNpredict(esn, 500, W_out)
    @test length(output) == 500

    # 5. This is a simplified analysis. A full analysis would involve
    #    mapping the ESN's learned state transitions to a tree structure.
    #    For this test, we'll just create a tree representing the model's order.
    #    A rooted tree of order equal to the reservoir size.
    #    This is a placeholder for a more complex analysis.
    tree = rooted_tree(res_size)
    @test order(tree) == res_size

end
