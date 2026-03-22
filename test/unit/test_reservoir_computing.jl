using Test
using ReservoirComputing
using Random

@testset "ReservoirComputing.jl ESN" begin
    # Test data
    Random.seed!(1234)
    train_len = 50
    predict_len = 10
    data = sin.(0.1 .* (1:(train_len + predict_len)))
    train_data = data[1:train_len]
    test_data = data[(train_len + 1):end]

    # ESN parameters
    res_size = 50
    radius = 1.2
    activation = tanh
    degree = 3
    sigma = 0.1
    beta = 0.001
    alpha = 1.0
    nla_type = NLADefault()
    in_size = 1
    out_size = 1

    # Create ESN
    esn = ESN(in_size, res_size, 
              train_data, activation, radius, degree, sigma, 
              alpha = alpha, nla_type = nla_type)

    # Test training
    W_out = ESNtrain(esn, beta)
    @test size(W_out) == (out_size, res_size + in_size)

    # Test prediction
    output = ESNpredict(esn, predict_len, W_out)
    @test length(output) == predict_len

    # Test error calculation
    error = nrmse(test_data, output)
    @test error isa Number
    @test error > 0.0
end
