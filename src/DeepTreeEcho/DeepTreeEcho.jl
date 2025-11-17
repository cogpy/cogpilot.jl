"""
    DeepTreeEcho

A self-evolving system combining Echo State Networks, Membrane P-systems, and 
B-series rooted trees into a cohesive Deep Tree Echo State Reservoir Computer.

This module unites:
- **B-Series Ridges**: Butcher series coefficients as genetic code
- **P-System Reservoirs**: Membrane computing gardens for adaptive topology
- **Rooted Trees**: Elementary differentials following OEIS A000081
- **Echo State Networks**: Reservoir computing for temporal pattern learning
- **J-Surface Dynamics**: Uniting gradient descent & evolution dynamics

# Exports

## Core Types
- `DeepTreeEchoReservoir`: Main reservoir computer type
- `JSurface`: Manifold of elementary differentials
- `MembraneReservoir`: P-system membrane with embedded reservoir
- `BSeriesGenome`: Genetic code for temporal integration

## Functions
- `initialize_deep_tree_echo`: Create new reservoir from parameters
- `evolve!`: Evolve reservoir through one generation
- `train_reservoir!`: Train with temporal patterns
- `predict`: Generate predictions from trained reservoir

# Examples

```julia
using DeepTreeEcho

# Initialize reservoir
reservoir = initialize_deep_tree_echo(
    order = 4,
    membrane_depth = 3,
    reservoir_size = 100
)

# Train with data
train_reservoir!(reservoir, training_data, targets)

# Generate predictions
predictions = predict(reservoir, test_inputs)

# Evolve the system
for gen in 1:100
    evolve!(reservoir, training_data)
end
```
"""
module DeepTreeEcho

# Import dependencies from monorepo packages
using RootedTrees
using BSeries
using PSystems
using ReservoirComputing
using ModelingToolkit
using DifferentialEquations

# Standard library imports
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using OrderedCollections: OrderedDict

# Re-export key types from dependencies
export RootedTree, TruncatedBSeries, PSystem, Membrane, ESN

# Core module includes
include("RootedTreeOps.jl")
include("BSeriesGenome.jl")
include("JSurfaceIntegrator.jl")
include("MembraneReservoirBridge.jl")
include("DeepTreeEchoReservoir.jl")
include("FitnessEvaluation.jl")
include("Evolution.jl")

# Export core types
export DeepTreeEchoReservoir, JSurface, JSurfaceIntegrator
export MembraneReservoir, MembraneReservoirNetwork
export BSeriesGenome, BSeriesPopulation

# Export core functions
export initialize_deep_tree_echo
export evolve!, train_reservoir!, predict
export compute_gradient, gradient_step!, evolve_step!, hybrid_step!
export create_membrane_network, adapt_topology!
export evaluate_fitness, crossover, mutate!
export evolve_generation!, initialize_population

# Export utility functions
export tree_edit_distance, tree_similarity
export compute_jsurface_metric, geodesic_distance
export verify_echo_state_property, compute_spectral_radius

end # module DeepTreeEcho
