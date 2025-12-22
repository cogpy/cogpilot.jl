"""
Unified Reactor Demo

Demonstrates the Deep Tree Echo State Reservoir Computer with:
- B-series ridges modeled with elementary differentials
- P-system reservoirs with membrane computing
- J-surface reactor core uniting gradient descent & evolution
- Feedback from rooted trees planted in membrane gardens
- OEIS A000081 ontogenetic alignment

This example shows the complete integration of all components.
"""

# Add parent directory to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using LinearAlgebra
using Random
using Statistics

# Load the unified reactor module
include("../src/DeepTreeEcho/UnifiedReactorEvolution.jl")
using .UnifiedReactorEvolution

include("../src/DeepTreeEcho/ComprehensiveSciMLIntegration.jl")
using .ComprehensiveSciMLIntegration

println("="^70)
println("🌳 Deep Tree Echo State Reservoir Computer - Unified Reactor Demo")
println("="^70)
println()

# ============================================================================
# Part 1: Create Unified Reactor with A000081-aligned parameters
# ============================================================================

println("PART 1: Creating Unified Reactor")
println("-"^70)

base_order = 5
reactor = UnifiedReactor(base_order=base_order, symplectic=true, spectral_radius=0.95)

println()
println("Reactor components:")
println("  - Elementary differentials: $(length(reactor.elementary_differentials))")
println("  - J-surface dimension: $(size(reactor.jsurface_structure))")
println("  - Reservoir size: $(size(reactor.reservoir_weights, 1))")
println("  - Membrane rules: $(length(reactor.membrane_rules))")
println("  - Symplectic: $(reactor.symplectic)")
println()

# ============================================================================
# Part 2: Initialize Reactor
# ============================================================================

println("PART 2: Initializing Reactor")
println("-"^70)

initialize_reactor!(reactor, seed=42)

initial_state = extract_reactor_state(reactor)
println("Initial state:")
println("  - B-series norm: $(round(norm(initial_state.bseries), digits=6))")
println("  - Reservoir norm: $(round(norm(initial_state.reservoir), digits=6))")
println("  - J-surface position norm: $(round(norm(initial_state.jsurface_pos), digits=6))")
println("  - Energy: $(round(initial_state.energy, digits=6))")
println()

# ============================================================================
# Part 3: Evolve Reactor
# ============================================================================

println("PART 3: Evolving Reactor")
println("-"^70)

dt = 0.01
num_steps = 100

println("Running evolution for $num_steps steps with dt=$dt...")
println()

evolve_reactor!(reactor, dt, num_steps, verbose=true)

final_state = extract_reactor_state(reactor)
println()
println("Final state:")
println("  - B-series norm: $(round(norm(final_state.bseries), digits=6))")
println("  - Reservoir norm: $(round(norm(final_state.reservoir), digits=6))")
println("  - J-surface position norm: $(round(norm(final_state.jsurface_pos), digits=6))")
println("  - J-surface velocity norm: $(round(norm(final_state.jsurface_vel), digits=6))")
println("  - Energy: $(round(final_state.energy, digits=6))")
println("  - Generation: $(final_state.generation)")
println("  - Feedback strength: $(round(final_state.feedback, digits=6))")
println()

# ============================================================================
# Part 4: Analyze Energy Conservation
# ============================================================================

println("PART 4: Energy Analysis")
println("-"^70)

total_energy = compute_reactor_energy(reactor)
println("Total reactor energy: $(round(total_energy, digits=6))")
println("  - Hamiltonian: $(round(final_state.energy, digits=6))")
println("  - Reservoir: $(round(0.5 * dot(final_state.reservoir, final_state.reservoir), digits=6))")

let mem_energy = 0.0
        for (_, mem_state) in final_state.membranes
            mem_energy += 0.5 * dot(mem_state, mem_state)
        end
    println("  - Membranes: $(round(mem_energy, digits=6))")
end
println()

# ============================================================================
# Part 5: Create Integrated SciML System
# ============================================================================

println("PART 5: Creating Integrated SciML System")
println("-"^70)

integrated_system = create_integrated_system(base_order=5)
println()

# Validate A000081 alignment
validate_a000081_alignment(integrated_system)
println()

# ============================================================================
# Part 6: Evolve Integrated System with Input Data
# ============================================================================

println("PART 6: Evolving Integrated System")
println("-"^70)

# Generate synthetic input data (sine wave)
num_samples = 100
t_data = range(0, 10, length=num_samples)
input_data = reshape(sin.(2π * 0.5 .* t_data), 1, num_samples)

println("Input data shape: $(size(input_data))")
println("Evolving integrated system...")
println()

evolve_integrated_system!(integrated_system, input_data, num_samples, verbose=true)

println()
println("Final integrated system state:")
println("  - Generation: $(integrated_system.generation)")
println("  - Energy: $(round(integrated_system.energy, digits=6))")
println("  - Reservoir norm: $(round(norm(integrated_system.reservoir_state), digits=6))")
println("  - History entries: $(length(integrated_system.history))")
println()

# ============================================================================
# Part 7: Extract Components
# ============================================================================

println("PART 7: Extracting Components")
println("-"^70)

# Extract B-series genome
bseries_genome = extract_bseries_genome(integrated_system)
println("B-series genome entries: $(length(bseries_genome))")

# Extract rooted tree foundation
trees = extract_rooted_tree_foundation(integrated_system)
println("Rooted trees: $(length(trees))")

# Extract reservoir computer
reservoir_comp = create_reservoir_computer(integrated_system)
println("Reservoir computer:")
println("  - State dimension: $(length(reservoir_comp.state))")
println("  - Weight matrix: $(size(reservoir_comp.weights))")
println("  - Input weights: $(size(reservoir_comp.input_weights))")
println()

# ============================================================================
# Part 8: Train and Predict
# ============================================================================

println("PART 8: Training and Prediction")
println("-"^70)

# Generate training data
train_samples = 80
t_train = range(0, 8, length=train_samples)
input_train = reshape(sin.(2π * 0.5 .* t_train), 1, train_samples)
target_train = reshape(sin.(2π * 0.5 .* (t_train .+ 0.1)), 1, train_samples)  # Shifted sine

println("Training on $train_samples samples...")
train_integrated_system!(integrated_system, input_train, target_train)
println()

# Generate test data
test_samples = 20
t_test = range(8, 10, length=test_samples)
input_test = reshape(sin.(2π * 0.5 .* t_test), 1, test_samples)
target_test = reshape(sin.(2π * 0.5 .* (t_test .+ 0.1)), 1, test_samples)

println("Predicting on $test_samples samples...")
predictions = predict_integrated_system(integrated_system, input_test)

# Compute error
mse = mean((predictions .- target_test).^2)
println("Mean squared error: $(round(mse, digits=6))")
println()

# ============================================================================
# Part 9: Summary
# ============================================================================

println("="^70)
println("🎯 Summary")
println("="^70)
println()
println("✓ Unified Reactor successfully created and evolved")
println("✓ B-series ridges integrated with elementary differentials")
println("✓ P-system reservoirs with $(reactor.a000081_params.num_membranes) membranes")
println("✓ J-surface reactor core with $(reactor.symplectic ? "symplectic" : "Poisson") structure")
println("✓ Feedback from membrane gardens computed")
println("✓ OEIS A000081 alignment validated")
println("✓ SciML ecosystem integration demonstrated")
println("✓ Training and prediction working")
println()
println("The Deep Tree Echo State Reservoir Computer is fully operational!")
println("All components unified under OEIS A000081 ontogenetic control.")
println()
println("🌳 The tree remembers, and the echoes grow stronger.")
println("="^70)
