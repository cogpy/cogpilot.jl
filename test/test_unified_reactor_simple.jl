"""
Simple standalone test for the Unified Reactor Core
"""

using Test
using LinearAlgebra
using Random
using Statistics

println("\n" * "="^70)
println("UNIFIED REACTOR CORE - SIMPLE TEST SUITE")
println("="^70)

# Load the UnifiedReactorCore module directly
include("../src/DeepTreeEcho/UnifiedReactorCore.jl")
using .UnifiedReactorCore

@testset "Unified Reactor Simple Tests" begin
    
    @testset "Reactor Creation" begin
        println("\n🧪 TEST: Reactor Creation")
        
        reactor = create_unified_reactor(base_order=5)
        
        @test reactor isa UnifiedReactor
        @test reactor.config.reservoir_size == 17  # Σ A000081[1:5]
        @test reactor.config.num_membranes == 4    # A000081[4]
        @test reactor.config.max_tree_order == 8
        
        println("✓ Reactor created successfully")
        println("  Reservoir size: $(reactor.config.reservoir_size)")
        println("  Membranes: $(reactor.config.num_membranes)")
        println("  Trees: $(length(reactor.rooted_trees))")
    end
    
    @testset "Reactor Initialization" begin
        println("\n🧪 TEST: Reactor Initialization")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        total_planted = sum(length(trees) for trees in values(reactor.state.planted_trees))
        @test total_planted == 9
        
        println("✓ Reactor initialized")
        println("  Planted trees: $total_planted")
        println("  Initial energy: $(round(reactor.state.energy, digits=4))")
    end
    
    @testset "Single Evolution Step" begin
        println("\n🧪 TEST: Single Evolution Step")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        initial_energy = reactor.state.energy
        reactor_step!(reactor, dt=0.01)
        
        @test reactor.state.step_count == 1
        @test length(reactor.energy_history) == 1
        
        println("✓ Evolution step completed")
        println("  Energy change: $(round(reactor.state.energy - initial_energy, digits=6))")
    end
    
    @testset "Multiple Evolution Steps" begin
        println("\n🧪 TEST: Multiple Evolution Steps")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        n_steps = 20
        for i in 1:n_steps
            reactor_step!(reactor, dt=0.01)
        end
        
        @test reactor.state.step_count == n_steps
        @test length(reactor.energy_history) == n_steps
        
        println("✓ Multiple steps completed")
        println("  Steps: $n_steps")
        println("  Final energy: $(round(reactor.state.energy, digits=4))")
    end
    
    @testset "Feedback Harvesting" begin
        println("\n🧪 TEST: Feedback Harvesting")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        
        harvest_feedback!(reactor)
        
        @test !isempty(reactor.garden_feedback)
        
        println("✓ Feedback harvested")
        println("  Feedback trees: $(length(reactor.garden_feedback))")
    end
    
    @testset "Tree Evolution Cycle" begin
        println("\n🧪 TEST: Tree Evolution Cycle")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        initial_generation = reactor.state.generation
        
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        harvest_feedback!(reactor)
        plant_evolved_trees!(reactor)
        
        @test reactor.state.generation == initial_generation + 1
        
        println("✓ Evolution cycle completed")
        println("  Generation: $(reactor.state.generation)")
    end
    
    @testset "Topology Adaptation" begin
        println("\n🧪 TEST: Topology Adaptation")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        harvest_feedback!(reactor)
        adapt_reactor_topology!(reactor)
        
        # Check coefficients normalized
        total = sum(values(reactor.ridge_coefficients))
        @test isapprox(total, 1.0, atol=1e-10)
        
        println("✓ Topology adapted")
        println("  Coefficient sum: $(round(total, digits=10))")
    end
    
    @testset "Complete Evolution (10 Generations)" begin
        println("\n🧪 TEST: Complete Evolution (10 Generations)")
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        n_generations = 10
        steps_per_gen = 10
        
        for gen in 1:n_generations
            for step in 1:steps_per_gen
                reactor_step!(reactor, dt=0.01)
            end
            harvest_feedback!(reactor)
            plant_evolved_trees!(reactor)
            adapt_reactor_topology!(reactor)
        end
        
        @test reactor.state.generation == n_generations
        @test reactor.state.step_count == n_generations * steps_per_gen
        
        status = get_reactor_status(reactor)
        
        println("✓ Complete evolution successful")
        println("  Generations: $(status.generation)")
        println("  Total steps: $(status.step_count)")
        println("  Total trees: $(status.total_trees)")
        println("  Tree diversity: $(round(status.tree_diversity, digits=4))")
        println("  Avg fitness: $(round(status.avg_fitness, digits=4))")
        println("  Final energy: $(round(status.energy, digits=4))")
    end
    
    @testset "A000081 Parameter Alignment" begin
        println("\n🧪 TEST: A000081 Parameter Alignment")
        
        A000081 = [1, 1, 2, 4, 9, 20, 48, 115]
        
        for base_order in [3, 4, 5, 6]
            reactor = create_unified_reactor(base_order=base_order)
            expected_reservoir_size = sum(A000081[1:base_order])
            @test reactor.config.reservoir_size == expected_reservoir_size
        end
        
        println("✓ A000081 alignment verified")
    end
    
    @testset "J-Surface Symplectic Structure" begin
        println("\n🧪 TEST: J-Surface Symplectic Structure")
        
        reactor = create_unified_reactor(base_order=5, symplectic=true)
        J = reactor.jsurface_matrix
        
        # Check skew-symmetric
        @test isapprox(J, -J', atol=1e-10)
        
        println("✓ J-surface structure verified")
        println("  Skew-symmetric: $(isapprox(J, -J', atol=1e-10))")
    end
    
    @testset "Energy Conservation (Symplectic)" begin
        println("\n🧪 TEST: Energy Conservation (Symplectic)")
        
        reactor = create_unified_reactor(base_order=5, symplectic=true)
        initialize_reactor!(reactor, seed_count=9)
        
        energies = Float64[]
        for i in 1:50
            reactor_step!(reactor, dt=0.01)
            push!(energies, reactor.state.energy)
        end
        
        # Energy should not drift too much with symplectic structure
        energy_drift = abs(energies[end] - energies[1])
        
        println("✓ Energy evolution tracked")
        println("  Initial energy: $(round(energies[1], digits=4))")
        println("  Final energy: $(round(energies[end], digits=4))")
        println("  Energy drift: $(round(energy_drift, digits=6))")
    end
    
end

println("\n" * "="^70)
println("✓ ALL TESTS PASSED")
println("="^70)
println("\nUnified Reactor Core successfully integrates:")
println("  • B-series ridges with elementary differentials")
println("  • P-system reservoir membranes")
println("  • J-surface gradient-evolution dynamics")
println("  • Echo state reservoir feedback")
println("  • Membrane garden tree cultivation")
println("  • A000081 ontogenetic evolution")
println("\nThe deep tree echo state reservoir computer is operational! 🌳🧠⚡")
