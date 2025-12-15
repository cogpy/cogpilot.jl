"""
Test suite for the Unified Reactor Core

Tests the complete integration of:
- B-series ridges
- P-system reservoirs  
- J-surface elementary differentials
- Echo state dynamics
- Membrane garden feedback
- A000081 ontogenetic evolution
"""

using Test
using LinearAlgebra
using Statistics

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "DeepTreeEcho"))

# Load modules
include("../src/DeepTreeEcho/UnifiedReactorCore.jl")
using .UnifiedReactorCore

include("../src/DeepTreeEcho/EnhancedPackageIntegration.jl")
using .EnhancedPackageIntegration

@testset "Unified Reactor Core Tests" begin
    
    @testset "Reactor Creation" begin
        println("\n" * "="^60)
        println("TEST: Reactor Creation")
        println("="^60)
        
        # Test reactor creation with A000081-aligned parameters
        reactor = create_unified_reactor(base_order=5)
        
        @test reactor isa UnifiedReactor
        @test reactor.config.reservoir_size == 17  # Σ A000081[1:5] = 1+1+2+4+9
        @test reactor.config.num_membranes == 4    # A000081[4] = 4
        @test reactor.config.max_tree_order == 8
        @test reactor.config.symplectic == true
        
        # Check components initialized
        @test !isempty(reactor.rooted_trees)
        @test size(reactor.jsurface_matrix) == (17, 17)
        @test !isempty(reactor.ridge_coefficients)
        @test !isempty(reactor.symmetry_factors)
        @test !isempty(reactor.elementary_differentials)
        @test size(reactor.reservoir_weights) == (17, 17)
        @test length(reactor.membrane_rules) == 4
        
        println("✓ Reactor created successfully")
        println("  Reservoir size: $(reactor.config.reservoir_size)")
        println("  Membranes: $(reactor.config.num_membranes)")
        println("  Trees: $(length(reactor.rooted_trees))")
    end
    
    @testset "Reactor Initialization" begin
        println("\n" * "="^60)
        println("TEST: Reactor Initialization")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        
        # Initialize with A000081[5] = 9 seed trees
        initialize_reactor!(reactor, seed_count=9)
        
        # Check planted trees
        total_planted = sum(length(trees) for trees in values(reactor.state.planted_trees))
        @test total_planted == 9
        
        # Check all membranes have trees
        for membrane_id in 1:reactor.config.num_membranes
            @test haskey(reactor.state.planted_trees, membrane_id)
        end
        
        # Check fitness initialized
        @test !isempty(reactor.state.tree_fitness)
        
        # Check energy computed
        @test reactor.state.energy != 0.0
        @test length(reactor.state.gradient) == reactor.config.reservoir_size
        
        println("✓ Reactor initialized successfully")
        println("  Planted trees: $total_planted")
        println("  Initial energy: $(round(reactor.state.energy, digits=4))")
    end
    
    @testset "Reactor Evolution Step" begin
        println("\n" * "="^60)
        println("TEST: Reactor Evolution Step")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        initial_energy = reactor.state.energy
        initial_step = reactor.state.step_count
        
        # Execute one reactor step
        reactor_step!(reactor, dt=0.01)
        
        # Check state updated
        @test reactor.state.step_count == initial_step + 1
        @test length(reactor.energy_history) == 1
        @test length(reactor.diversity_history) == 1
        
        # Check energy evolved
        @test reactor.state.energy != initial_energy
        
        println("✓ Reactor step executed successfully")
        println("  Initial energy: $(round(initial_energy, digits=4))")
        println("  Final energy: $(round(reactor.state.energy, digits=4))")
        println("  Energy change: $(round(reactor.state.energy - initial_energy, digits=6))")
    end
    
    @testset "Multiple Evolution Steps" begin
        println("\n" * "="^60)
        println("TEST: Multiple Evolution Steps")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        n_steps = 10
        
        for i in 1:n_steps
            reactor_step!(reactor, dt=0.01)
        end
        
        @test reactor.state.step_count == n_steps
        @test length(reactor.energy_history) == n_steps
        @test length(reactor.diversity_history) == n_steps
        
        println("✓ Multiple steps executed successfully")
        println("  Steps: $n_steps")
        println("  Final energy: $(round(reactor.state.energy, digits=4))")
        println("  Energy trajectory: $(round.(reactor.energy_history[1:min(5, n_steps)], digits=4))")
    end
    
    @testset "Feedback Harvesting" begin
        println("\n" * "="^60)
        println("TEST: Feedback Harvesting")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        # Evolve a bit
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        
        # Harvest feedback
        harvest_feedback!(reactor)
        
        # Check feedback trees selected
        @test !isempty(reactor.garden_feedback)
        @test length(reactor.garden_feedback) <= length(reactor.state.tree_fitness)
        
        # Check all feedback trees have fitness
        for tree in reactor.garden_feedback
            @test haskey(reactor.state.tree_fitness, tree)
        end
        
        println("✓ Feedback harvested successfully")
        println("  Feedback trees: $(length(reactor.garden_feedback))")
        println("  Total trees: $(sum(length(trees) for trees in values(reactor.state.planted_trees)))")
    end
    
    @testset "Tree Planting and Evolution" begin
        println("\n" * "="^60)
        println("TEST: Tree Planting and Evolution")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        initial_generation = reactor.state.generation
        initial_tree_count = sum(length(trees) for trees in values(reactor.state.planted_trees))
        
        # Evolve and harvest
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        harvest_feedback!(reactor)
        
        # Plant evolved trees
        plant_evolved_trees!(reactor)
        
        # Check generation incremented
        @test reactor.state.generation == initial_generation + 1
        
        # Check new trees may have been planted
        final_tree_count = sum(length(trees) for trees in values(reactor.state.planted_trees))
        @test final_tree_count >= initial_tree_count
        
        println("✓ Tree evolution cycle completed")
        println("  Generation: $(reactor.state.generation)")
        println("  Initial trees: $initial_tree_count")
        println("  Final trees: $final_tree_count")
    end
    
    @testset "Topology Adaptation" begin
        println("\n" * "="^60)
        println("TEST: Topology Adaptation")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        # Store initial coefficients
        initial_coeffs = copy(reactor.ridge_coefficients)
        
        # Evolve and adapt
        for i in 1:5
            reactor_step!(reactor, dt=0.01)
        end
        harvest_feedback!(reactor)
        adapt_reactor_topology!(reactor)
        
        # Check coefficients changed
        coeffs_changed = false
        for tree in keys(initial_coeffs)
            if haskey(reactor.ridge_coefficients, tree)
                if reactor.ridge_coefficients[tree] != initial_coeffs[tree]
                    coeffs_changed = true
                    break
                end
            end
        end
        
        @test coeffs_changed
        
        # Check coefficients normalized
        total = sum(values(reactor.ridge_coefficients))
        @test isapprox(total, 1.0, atol=1e-10)
        
        println("✓ Topology adapted successfully")
        println("  Coefficients changed: $coeffs_changed")
        println("  Coefficient sum: $(round(total, digits=10))")
    end
    
    @testset "Complete Evolution Cycle" begin
        println("\n" * "="^60)
        println("TEST: Complete Evolution Cycle")
        println("="^60)
        
        reactor = create_unified_reactor(base_order=5)
        initialize_reactor!(reactor, seed_count=9)
        
        n_generations = 5
        
        for gen in 1:n_generations
            # Reactor steps
            for step in 1:10
                reactor_step!(reactor, dt=0.01)
            end
            
            # Harvest feedback
            harvest_feedback!(reactor)
            
            # Plant evolved trees
            plant_evolved_trees!(reactor)
            
            # Adapt topology
            adapt_reactor_topology!(reactor)
        end
        
        @test reactor.state.generation == n_generations
        @test reactor.state.step_count == n_generations * 10
        
        # Get status
        status = get_reactor_status(reactor)
        
        @test status.generation == n_generations
        @test status.step_count == n_generations * 10
        @test status.total_trees > 0
        @test status.tree_diversity >= 0.0
        @test status.tree_diversity <= 1.0
        
        println("✓ Complete evolution cycle successful")
        println("  Generations: $(status.generation)")
        println("  Steps: $(status.step_count)")
        println("  Total trees: $(status.total_trees)")
        println("  Tree diversity: $(round(status.tree_diversity, digits=4))")
        println("  Avg fitness: $(round(status.avg_fitness, digits=4))")
        println("  Final energy: $(round(status.energy, digits=4))")
    end
    
    @testset "A000081 Parameter Alignment" begin
        println("\n" * "="^60)
        println("TEST: A000081 Parameter Alignment")
        println("="^60)
        
        # Test different base orders
        for base_order in [3, 4, 5, 6]
            reactor = create_unified_reactor(base_order=base_order)
            
            # Check reservoir size is cumulative sum
            A000081 = [1, 1, 2, 4, 9, 20, 48, 115]
            expected_reservoir_size = sum(A000081[1:base_order])
            @test reactor.config.reservoir_size == expected_reservoir_size
            
            # Check num_membranes from A000081
            @test reactor.config.num_membranes == 4  # A000081[4]
            
            println("  Base order $base_order: reservoir_size=$(reactor.config.reservoir_size) (expected=$expected_reservoir_size)")
        end
        
        println("✓ A000081 alignment verified")
    end
    
    @testset "J-Surface Structure" begin
        println("\n" * "="^60)
        println("TEST: J-Surface Structure")
        println("="^60)
        
        # Test symplectic structure
        reactor_symplectic = create_unified_reactor(base_order=5, symplectic=true)
        J = reactor_symplectic.jsurface_matrix
        
        # Check skew-symmetric
        @test isapprox(J, -J', atol=1e-10)
        
        # For symplectic, check block structure
        if iseven(size(J, 1))
            half = size(J, 1) ÷ 2
            @test isapprox(J[1:half, 1:half], zeros(half, half), atol=1e-10)
            @test isapprox(J[(half+1):end, (half+1):end], zeros(half, half), atol=1e-10)
        end
        
        println("✓ J-surface structure verified")
        println("  Symplectic: $(reactor_symplectic.config.symplectic)")
        println("  Skew-symmetric: $(isapprox(J, -J', atol=1e-10))")
    end
    
end

@testset "SciML Package Integration Tests" begin
    
    @testset "SciML Reactor Creation" begin
        println("\n" * "="^60)
        println("TEST: SciML Reactor Creation")
        println("="^60)
        
        sciml_reactor = create_sciml_integrated_reactor(base_order=5)
        
        @test sciml_reactor isa SciMLIntegratedReactor
        @test !isnothing(sciml_reactor.base_reactor)
        @test !isempty(sciml_reactor.package_status)
        
        println("✓ SciML reactor created")
        println("  Packages available:")
        for (pkg, status) in sciml_reactor.package_status
            symbol = status ? "✓" : "✗"
            println("    $symbol $pkg")
        end
    end
    
    @testset "Rooted Trees Generation" begin
        println("\n" * "="^60)
        println("TEST: Rooted Trees Generation")
        println("="^60)
        
        sciml_reactor = create_sciml_integrated_reactor(base_order=5)
        
        trees = generate_rooted_trees_sciml(sciml_reactor, max_order=6)
        
        @test !isempty(trees) || !sciml_reactor.package_status["RootedTrees"]
        
        println("✓ Rooted trees generation tested")
        println("  Trees generated: $(length(trees))")
    end
    
    @testset "Full Integration Demo" begin
        println("\n" * "="^60)
        println("TEST: Full Integration Demo")
        println("="^60)
        
        # Run full demo
        demo_reactor = create_full_sciml_integration_demo()
        
        @test demo_reactor isa SciMLIntegratedReactor
        
        println("✓ Full integration demo completed")
    end
    
end

println("\n" * "="^70)
println("ALL TESTS COMPLETED SUCCESSFULLY")
println("="^70)
