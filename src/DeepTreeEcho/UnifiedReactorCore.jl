"""
    UnifiedReactorCore

The complete unification of B-series ridges, P-system reservoirs, and J-surface
elementary differentials into a cohesive echo state reactor core. This module
implements the feedback from rooted trees planted in membrane computing gardens
to unify the ontogenetic engine under OEIS A000081.

# Mathematical Foundation

The unified reactor evolves according to:

    ∂ψ/∂t = J(ψ) · ∇H_A000081(ψ) + Σ_{τ∈T} b(τ)/σ(τ) · F(τ)(ψ) + R_echo(ψ) + M_membrane(ψ)

Where:
- **J(ψ)**: J-surface structure matrix (symplectic/Poisson)
- **H_A000081(ψ)**: Hamiltonian encoding A000081 tree complexity
- **b(τ)/σ(τ)**: B-series ridge coefficients with symmetry factors
- **F(τ)**: Elementary differentials indexed by rooted trees τ
- **R_echo(ψ)**: Echo state reservoir dynamics
- **M_membrane(ψ)**: P-system membrane evolution rules

# The Feedback Loop

```
    Rooted Trees (A000081)
           ↓
    Elementary Differentials F(τ)
           ↓
    B-Series Ridge Coefficients b(τ)
           ↓
    J-Surface Gradient Flow
           ↓
    Echo State Reservoir
           ↓
    P-System Membranes
           ↓
    Membrane Garden Feedback
           ↓
    New Trees → Planted → Evolution
           ↓
    [LOOP BACK TO TOP]
```

# Ontogenetic Evolution

The system self-evolves through:
1. **Tree Generation**: A000081 generates rooted tree structures
2. **Differential Mapping**: Trees → Elementary differentials F(τ)
3. **Ridge Formation**: Differentials → B-series coefficients b(τ)
4. **Gradient Flow**: J-surface integrates continuous dynamics
5. **Echo Processing**: Reservoir learns temporal patterns
6. **Membrane Evolution**: P-systems evolve tree populations
7. **Garden Feedback**: Successful trees replanted, mutated, crossed
8. **Ontogenetic Loop**: New generation of trees emerges

# Usage

```julia
using DeepTreeEcho.UnifiedReactorCore

# Create unified reactor with A000081-aligned parameters
reactor = create_unified_reactor(base_order=5)

# Initialize with seed trees
initialize_reactor!(reactor, seed_count=9)  # A000081[5] = 9

# Evolve the complete system
for generation in 1:50
    # Reactor step: gradient + ridge + echo + membrane
    reactor_step!(reactor, dt=0.01)
    
    # Harvest feedback from membranes
    harvest_feedback!(reactor)
    
    # Plant new trees in garden
    plant_evolved_trees!(reactor)
    
    # Adapt topology based on fitness
    adapt_reactor_topology!(reactor)
end

# Get reactor status
status = get_reactor_status(reactor)
println("Generation: ", status.generation)
println("Energy: ", status.energy)
println("Tree diversity: ", status.tree_diversity)
```
"""
module UnifiedReactorCore

using LinearAlgebra
using Random
using Statistics

# Note: Sibling modules should be loaded at package level
# This module provides standalone functionality

export UnifiedReactor
export create_unified_reactor, initialize_reactor!
export reactor_step!, harvest_feedback!, plant_evolved_trees!
export adapt_reactor_topology!, get_reactor_status
export ReactorState, ReactorConfig

# The sacred A000081 sequence
const A000081 = [
    1, 1, 2, 4, 9, 20, 48, 115, 286, 719,
    1842, 4766, 12486, 32973, 87811, 235381
]

"""
    get_parameter_set(base_order::Int; membrane_order::Int=4)

Derive A000081-aligned parameters for the reactor.
"""
function get_parameter_set(base_order::Int; membrane_order::Int=4)
    reservoir_size = sum(A000081[1:base_order])
    max_tree_order = base_order + 3
    num_membranes = A000081[membrane_order]
    growth_rate = A000081[base_order+1] / A000081[base_order]
    mutation_rate = 1.0 / A000081[base_order]
    
    return (
        reservoir_size = reservoir_size,
        max_tree_order = max_tree_order,
        num_membranes = num_membranes,
        growth_rate = growth_rate,
        mutation_rate = mutation_rate,
        base_order = base_order
    )
end

"""
    ReactorState

Complete state of the unified reactor system.

# Fields
- `ψ::Vector{Float64}`: Main state vector
- `reservoir_state::Vector{Float64}`: Echo state reservoir
- `membrane_states::Dict{Int, Vector{Float64}}`: P-system membrane states
- `planted_trees::Dict{Int, Vector{Vector{Int}}}`: Trees in each membrane
- `tree_fitness::Dict{Vector{Int}, Float64}`: Fitness of each tree
- `energy::Float64`: Current system energy H(ψ)
- `gradient::Vector{Float64}`: Current gradient ∇H(ψ)
- `generation::Int`: Current generation
- `step_count::Int`: Total steps
"""
mutable struct ReactorState
    ψ::Vector{Float64}
    reservoir_state::Vector{Float64}
    membrane_states::Dict{Int, Vector{Float64}}
    planted_trees::Dict{Int, Vector{Vector{Int}}}
    tree_fitness::Dict{Vector{Int}, Float64}
    energy::Float64
    gradient::Vector{Float64}
    generation::Int
    step_count::Int
end

"""
    ReactorConfig

Configuration for the unified reactor.

# Fields
- `reservoir_size::Int`: Size of reservoir (from A000081)
- `max_tree_order::Int`: Maximum tree order
- `num_membranes::Int`: Number of P-system membranes (from A000081)
- `growth_rate::Float64`: Tree growth rate (from A000081 ratio)
- `mutation_rate::Float64`: Mutation rate (from A000081 inverse)
- `crossover_rate::Float64`: Crossover rate for tree breeding
- `symplectic::Bool`: Use symplectic J-surface structure
- `base_order::Int`: Base order for A000081 derivation
"""
struct ReactorConfig
    reservoir_size::Int
    max_tree_order::Int
    num_membranes::Int
    growth_rate::Float64
    mutation_rate::Float64
    crossover_rate::Float64
    symplectic::Bool
    base_order::Int
end

"""
    UnifiedReactor

The complete unified reactor integrating all DTE-RC components.

# Fields
- `config::ReactorConfig`: Reactor configuration
- `state::ReactorState`: Current reactor state
- `rooted_trees::Vector{Vector{Int}}`: All trees from A000081
- `jsurface_matrix::Matrix{Float64}`: J-surface structure matrix J(ψ)
- `hamiltonian::Function`: Energy function H_A000081(ψ)
- `ridge_coefficients::Dict{Vector{Int}, Float64}`: B-series coefficients b(τ)
- `symmetry_factors::Dict{Vector{Int}, Int}`: Tree symmetry factors σ(τ)
- `elementary_differentials::Dict{Vector{Int}, Function}`: Differential map F(τ)
- `reservoir_weights::Matrix{Float64}`: Echo state reservoir weights
- `membrane_rules::Dict{Int, Function}`: P-system evolution rules
- `garden_feedback::Vector{Vector{Int}}`: Trees from garden feedback
- `energy_history::Vector{Float64}`: Energy trajectory
- `diversity_history::Vector{Float64}`: Tree diversity over time
"""
mutable struct UnifiedReactor
    config::ReactorConfig
    state::ReactorState
    rooted_trees::Vector{Vector{Int}}
    jsurface_matrix::Matrix{Float64}
    hamiltonian::Function
    ridge_coefficients::Dict{Vector{Int}, Float64}
    symmetry_factors::Dict{Vector{Int}, Int}
    elementary_differentials::Dict{Vector{Int}, Function}
    reservoir_weights::Matrix{Float64}
    membrane_rules::Dict{Int, Function}
    garden_feedback::Vector{Vector{Int}}
    energy_history::Vector{Float64}
    diversity_history::Vector{Float64}
end

"""
    create_unified_reactor(; base_order::Int=5, kwargs...)

Create a unified reactor with A000081-aligned parameters.

# Arguments
- `base_order::Int=5`: Base order for A000081 parameter derivation
- `symplectic::Bool=true`: Use symplectic J-surface structure
- `crossover_rate::Float64=0.7`: Crossover rate for tree breeding

# Returns
- `UnifiedReactor`: Initialized reactor system

# Example
```julia
reactor = create_unified_reactor(base_order=5)
```
"""
function create_unified_reactor(;
    base_order::Int=5,
    symplectic::Bool=true,
    crossover_rate::Float64=0.7)
    
    println("\n🌳 Creating Unified Reactor Core (A000081-aligned)")
    println("=" ^ 60)
    
    # Derive all parameters from A000081
    params = get_parameter_set(base_order, membrane_order=4)
    
    config = ReactorConfig(
        params.reservoir_size,
        params.max_tree_order,
        params.num_membranes,
        params.growth_rate,
        params.mutation_rate,
        crossover_rate,
        symplectic,
        base_order
    )
    
    println("✓ Configuration:")
    println("  reservoir_size  = $(config.reservoir_size) (Σ A000081[1:$base_order])")
    println("  max_tree_order  = $(config.max_tree_order)")
    println("  num_membranes   = $(config.num_membranes) (A000081[4])")
    println("  growth_rate     = $(round(config.growth_rate, digits=4))")
    println("  mutation_rate   = $(round(config.mutation_rate, digits=4))")
    println("  crossover_rate  = $(config.crossover_rate)")
    println("  symplectic      = $(config.symplectic)")
    
    # Generate rooted trees from A000081
    rooted_trees = generate_all_trees(config.max_tree_order)
    println("\n✓ Generated $(length(rooted_trees)) rooted trees")
    
    # Create J-surface structure matrix
    jsurface_matrix = create_jsurface_structure(config.reservoir_size, symplectic)
    println("✓ Created J-surface structure ($(size(jsurface_matrix)))")
    
    # Create A000081-based Hamiltonian
    hamiltonian = create_a000081_hamiltonian(rooted_trees)
    println("✓ Created A000081 Hamiltonian")
    
    # Initialize B-series ridge coefficients
    ridge_coefficients = initialize_ridge_coefficients(rooted_trees)
    println("✓ Initialized B-series ridge ($(length(ridge_coefficients)) coefficients)")
    
    # Compute symmetry factors
    symmetry_factors = compute_all_symmetry_factors(rooted_trees)
    println("✓ Computed symmetry factors")
    
    # Create elementary differentials
    elementary_differentials = create_all_elementary_differentials(rooted_trees, config.reservoir_size)
    println("✓ Created elementary differentials F(τ)")
    
    # Initialize echo state reservoir
    reservoir_weights = initialize_reservoir_weights(config.reservoir_size, rooted_trees)
    println("✓ Initialized echo state reservoir")
    
    # Create P-system membrane rules
    membrane_rules = create_membrane_rules(config.num_membranes)
    println("✓ Created P-system membrane rules")
    
    # Initialize reactor state
    state = ReactorState(
        randn(config.reservoir_size),  # ψ
        randn(config.reservoir_size),  # reservoir_state
        Dict{Int, Vector{Float64}}(),  # membrane_states
        Dict{Int, Vector{Vector{Int}}}(),  # planted_trees
        Dict{Vector{Int}, Float64}(),  # tree_fitness
        0.0,  # energy
        zeros(config.reservoir_size),  # gradient
        0,  # generation
        0   # step_count
    )
    
    # Initialize membrane states
    for i in 1:config.num_membranes
        state.membrane_states[i] = randn(config.reservoir_size)
        state.planted_trees[i] = Vector{Int}[]
    end
    
    println("\n✓ Unified Reactor Core created successfully!")
    println("=" ^ 60)
    
    UnifiedReactor(
        config,
        state,
        rooted_trees,
        jsurface_matrix,
        hamiltonian,
        ridge_coefficients,
        symmetry_factors,
        elementary_differentials,
        reservoir_weights,
        membrane_rules,
        Vector{Int}[],
        Float64[],
        Float64[]
    )
end

"""
    initialize_reactor!(reactor::UnifiedReactor; seed_count::Int=9)

Initialize the reactor with seed trees from A000081.

# Arguments
- `reactor::UnifiedReactor`: The reactor to initialize
- `seed_count::Int=9`: Number of seed trees (default: A000081[5] = 9)
"""
function initialize_reactor!(reactor::UnifiedReactor; seed_count::Int=9)
    println("\n🌱 Initializing Reactor with $seed_count seed trees")
    
    # Select diverse seed trees from different orders
    seed_trees = Vector{Int}[]
    for order in 1:min(5, reactor.config.max_tree_order)
        trees_of_order = filter(t -> tree_order(t) == order, reactor.rooted_trees)
        n_take = min(2, length(trees_of_order))
        if n_take > 0
            append!(seed_trees, trees_of_order[1:n_take])
        end
    end
    
    # Fill to seed_count
    while length(seed_trees) < seed_count && length(seed_trees) < length(reactor.rooted_trees)
        tree = rand(reactor.rooted_trees)
        if !(tree in seed_trees)
            push!(seed_trees, tree)
        end
    end
    
    # Plant trees in membranes (distribute evenly)
    for (i, tree) in enumerate(seed_trees)
        membrane_id = ((i - 1) % reactor.config.num_membranes) + 1
        push!(reactor.state.planted_trees[membrane_id], tree)
        reactor.state.tree_fitness[tree] = rand()  # Initial random fitness
    end
    
    # Update state based on planted trees
    reactor.state.energy = reactor.hamiltonian(reactor.state.ψ)
    reactor.state.gradient = compute_gradient(reactor.hamiltonian, reactor.state.ψ)
    
    println("✓ Planted $(length(seed_trees)) trees across $(reactor.config.num_membranes) membranes")
    for membrane_id in 1:reactor.config.num_membranes
        n_trees = length(reactor.state.planted_trees[membrane_id])
        println("  Membrane $membrane_id: $n_trees trees")
    end
end

"""
    reactor_step!(reactor::UnifiedReactor; dt::Float64=0.01)

Execute one complete reactor evolution step.

This integrates:
1. J-surface gradient flow: J(ψ)·∇H(ψ)
2. B-series ridge: Σ b(τ)/σ(τ)·F(τ)(ψ)
3. Echo state reservoir: R_echo(ψ)
4. P-system membranes: M_membrane(ψ)

# Arguments
- `reactor::UnifiedReactor`: The reactor system
- `dt::Float64=0.01`: Time step
"""
function reactor_step!(reactor::UnifiedReactor; dt::Float64=0.01)
    state = reactor.state
    
    # 1. Compute J-surface gradient flow
    gradient_flow = reactor.jsurface_matrix * state.gradient
    
    # 2. Compute B-series ridge contribution
    ridge_flow = zeros(reactor.config.reservoir_size)
    for tree in keys(reactor.ridge_coefficients)
        b_tau = reactor.ridge_coefficients[tree]
        sigma_tau = reactor.symmetry_factors[tree]
        F_tau = reactor.elementary_differentials[tree]
        
        # Add contribution: b(τ)/σ(τ) · F(τ)(ψ)
        ridge_flow .+= (b_tau / sigma_tau) * F_tau(state.ψ)
    end
    
    # 3. Compute echo state reservoir contribution
    echo_flow = tanh.(reactor.reservoir_weights * state.reservoir_state)
    
    # 4. Compute P-system membrane contribution
    membrane_flow = zeros(reactor.config.reservoir_size)
    for membrane_id in 1:reactor.config.num_membranes
        rule = reactor.membrane_rules[membrane_id]
        membrane_contribution = rule(state.membrane_states[membrane_id], state.planted_trees[membrane_id])
        membrane_flow .+= membrane_contribution
    end
    
    # 5. Unified evolution equation
    dψ_dt = gradient_flow + ridge_flow + echo_flow + membrane_flow
    
    # 6. Update state
    state.ψ .+= dt * dψ_dt
    state.reservoir_state .+= dt * echo_flow
    
    # Update membrane states
    for membrane_id in 1:reactor.config.num_membranes
        state.membrane_states[membrane_id] .+= dt * membrane_flow / reactor.config.num_membranes
    end
    
    # Update energy and gradient
    state.energy = reactor.hamiltonian(state.ψ)
    state.gradient = compute_gradient(reactor.hamiltonian, state.ψ)
    
    # Increment step count
    state.step_count += 1
    
    # Record history
    push!(reactor.energy_history, state.energy)
    push!(reactor.diversity_history, compute_tree_diversity(state.planted_trees))
end

"""
    harvest_feedback!(reactor::UnifiedReactor)

Harvest feedback from membrane gardens to generate new trees.

This evaluates tree fitness and selects successful trees for propagation.
"""
function harvest_feedback!(reactor::UnifiedReactor)
    state = reactor.state
    
    # Evaluate fitness for all planted trees
    for membrane_id in 1:reactor.config.num_membranes
        for tree in state.planted_trees[membrane_id]
            # Fitness based on: energy reduction, stability, membrane state
            membrane_state = state.membrane_states[membrane_id]
            fitness = evaluate_tree_fitness(tree, state.ψ, membrane_state, state.energy)
            state.tree_fitness[tree] = fitness
        end
    end
    
    # Select top trees for feedback
    all_trees_with_fitness = collect(state.tree_fitness)
    sort!(all_trees_with_fitness, by=x->x[2], rev=true)
    
    # Top 20% go to garden feedback
    n_feedback = max(1, length(all_trees_with_fitness) ÷ 5)
    reactor.garden_feedback = [tree for (tree, fitness) in all_trees_with_fitness[1:n_feedback]]
end

"""
    plant_evolved_trees!(reactor::UnifiedReactor)

Plant evolved trees back into membranes through mutation and crossover.
"""
function plant_evolved_trees!(reactor::UnifiedReactor)
    state = reactor.state
    config = reactor.config
    
    new_trees = Vector{Int}[]
    
    # Mutation: mutate feedback trees
    for tree in reactor.garden_feedback
        if rand() < config.mutation_rate
            mutated = mutate_tree(tree, reactor.rooted_trees)
            push!(new_trees, mutated)
        end
    end
    
    # Crossover: breed pairs of feedback trees
    for i in 1:2:length(reactor.garden_feedback)-1
        if rand() < config.crossover_rate
            parent1 = reactor.garden_feedback[i]
            parent2 = reactor.garden_feedback[i+1]
            offspring = crossover_trees(parent1, parent2, reactor.rooted_trees)
            push!(new_trees, offspring)
        end
    end
    
    # Plant new trees in membranes
    for tree in new_trees
        membrane_id = rand(1:config.num_membranes)
        push!(state.planted_trees[membrane_id], tree)
        state.tree_fitness[tree] = 0.0  # Will be evaluated next cycle
    end
    
    # Increment generation
    state.generation += 1
end

"""
    adapt_reactor_topology!(reactor::UnifiedReactor)

Adapt reactor topology based on evolutionary feedback.

This adjusts B-series coefficients and reservoir weights based on tree fitness.
"""
function adapt_reactor_topology!(reactor::UnifiedReactor)
    # Adjust ridge coefficients based on tree fitness
    for (tree, fitness) in reactor.state.tree_fitness
        if haskey(reactor.ridge_coefficients, tree)
            # Increase coefficient for high-fitness trees
            reactor.ridge_coefficients[tree] *= (1.0 + 0.1 * fitness)
        end
    end
    
    # Normalize ridge coefficients
    total = sum(values(reactor.ridge_coefficients))
    for tree in keys(reactor.ridge_coefficients)
        reactor.ridge_coefficients[tree] /= total
    end
    
    # Adapt reservoir weights (small random perturbation)
    reactor.reservoir_weights .+= 0.01 * randn(size(reactor.reservoir_weights))
end

"""
    get_reactor_status(reactor::UnifiedReactor)

Get comprehensive status of the reactor system.
"""
function get_reactor_status(reactor::UnifiedReactor)
    state = reactor.state
    
    total_trees = sum(length(trees) for trees in values(state.planted_trees))
    avg_fitness = isempty(state.tree_fitness) ? 0.0 : mean(values(state.tree_fitness))
    
    (
        generation = state.generation,
        step_count = state.step_count,
        energy = state.energy,
        total_trees = total_trees,
        avg_fitness = avg_fitness,
        tree_diversity = compute_tree_diversity(state.planted_trees),
        feedback_trees = length(reactor.garden_feedback)
    )
end

# ============================================================================
# Helper Functions
# ============================================================================

function generate_all_trees(max_order::Int)
    trees = Vector{Int}[]
    
    # Order 1: single node
    push!(trees, [1])
    
    # Order 2: two nodes
    push!(trees, [1, 2])
    
    # Order 3: three nodes (2 trees)
    push!(trees, [1, 2, 3])
    push!(trees, [1, 2, 2])
    
    # Order 4: four nodes (4 trees)
    push!(trees, [1, 2, 3, 4])
    push!(trees, [1, 2, 3, 3])
    push!(trees, [1, 2, 2, 3])
    push!(trees, [1, 2, 2, 2])
    
    # For higher orders, use simple generation
    for order in 5:max_order
        # Generate a few representative trees
        for _ in 1:min(A000081[order], 10)
            tree = [rand(1:i) for i in 1:order]
            push!(trees, tree)
        end
    end
    
    trees
end

function create_jsurface_structure(dim::Int, symplectic::Bool)
    if symplectic && iseven(dim)
        half = dim ÷ 2
        J = zeros(dim, dim)
        J[1:half, (half+1):end] = I(half)
        J[(half+1):end, 1:half] = -I(half)
        return J
    else
        # Poisson structure
        J = randn(dim, dim)
        return (J - J') * 0.1
    end
end

function create_a000081_hamiltonian(trees::Vector{Vector{Int}})
    function H(ψ::Vector{Float64})
        # Hamiltonian encoding tree complexity
        energy = 0.5 * dot(ψ, ψ)  # Quadratic base
        
        # Add tree-complexity terms
        for (i, tree) in enumerate(trees)
            if i <= length(ψ)
                complexity = tree_order(tree)
                energy += 0.01 * complexity * ψ[i]^2
            end
        end
        
        return energy
    end
    return H
end

function initialize_ridge_coefficients(trees::Vector{Vector{Int}})
    coeffs = Dict{Vector{Int}, Float64}()
    for tree in trees
        # Initialize with classical RK4-like coefficients
        order = tree_order(tree)
        coeffs[tree] = 1.0 / factorial(order)
    end
    return coeffs
end

function compute_all_symmetry_factors(trees::Vector{Vector{Int}})
    factors = Dict{Vector{Int}, Int}()
    for tree in trees
        factors[tree] = compute_symmetry_factor(tree)
    end
    return factors
end

function compute_symmetry_factor(tree::Vector{Int})
    # Simple symmetry computation
    n = length(tree)
    if n <= 2
        return 1
    else
        # Count automorphisms (simplified)
        return n
    end
end

function create_all_elementary_differentials(trees::Vector{Vector{Int}}, dim::Int)
    diffs = Dict{Vector{Int}, Function}()
    
    for tree in trees
        order = tree_order(tree)
        
        # Create elementary differential for this tree
        F_tau = function(ψ::Vector{Float64})
            result = zeros(dim)
            
            if order == 1
                # F(•)(ψ) = f(ψ) = -ψ (simple dynamics)
                result = -ψ
            elseif order == 2
                # F(τ)(ψ) = f'(ψ)·f(ψ)
                result = -(-ψ)  # Derivative of -ψ times -ψ
            else
                # Higher order: recursive application
                result = (-1.0)^order * ψ / order
            end
            
            return result
        end
        
        diffs[tree] = F_tau
    end
    
    return diffs
end

function initialize_reservoir_weights(dim::Int, trees::Vector{Vector{Int}})
    W = randn(dim, dim) * 0.1
    
    # Add tree-structured connections
    for tree in trees
        for i in 1:length(tree)-1
            if tree[i] <= dim && tree[i+1] <= dim
                W[tree[i], tree[i+1]] += 0.05
            end
        end
    end
    
    # Scale to spectral radius < 1
    eigenvalues = eigvals(W)
    max_eigenvalue = maximum(abs.(eigenvalues))
    if max_eigenvalue > 0
        W = W * 0.9 / max_eigenvalue
    end
    
    return W
end

function create_membrane_rules(num_membranes::Int)
    rules = Dict{Int, Function}()
    
    for i in 1:num_membranes
        # Each membrane has its own evolution rule
        rules[i] = function(membrane_state::Vector{Float64}, planted_trees::Vector{Vector{Int}})
            contribution = zeros(length(membrane_state))
            
            # Contribution based on planted trees
            for tree in planted_trees
                order = tree_order(tree)
                # Trees influence membrane state
                contribution .+= 0.01 * order * tanh.(membrane_state)
            end
            
            return contribution
        end
    end
    
    return rules
end

function tree_order(tree::Vector{Int})
    return length(tree)
end

function compute_gradient(H::Function, ψ::Vector{Float64})
    # Numerical gradient
    ε = 1e-7
    grad = zeros(length(ψ))
    
    for i in 1:length(ψ)
        ψ_plus = copy(ψ)
        ψ_plus[i] += ε
        
        ψ_minus = copy(ψ)
        ψ_minus[i] -= ε
        
        grad[i] = (H(ψ_plus) - H(ψ_minus)) / (2ε)
    end
    
    return grad
end

function evaluate_tree_fitness(tree::Vector{Int}, ψ::Vector{Float64}, 
                               membrane_state::Vector{Float64}, energy::Float64)
    # Fitness based on multiple criteria
    order = tree_order(tree)
    
    # 1. Complexity bonus (higher order = more complex)
    complexity_score = order / 10.0
    
    # 2. Energy efficiency (lower energy = better)
    energy_score = 1.0 / (1.0 + abs(energy))
    
    # 3. Membrane stability
    stability_score = 1.0 / (1.0 + norm(membrane_state))
    
    # Combined fitness
    fitness = 0.4 * complexity_score + 0.3 * energy_score + 0.3 * stability_score
    
    return fitness
end

function compute_tree_diversity(planted_trees::Dict{Int, Vector{Vector{Int}}})
    all_trees = Vector{Int}[]
    for trees in values(planted_trees)
        append!(all_trees, trees)
    end
    
    if isempty(all_trees)
        return 0.0
    end
    
    # Diversity = number of unique trees / total trees
    unique_trees = unique(all_trees)
    return length(unique_trees) / length(all_trees)
end

function mutate_tree(tree::Vector{Int}, available_trees::Vector{Vector{Int}})
    # Simple mutation: randomly change one node
    mutated = copy(tree)
    if !isempty(mutated)
        idx = rand(1:length(mutated))
        mutated[idx] = rand(1:length(mutated))
    end
    
    # Ensure it's a valid tree from available set
    if mutated in available_trees
        return mutated
    else
        return tree  # Return original if mutation invalid
    end
end

function crossover_trees(parent1::Vector{Int}, parent2::Vector{Int}, 
                        available_trees::Vector{Vector{Int}})
    # Single-point crossover
    if isempty(parent1) || isempty(parent2)
        return rand([parent1, parent2])
    end
    
    min_len = min(length(parent1), length(parent2))
    if min_len <= 1
        return rand([parent1, parent2])
    end
    
    point = rand(1:min_len-1)
    offspring = vcat(parent1[1:point], parent2[point+1:end])
    
    # Validate
    if offspring in available_trees
        return offspring
    else
        return parent1  # Return parent1 if offspring invalid
    end
end

end # module UnifiedReactorCore
