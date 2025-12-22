"""
    UnifiedReactorEvolution

Unified evolution dynamics integrating B-series ridges, P-system reservoirs, and J-surface
elementary differentials with feedback from rooted trees planted in membrane computing gardens.

This module implements the core reactor that unifies gradient descent and evolution dynamics
as the echo state reactor core, following OEIS A000081 ontogenetic principles.

# Mathematical Foundation

The unified dynamics equation:

```
∂ψ/∂t = J(ψ) · ∇H(ψ) + R(ψ, t) + M(ψ)
```

Where:
- J(ψ): J-surface structure matrix (symplectic/Poisson)
- ∇H(ψ): Gradient of Hamiltonian (energy landscape)
- R(ψ, t): Reservoir echo state dynamics
- M(ψ): Membrane evolution rules

# B-Series Ridge Structure

Each computational ridge is a B-series expansion:

```
y_{n+1} = y_n + h Σ_{τ ∈ T} b(τ)/σ(τ) · F(τ)(y_n)
```

Where:
- T: Set of rooted trees from A000081
- b(τ): Coefficients (genetic material)
- σ(τ): Symmetry factor of tree τ
- F(τ): Elementary differential associated with τ (J-surface)

# OEIS A000081 Alignment

All parameters derived from the sequence: {1, 1, 2, 4, 9, 20, 48, 115, 286, 719, ...}
"""
module UnifiedReactorEvolution

using LinearAlgebra
using Random
using Statistics

export UnifiedReactor, ReactorState, ElementaryDifferential
export initialize_reactor!, evolve_reactor!, compute_feedback!
export integrate_bseries_ridge!, update_psystem_reservoir!, flow_jsurface!
export extract_reactor_state, compute_reactor_energy

"""
    ElementaryDifferential

Represents an elementary differential F(τ) associated with a rooted tree τ.
These are the fundamental building blocks of B-series expansions.

# Fields
- `tree::Vector{Int}`: Rooted tree representation (level sequence)
- `order::Int`: Order of the tree (number of nodes)
- `symmetry::Float64`: Symmetry factor σ(τ)
- `coefficient::Float64`: B-series coefficient b(τ)
- `differential_op::Function`: The differential operator F(τ)
"""
struct ElementaryDifferential
    tree::Vector{Int}
    order::Int
    symmetry::Float64
    coefficient::Float64
    differential_op::Function
    
    function ElementaryDifferential(tree::Vector{Int}, coeff::Float64)
        order = length(tree)
        symmetry = compute_tree_symmetry(tree)
        
        # Create differential operator based on tree structure
        diff_op = create_differential_operator(tree)
        
        new(tree, order, symmetry, coeff, diff_op)
    end
end

"""
    ReactorState

Current state of the unified reactor, integrating all components.

# Fields
- `bseries_state::Vector{Float64}`: B-series ridge state
- `reservoir_state::Vector{Float64}`: Echo state reservoir state
- `membrane_states::Dict{Int,Vector{Float64}}`: P-system membrane states
- `jsurface_position::Vector{Float64}`: Position on J-surface manifold
- `jsurface_velocity::Vector{Float64}`: Velocity on J-surface
- `energy::Float64`: Total system energy H(ψ)
- `generation::Int`: Current generation (ontogenetic age)
- `feedback_strength::Float64`: Feedback from membrane gardens
"""
mutable struct ReactorState
    bseries_state::Vector{Float64}
    reservoir_state::Vector{Float64}
    membrane_states::Dict{Int,Vector{Float64}}
    jsurface_position::Vector{Float64}
    jsurface_velocity::Vector{Float64}
    energy::Float64
    generation::Int
    feedback_strength::Float64
    
    function ReactorState(dim::Int, num_membranes::Int)
        new(
            zeros(dim),
            zeros(dim),
            Dict(i => zeros(dim) for i in 1:num_membranes),
            zeros(dim),
            zeros(dim),
            0.0,
            0,
            0.0
        )
    end
end

"""
    UnifiedReactor

The main reactor core that unifies all components under OEIS A000081 ontogenetic control.

# Fields
- `state::ReactorState`: Current reactor state
- `elementary_differentials::Vector{ElementaryDifferential}`: Elementary differentials from rooted trees
- `jsurface_structure::Matrix{Float64}`: J-surface structure matrix J(ψ)
- `hamiltonian::Function`: Energy function H(ψ)
- `reservoir_weights::Matrix{Float64}`: Echo state network weights
- `membrane_rules::Vector{Function}`: P-system evolution rules
- `feedback_matrix::Matrix{Float64}`: Feedback from membrane gardens to reactor
- `a000081_params::NamedTuple`: Parameters derived from A000081
- `symplectic::Bool`: Whether J-surface is symplectic
"""
mutable struct UnifiedReactor
    state::ReactorState
    elementary_differentials::Vector{ElementaryDifferential}
    jsurface_structure::Matrix{Float64}
    hamiltonian::Function
    reservoir_weights::Matrix{Float64}
    membrane_rules::Vector{Function}
    feedback_matrix::Matrix{Float64}
    a000081_params::NamedTuple
    symplectic::Bool
    
    function UnifiedReactor(;
        base_order::Int=5,
        symplectic::Bool=true,
        spectral_radius::Float64=0.95)
        
        # Derive parameters from A000081
        A000081 = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719]
        
        reservoir_size = sum(A000081[1:base_order])  # Cumulative tree count
        num_membranes = A000081[min(base_order, length(A000081))]
        max_tree_order = base_order + 3
        growth_rate = A000081[min(base_order+1, length(A000081))] / A000081[base_order]
        mutation_rate = 1.0 / A000081[base_order]
        
        params = (
            reservoir_size = reservoir_size,
            num_membranes = num_membranes,
            max_tree_order = max_tree_order,
            growth_rate = growth_rate,
            mutation_rate = mutation_rate,
            base_order = base_order
        )
        
        println("🌳 Initializing Unified Reactor with A000081-derived parameters:")
        println("   reservoir_size  = $reservoir_size (cumulative trees 1:$base_order)")
        println("   num_membranes   = $num_membranes (A000081[$base_order])")
        println("   max_tree_order  = $max_tree_order")
        println("   growth_rate     = $(round(growth_rate, digits=4))")
        println("   mutation_rate   = $(round(mutation_rate, digits=4))")
        
        # Initialize state
        state = ReactorState(reservoir_size, num_membranes)
        
        # Generate elementary differentials from rooted trees
        elementary_diffs = generate_elementary_differentials(max_tree_order)
        
        # Create J-surface structure matrix
        jsurface = create_jsurface_structure(reservoir_size, symplectic)
        
        # Define Hamiltonian (energy function)
        hamiltonian = create_hamiltonian(reservoir_size)
        
        # Initialize reservoir weights with echo state property
        reservoir_weights = initialize_reservoir_weights(reservoir_size, spectral_radius)
        
        # Create membrane evolution rules
        membrane_rules = create_membrane_rules(num_membranes)
        
        # Initialize feedback matrix
        feedback_matrix = initialize_feedback_matrix(reservoir_size, num_membranes)
        
        new(state, elementary_diffs, jsurface, hamiltonian,
            reservoir_weights, membrane_rules, feedback_matrix,
            params, symplectic)
    end
end

"""
    generate_elementary_differentials(max_order::Int)

Generate elementary differentials F(τ) for all rooted trees up to given order.
These correspond to the OEIS A000081 sequence.
"""
function generate_elementary_differentials(max_order::Int)
    differentials = ElementaryDifferential[]
    
    # A000081 sequence
    A000081 = [1, 1, 2, 4, 9, 20, 48, 115, 286, 719]
    
    for order in 1:min(max_order, length(A000081))
        # Generate trees of this order
        trees = generate_rooted_trees_of_order(order)
        
        # Create elementary differential for each tree
        for tree in trees
            # Coefficient based on order and symmetry
            coeff = 1.0 / (order * factorial(order))
            diff = ElementaryDifferential(tree, coeff)
            push!(differentials, diff)
        end
    end
    
    return differentials
end

"""
    generate_rooted_trees_of_order(order::Int)

Generate all unlabeled rooted trees with given number of nodes.
Returns level sequence representation.
"""
function generate_rooted_trees_of_order(order::Int)
    if order == 1
        return [[1]]
    elseif order == 2
        return [[1, 2]]
    elseif order == 3
        return [[1, 2, 3], [1, 2, 2]]
    elseif order == 4
        return [
            [1, 2, 3, 4],  # Path
            [1, 2, 3, 3],  # Y-shape
            [1, 2, 2, 3],  # T-shape
            [1, 2, 2, 2]   # Star
        ]
    else
        # For higher orders, use simplified generation
        # In practice, would use proper tree enumeration algorithm
        trees = Vector{Int}[]
        for i in 1:min(order, 10)
            tree = vcat([1], fill(2, order-1))
            push!(trees, tree)
        end
        return trees
    end
end

"""
    compute_tree_symmetry(tree::Vector{Int})

Compute symmetry factor σ(τ) of a rooted tree.
"""
function compute_tree_symmetry(tree::Vector{Int})
    # Count node multiplicities
    level_counts = Dict{Int,Int}()
    for level in tree
        level_counts[level] = get(level_counts, level, 0) + 1
    end
    
    # Symmetry factor is product of factorials
    symmetry = 1.0
    for count in values(level_counts)
        symmetry *= factorial(count)
    end
    
    return symmetry
end

"""
    create_differential_operator(tree::Vector{Int})

Create the differential operator F(τ) for a given rooted tree.
This operator acts on the state space.
"""
function create_differential_operator(tree::Vector{Int})
    order = length(tree)
    
    # Create operator based on tree structure
    function operator(state::Vector{Float64}, f::Function)
        result = copy(state)
        
        # Apply recursive structure of tree
        for (i, level) in enumerate(tree)
            if level > 1
                # Apply function composition based on tree depth
                result = f(result) .* (1.0 / level)
            end
        end
        
        return result
    end
    
    return operator
end

"""
    create_jsurface_structure(dim::Int, symplectic::Bool)

Create J-surface structure matrix J(ψ).
If symplectic, creates a symplectic matrix. Otherwise, creates a Poisson structure.
"""
function create_jsurface_structure(dim::Int, symplectic::Bool)
    if symplectic
        # Create symplectic matrix J = [0 I; -I 0]
        # Ensure even dimension for symplectic structure
        if dim % 2 != 0
            # If odd, use (dim-1) for symplectic part and pad
            half_dim = div(dim - 1, 2)
            J = zeros(dim, dim)
            if half_dim > 0
                J[1:half_dim, half_dim+1:2*half_dim] = Matrix(I, half_dim, half_dim)
                J[half_dim+1:2*half_dim, 1:half_dim] = -Matrix(I, half_dim, half_dim)
            end
            return J
        else
            half_dim = div(dim, 2)
            J = zeros(dim, dim)
            J[1:half_dim, half_dim+1:end] = Matrix(I, half_dim, half_dim)
            J[half_dim+1:end, 1:half_dim] = -Matrix(I, half_dim, half_dim)
            return J
        end
    else
        # Create skew-symmetric Poisson structure
        J = randn(dim, dim)
        J = J - J'  # Make skew-symmetric
        return J ./ norm(J)
    end
end

"""
    create_hamiltonian(dim::Int)

Create Hamiltonian energy function H(ψ).
"""
function create_hamiltonian(dim::Int)
    function H(state::Vector{Float64})
        # Quadratic Hamiltonian: H = 0.5 * state' * state
        return 0.5 * dot(state, state)
    end
    return H
end

"""
    initialize_reservoir_weights(dim::Int, spectral_radius::Float64)

Initialize echo state network weights with specified spectral radius.
"""
function initialize_reservoir_weights(dim::Int, spectral_radius::Float64)
    # Create sparse random matrix
    sparsity = 0.1
    W = randn(dim, dim) .* (rand(dim, dim) .< sparsity)
    
    # Scale to desired spectral radius
    eigenvalues = eigvals(W)
    current_radius = maximum(abs.(eigenvalues))
    W = W .* (spectral_radius / current_radius)
    
    return W
end

"""
    create_membrane_rules(num_membranes::Int)

Create P-system evolution rules for membranes.
"""
function create_membrane_rules(num_membranes::Int)
    rules = Function[]
    
    for i in 1:num_membranes
        # Each membrane has its own evolution rule
        rule = function(state::Vector{Float64}, t::Float64)
            # Simple nonlinear transformation
            return tanh.(state .+ 0.1 * sin(t * i))
        end
        push!(rules, rule)
    end
    
    return rules
end

"""
    initialize_feedback_matrix(reservoir_size::Int, num_membranes::Int)

Initialize feedback matrix from membrane gardens to reactor core.
"""
function initialize_feedback_matrix(reservoir_size::Int, num_membranes::Int)
    # Random feedback connections
    F = randn(reservoir_size, num_membranes) ./ sqrt(num_membranes)
    return F
end

"""
    initialize_reactor!(reactor::UnifiedReactor; seed::Int=42)

Initialize the reactor state with A000081-aligned seed.
"""
function initialize_reactor!(reactor::UnifiedReactor; seed::Int=42)
    Random.seed!(seed)
    
    dim = reactor.a000081_params.reservoir_size
    
    # Initialize B-series state
    reactor.state.bseries_state = randn(dim) .* 0.1
    
    # Initialize reservoir state
    reactor.state.reservoir_state = randn(dim) .* 0.1
    
    # Initialize membrane states
    for (mem_id, _) in reactor.state.membrane_states
        reactor.state.membrane_states[mem_id] = randn(dim) .* 0.1
    end
    
    # Initialize J-surface position and velocity
    reactor.state.jsurface_position = randn(dim) .* 0.1
    reactor.state.jsurface_velocity = zeros(dim)
    
    # Compute initial energy
    reactor.state.energy = reactor.hamiltonian(reactor.state.jsurface_position)
    
    reactor.state.generation = 0
    reactor.state.feedback_strength = 0.0
    
    println("✓ Reactor initialized with seed=$seed")
end

"""
    evolve_reactor!(reactor::UnifiedReactor, dt::Float64, num_steps::Int; verbose::Bool=false)

Evolve the unified reactor through time using the integrated dynamics.

Implements: ∂ψ/∂t = J(ψ) · ∇H(ψ) + R(ψ, t) + M(ψ)
"""
function evolve_reactor!(reactor::UnifiedReactor, dt::Float64, num_steps::Int; verbose::Bool=false)
    for step in 1:num_steps
        t = step * dt
        
        # 1. Integrate B-series ridge (elementary differentials)
        integrate_bseries_ridge!(reactor, dt)
        
        # 2. Update P-system reservoir (membrane evolution)
        update_psystem_reservoir!(reactor, t, dt)
        
        # 3. Flow on J-surface (gradient-evolution dynamics)
        flow_jsurface!(reactor, dt)
        
        # 4. Compute feedback from membrane gardens
        compute_feedback!(reactor)
        
        # 5. Update reservoir with echo state dynamics
        update_reservoir_state!(reactor, dt)
        
        # 6. Update energy
        reactor.state.energy = reactor.hamiltonian(reactor.state.jsurface_position)
        
        reactor.state.generation += 1
        
        if verbose && step % 10 == 0
            println("Step $step: Energy = $(round(reactor.state.energy, digits=6)), " *
                   "Feedback = $(round(reactor.state.feedback_strength, digits=6))")
        end
    end
end

"""
    integrate_bseries_ridge!(reactor::UnifiedReactor, dt::Float64)

Integrate B-series ridge using elementary differentials.

Implements: y_{n+1} = y_n + h Σ_{τ ∈ T} b(τ)/σ(τ) · F(τ)(y_n)
"""
function integrate_bseries_ridge!(reactor::UnifiedReactor, dt::Float64)
    state = reactor.state.bseries_state
    
    # B-series integration
    increment = zeros(length(state))
    
    for diff in reactor.elementary_differentials
        # Compute elementary differential contribution
        # F(τ)(state) weighted by b(τ)/σ(τ)
        weight = diff.coefficient / diff.symmetry
        
        # Simple vector field for demonstration
        f(x) = -0.1 * x .+ 0.01 * sin.(x)
        
        contribution = diff.differential_op(state, f)
        increment .+= weight .* contribution
    end
    
    # Update state
    reactor.state.bseries_state .+= dt .* increment
end

"""
    update_psystem_reservoir!(reactor::UnifiedReactor, t::Float64, dt::Float64)

Update P-system membrane reservoir using evolution rules.

Implements the M(ψ) term in the unified dynamics.
"""
function update_psystem_reservoir!(reactor::UnifiedReactor, t::Float64, dt::Float64)
    for (mem_id, rule) in enumerate(reactor.membrane_rules)
        if haskey(reactor.state.membrane_states, mem_id)
            current_state = reactor.state.membrane_states[mem_id]
            
            # Apply membrane evolution rule
            new_state = rule(current_state, t)
            
            # Update with time step
            reactor.state.membrane_states[mem_id] .+= dt .* (new_state .- current_state)
        end
    end
end

"""
    flow_jsurface!(reactor::UnifiedReactor, dt::Float64)

Flow on J-surface manifold using Hamiltonian dynamics.

Implements: ∂ψ/∂t = J(ψ) · ∇H(ψ)
"""
function flow_jsurface!(reactor::UnifiedReactor, dt::Float64)
    position = reactor.state.jsurface_position
    
    # Compute gradient of Hamiltonian
    grad_H = position  # For quadratic Hamiltonian, ∇H = position
    
    # J-surface flow: velocity = J · ∇H
    velocity = reactor.jsurface_structure * grad_H
    
    # Update position
    reactor.state.jsurface_position .+= dt .* velocity
    reactor.state.jsurface_velocity .= velocity
end

"""
    compute_feedback!(reactor::UnifiedReactor)

Compute feedback from membrane gardens to reactor core.

This implements the coupling between rooted trees planted in membranes
and the reactor core dynamics.
"""
function compute_feedback!(reactor::UnifiedReactor)
    # Aggregate membrane states
    membrane_vector = Float64[]
    for i in 1:reactor.a000081_params.num_membranes
        if haskey(reactor.state.membrane_states, i)
            push!(membrane_vector, mean(reactor.state.membrane_states[i]))
        end
    end
    
    # Compute feedback strength
    if !isempty(membrane_vector)
        feedback = reactor.feedback_matrix * membrane_vector
        reactor.state.feedback_strength = norm(feedback)
        
        # Apply feedback to B-series state
        reactor.state.bseries_state .+= 0.01 .* feedback
    end
end

"""
    update_reservoir_state!(reactor::UnifiedReactor, dt::Float64)

Update echo state reservoir with feedback.

Implements: R(ψ, t) term in unified dynamics.
"""
function update_reservoir_state!(reactor::UnifiedReactor, dt::Float64)
    # Echo state update: x(t+1) = (1-α)x(t) + α·tanh(W·x(t) + input)
    α = 0.3
    
    # Input from B-series and J-surface
    input = 0.5 * reactor.state.bseries_state .+ 0.5 * reactor.state.jsurface_position
    
    # Reservoir dynamics
    new_state = (1 - α) .* reactor.state.reservoir_state .+ 
                α .* tanh.(reactor.reservoir_weights * reactor.state.reservoir_state .+ input)
    
    reactor.state.reservoir_state .= new_state
end

"""
    extract_reactor_state(reactor::UnifiedReactor)

Extract complete reactor state for analysis.
"""
function extract_reactor_state(reactor::UnifiedReactor)
    return (
        bseries = copy(reactor.state.bseries_state),
        reservoir = copy(reactor.state.reservoir_state),
        membranes = copy(reactor.state.membrane_states),
        jsurface_pos = copy(reactor.state.jsurface_position),
        jsurface_vel = copy(reactor.state.jsurface_velocity),
        energy = reactor.state.energy,
        generation = reactor.state.generation,
        feedback = reactor.state.feedback_strength
    )
end

"""
    compute_reactor_energy(reactor::UnifiedReactor)

Compute total reactor energy across all components.
"""
function compute_reactor_energy(reactor::UnifiedReactor)
    # Hamiltonian energy
    H_energy = reactor.state.energy
    
    # Reservoir energy
    R_energy = 0.5 * dot(reactor.state.reservoir_state, reactor.state.reservoir_state)
    
    # Membrane energies
    M_energy = 0.0
    for (_, mem_state) in reactor.state.membrane_states
        M_energy += 0.5 * dot(mem_state, mem_state)
    end
    
    return H_energy + R_energy + M_energy
end

end # module UnifiedReactorEvolution
