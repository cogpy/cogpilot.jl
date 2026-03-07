// cog/pilot/pilot.hpp — Deep Tree Echo State Reservoir Computer
// A000081 rooted trees, B-Series, J-Surface, P-System membranes
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
#ifndef COG_PILOT_HPP
#define COG_PILOT_HPP

#include "../core/core.hpp"
#include <cstdint>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

namespace cog { namespace pilot {

// ─────────────────────────────────────────────────────────────────────────────
// A000081 — Number of rooted trees with n nodes (OEIS A000081)
// Generates the sequence: 1, 1, 2, 4, 9, 20, 48, 115, 286, 719, ...
// All system parameters derive from this sequence.
// ─────────────────────────────────────────────────────────────────────────────
class A000081 {
public:
    explicit A000081(size_t max_n = 64) : t_(max_n + 1, 0) {
        compute(max_n);
    }

    // Get t(n) = number of rooted trees with n nodes
    uint64_t operator()(size_t n) const {
        return (n < t_.size()) ? t_[n] : 0;
    }

    // Derive a parameter in [lo, hi] from t(n)
    double param(size_t n, double lo, double hi) const {
        if (n == 0 || n >= t_.size()) return lo;
        double frac = static_cast<double>(t_[n] % 1000) / 1000.0;
        return lo + frac * (hi - lo);
    }

    // Spectral radius from A000081
    double spectral_radius(size_t n) const {
        return param(n, 0.8, 0.999);
    }

    // Reservoir size from A000081
    size_t reservoir_size(size_t n) const {
        return static_cast<size_t>((*this)(n));
    }

    size_t max_n() const { return t_.size() - 1; }

private:
    std::vector<uint64_t> t_;

    void compute(size_t max_n) {
        // Recurrence: t(0)=0, t(1)=1
        // t(n) = (1/(n-1)) * sum_{k=1}^{n-1} ( sum_{d|k} d*t(d) ) * t(n-k)
        if (max_n < 1) return;
        t_[0] = 0;
        t_[1] = 1;
        // Precompute s(k) = sum_{d|k} d * t(d)
        std::vector<uint64_t> s(max_n + 1, 0);
        for (size_t n = 2; n <= max_n; ++n) {
            // Update s for k = n-1
            for (size_t k = 1; k < n; ++k) {
                s[k] = 0;
                for (size_t d = 1; d <= k; ++d) {
                    if (k % d == 0) {
                        s[k] += d * t_[d];
                    }
                }
            }
            uint64_t sum = 0;
            for (size_t k = 1; k < n; ++k) {
                sum += s[k] * t_[n - k];
            }
            t_[n] = sum / (n - 1);
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RootedTree — Explicit tree structure for B-Series computation
// ─────────────────────────────────────────────────────────────────────────────
struct RootedTree {
    size_t id;
    std::vector<size_t> children; // Indices into a tree collection

    RootedTree() : id(0) {}
    explicit RootedTree(size_t id) : id(id) {}

    size_t order() const {
        // Order = 1 + sum of children orders (computed externally)
        return 1; // Leaf
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// BSeriesNode — Butcher tree node for numerical ODE methods
// Elementary differentials F(t) for a tree t
// ─────────────────────────────────────────────────────────────────────────────
struct BSeriesNode {
    size_t order;       // Number of nodes in the tree
    double sigma;       // Symmetry coefficient
    double gamma;       // Density (order!)
    double alpha;       // Weight coefficient a(t) = 1/(sigma*gamma)

    BSeriesNode() : order(1), sigma(1.0), gamma(1.0), alpha(1.0) {}

    // Elementary weight for explicit Euler
    static double euler_weight(size_t order) {
        return (order == 1) ? 1.0 : 0.0;
    }

    // Elementary weight for implicit midpoint
    static double midpoint_weight(size_t order) {
        return 1.0 / std::pow(2.0, static_cast<double>(order));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Ridge — Ridge structure on J-Surface
// ─────────────────────────────────────────────────────────────────────────────
struct RidgePoint {
    double x, y, z;
    double curvature;
    double potential;

    RidgePoint() : x(0), y(0), z(0), curvature(0), potential(0) {}
    RidgePoint(double x, double y, double z)
        : x(x), y(y), z(z), curvature(0), potential(0) {}

    double norm() const { return std::sqrt(x*x + y*y + z*z); }

    RidgePoint operator+(const RidgePoint& o) const {
        return RidgePoint(x+o.x, y+o.y, z+o.z);
    }
    RidgePoint operator*(double s) const {
        return RidgePoint(x*s, y*s, z*s);
    }
};

struct Ridge {
    std::vector<RidgePoint> points;
    double total_curvature;

    Ridge() : total_curvature(0) {}

    void compute_curvatures() {
        total_curvature = 0;
        for (size_t i = 1; i + 1 < points.size(); ++i) {
            // Discrete curvature: angle between consecutive segments
            double dx1 = points[i].x - points[i-1].x;
            double dy1 = points[i].y - points[i-1].y;
            double dz1 = points[i].z - points[i-1].z;
            double dx2 = points[i+1].x - points[i].x;
            double dy2 = points[i+1].y - points[i].y;
            double dz2 = points[i+1].z - points[i].z;
            double dot = dx1*dx2 + dy1*dy2 + dz1*dz2;
            double n1 = std::sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);
            double n2 = std::sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
            double cos_angle = (n1 > 1e-12 && n2 > 1e-12) ? dot / (n1 * n2) : 1.0;
            cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
            points[i].curvature = std::acos(cos_angle);
            total_curvature += points[i].curvature;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// JSurface — J-Surface reactor for unified dynamics
// Elementary differentials on the surface drive reservoir evolution
// ─────────────────────────────────────────────────────────────────────────────
class JSurface {
public:
    struct State {
        std::vector<double> coords;  // Surface coordinates
        double energy;
        double temperature;

        State() : energy(0), temperature(1.0) {}
    };

    explicit JSurface(size_t dim = 3) : dim_(dim) {}

    // Compute potential at a point
    double potential(const std::vector<double>& x) const {
        assert(x.size() == dim_);
        // Morse-like potential with multiple wells
        double V = 0;
        for (size_t i = 0; i < dim_; ++i) {
            double r = x[i];
            V += (1.0 - std::exp(-r * r)) * (1.0 - std::exp(-r * r));
        }
        // Cross-coupling terms
        for (size_t i = 0; i + 1 < dim_; ++i) {
            V += 0.1 * x[i] * x[i+1];
        }
        return V;
    }

    // Gradient of potential
    std::vector<double> gradient(const std::vector<double>& x) const {
        std::vector<double> grad(dim_);
        double eps = 1e-6;
        for (size_t i = 0; i < dim_; ++i) {
            std::vector<double> xp = x, xm = x;
            xp[i] += eps;
            xm[i] -= eps;
            grad[i] = (potential(xp) - potential(xm)) / (2.0 * eps);
        }
        return grad;
    }

    // Evolve state using gradient descent with noise (Langevin dynamics)
    void step(State& state, double dt, double noise_scale, std::mt19937& rng) const {
        auto grad = gradient(state.coords);
        std::normal_distribution<double> noise(0.0, 1.0);
        for (size_t i = 0; i < dim_; ++i) {
            double thermal = noise_scale * std::sqrt(2.0 * state.temperature * dt) * noise(rng);
            state.coords[i] -= dt * grad[i] + thermal;
        }
        state.energy = potential(state.coords);
    }

    // Find ridges: points where gradient is perpendicular to one eigenvector
    Ridge find_ridge(const std::vector<double>& start, size_t steps, double dt) const {
        Ridge ridge;
        std::vector<double> pos = start;
        for (size_t s = 0; s < steps; ++s) {
            auto grad = gradient(pos);
            double gnorm = 0;
            for (auto g : grad) gnorm += g * g;
            gnorm = std::sqrt(gnorm + 1e-12);
            // Follow the ridge: move perpendicular to steepest descent
            // Use the component with smallest gradient magnitude
            size_t min_idx = 0;
            double min_val = std::fabs(grad[0]);
            for (size_t i = 1; i < dim_; ++i) {
                if (std::fabs(grad[i]) < min_val) {
                    min_val = std::fabs(grad[i]);
                    min_idx = i;
                }
            }
            // Step along the ridge direction
            for (size_t i = 0; i < dim_; ++i) {
                if (i == min_idx) {
                    pos[i] += dt * (grad[i] > 0 ? 1.0 : -1.0);
                } else {
                    pos[i] -= dt * grad[i] / gnorm;
                }
            }
            RidgePoint rp;
            if (dim_ >= 1) rp.x = pos[0];
            if (dim_ >= 2) rp.y = pos[1];
            if (dim_ >= 3) rp.z = pos[2];
            rp.potential = potential(pos);
            ridge.points.push_back(rp);
        }
        ridge.compute_curvatures();
        return ridge;
    }

    size_t dim() const { return dim_; }

private:
    size_t dim_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Membrane — P-System membrane for nested computation
// ─────────────────────────────────────────────────────────────────────────────
struct Membrane {
    size_t id;
    size_t parent_id;                // 0 = skin membrane
    std::vector<size_t> children;
    std::vector<double> multiset;    // Objects (as real-valued concentrations)
    std::vector<std::pair<std::vector<double>, std::vector<double>>> rules;
    // rule: (reactants, products) — applied when reactants are available
    bool dissolved;

    Membrane() : id(0), parent_id(0), dissolved(false) {}

    // Apply rules: consume reactants, produce products
    bool apply_rules() {
        bool applied = false;
        for (auto& rule : rules) {
            auto& reactants = rule.first;
            auto& products = rule.second;
            // Check if reactants are available
            bool can_apply = true;
            for (size_t i = 0; i < reactants.size() && i < multiset.size(); ++i) {
                if (multiset[i] < reactants[i]) { can_apply = false; break; }
            }
            if (can_apply) {
                for (size_t i = 0; i < reactants.size() && i < multiset.size(); ++i) {
                    multiset[i] -= reactants[i];
                }
                // Ensure multiset is large enough for products
                if (products.size() > multiset.size()) multiset.resize(products.size(), 0);
                for (size_t i = 0; i < products.size(); ++i) {
                    multiset[i] += products[i];
                }
                applied = true;
            }
        }
        return applied;
    }
};

class MembraneGarden {
public:
    MembraneGarden() : next_id_(1) {
        // Create skin membrane
        Membrane skin;
        skin.id = next_id_++;
        skin.parent_id = 0;
        membranes_[skin.id] = skin;
        skin_id_ = skin.id;
    }

    size_t skin_id() const { return skin_id_; }

    // Add a child membrane
    size_t add_membrane(size_t parent_id, size_t multiset_size = 8) {
        Membrane m;
        m.id = next_id_++;
        m.parent_id = parent_id;
        m.multiset.resize(multiset_size, 0);
        membranes_[m.id] = m;
        membranes_[parent_id].children.push_back(m.id);
        return m.id;
    }

    // Get membrane
    Membrane* get(size_t id) {
        auto it = membranes_.find(id);
        return (it != membranes_.end()) ? &it->second : nullptr;
    }

    const Membrane* get(size_t id) const {
        auto it = membranes_.find(id);
        return (it != membranes_.end()) ? &it->second : nullptr;
    }

    // Step: apply rules in all membranes (maximally parallel)
    void step() {
        // Collect dissolved membranes
        std::vector<size_t> to_dissolve;
        for (auto& kv : membranes_) {
            kv.second.apply_rules();
            if (kv.second.dissolved && kv.first != skin_id_) {
                to_dissolve.push_back(kv.first);
            }
        }
        // Dissolve: move contents to parent
        for (auto id : to_dissolve) {
            auto* m = get(id);
            if (!m) continue;
            auto* parent = get(m->parent_id);
            if (!parent) continue;
            // Move multiset to parent
            if (parent->multiset.size() < m->multiset.size())
                parent->multiset.resize(m->multiset.size(), 0);
            for (size_t i = 0; i < m->multiset.size(); ++i) {
                parent->multiset[i] += m->multiset[i];
            }
            // Move children to parent
            for (auto cid : m->children) {
                membranes_[cid].parent_id = m->parent_id;
                parent->children.push_back(cid);
            }
            // Remove from parent's children list
            auto& pc = parent->children;
            pc.erase(std::remove(pc.begin(), pc.end(), id), pc.end());
            membranes_.erase(id);
        }
    }

    size_t size() const { return membranes_.size(); }

    template<typename Fn>
    void foreach_membrane(Fn fn) const {
        for (auto& kv : membranes_) fn(kv.second);
    }

private:
    size_t next_id_;
    size_t skin_id_;
    std::unordered_map<size_t, Membrane> membranes_;
};

// ─────────────────────────────────────────────────────────────────────────────
// EchoStateNetwork — Reservoir computing with A000081-derived parameters
// ─────────────────────────────────────────────────────────────────────────────
class EchoStateNetwork {
public:
    EchoStateNetwork(size_t input_dim, size_t reservoir_dim, size_t output_dim,
                     double spectral_radius = 0.95, uint32_t seed = 42)
        : N_in_(input_dim), N_res_(reservoir_dim), N_out_(output_dim),
          rho_(spectral_radius), rng_(seed)
    {
        // Initialize reservoir weights
        W_in_.resize(N_res_ * N_in_);
        W_res_.resize(N_res_ * N_res_);
        W_out_.resize(N_out_ * (N_res_ + N_in_));
        state_.resize(N_res_, 0.0);

        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        // Input weights: sparse random
        for (auto& w : W_in_) w = dist(rng_) * 0.1;

        // Reservoir weights: sparse random, then scale to spectral radius
        double sparsity = 0.1;
        std::uniform_real_distribution<double> prob(0.0, 1.0);
        for (auto& w : W_res_) {
            w = (prob(rng_) < sparsity) ? dist(rng_) : 0.0;
        }
        scale_to_spectral_radius();

        // Output weights: zero (to be trained)
        std::fill(W_out_.begin(), W_out_.end(), 0.0);
    }

    // Advance reservoir state given input
    std::vector<double> step(const std::vector<double>& input) {
        assert(input.size() == N_in_);
        std::vector<double> new_state(N_res_);

        for (size_t i = 0; i < N_res_; ++i) {
            double sum = 0;
            // Input contribution
            for (size_t j = 0; j < N_in_; ++j) {
                sum += W_in_[i * N_in_ + j] * input[j];
            }
            // Reservoir recurrence
            for (size_t j = 0; j < N_res_; ++j) {
                sum += W_res_[i * N_res_ + j] * state_[j];
            }
            new_state[i] = std::tanh(sum);
        }

        state_ = new_state;
        return readout(input);
    }

    // Compute output from current state
    std::vector<double> readout(const std::vector<double>& input) const {
        std::vector<double> output(N_out_);
        // Concatenate [state, input]
        size_t ext_dim = N_res_ + N_in_;
        for (size_t i = 0; i < N_out_; ++i) {
            double sum = 0;
            for (size_t j = 0; j < N_res_; ++j) {
                sum += W_out_[i * ext_dim + j] * state_[j];
            }
            for (size_t j = 0; j < N_in_; ++j) {
                sum += W_out_[i * ext_dim + N_res_ + j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    // Train output weights using ridge regression (Tikhonov)
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               double ridge_param = 1e-6) {
        size_t T = inputs.size();
        size_t ext_dim = N_res_ + N_in_;

        // Collect states
        std::vector<std::vector<double>> states;
        reset_state();
        for (size_t t = 0; t < T; ++t) {
            step_internal(inputs[t]);
            // Extended state = [state, input]
            std::vector<double> ext(ext_dim);
            std::copy(state_.begin(), state_.end(), ext.begin());
            std::copy(inputs[t].begin(), inputs[t].end(), ext.begin() + N_res_);
            states.push_back(ext);
        }

        // Ridge regression: W_out = Y * X^T * (X * X^T + lambda*I)^{-1}
        // Simplified: solve normal equations via Cholesky-like approach
        // X^T X + lambda*I
        std::vector<double> XtX(ext_dim * ext_dim, 0);
        std::vector<double> XtY(ext_dim * N_out_, 0);

        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < ext_dim; ++i) {
                for (size_t j = 0; j < ext_dim; ++j) {
                    XtX[i * ext_dim + j] += states[t][i] * states[t][j];
                }
                for (size_t k = 0; k < N_out_; ++k) {
                    XtY[i * N_out_ + k] += states[t][i] * targets[t][k];
                }
            }
        }

        // Add ridge
        for (size_t i = 0; i < ext_dim; ++i) {
            XtX[i * ext_dim + i] += ridge_param;
        }

        // Solve via Gauss elimination
        std::vector<double> A(ext_dim * (ext_dim + N_out_));
        for (size_t i = 0; i < ext_dim; ++i) {
            for (size_t j = 0; j < ext_dim; ++j) {
                A[i * (ext_dim + N_out_) + j] = XtX[i * ext_dim + j];
            }
            for (size_t k = 0; k < N_out_; ++k) {
                A[i * (ext_dim + N_out_) + ext_dim + k] = XtY[i * N_out_ + k];
            }
        }

        // Forward elimination
        for (size_t col = 0; col < ext_dim; ++col) {
            // Partial pivoting
            size_t max_row = col;
            double max_val = std::fabs(A[col * (ext_dim + N_out_) + col]);
            for (size_t row = col + 1; row < ext_dim; ++row) {
                double val = std::fabs(A[row * (ext_dim + N_out_) + col]);
                if (val > max_val) { max_val = val; max_row = row; }
            }
            if (max_row != col) {
                for (size_t j = 0; j < ext_dim + N_out_; ++j) {
                    std::swap(A[col * (ext_dim + N_out_) + j],
                              A[max_row * (ext_dim + N_out_) + j]);
                }
            }
            double pivot = A[col * (ext_dim + N_out_) + col];
            if (std::fabs(pivot) < 1e-12) continue;
            for (size_t row = col + 1; row < ext_dim; ++row) {
                double factor = A[row * (ext_dim + N_out_) + col] / pivot;
                for (size_t j = col; j < ext_dim + N_out_; ++j) {
                    A[row * (ext_dim + N_out_) + j] -= factor * A[col * (ext_dim + N_out_) + j];
                }
            }
        }

        // Back substitution
        std::vector<double> solution(ext_dim * N_out_, 0);
        for (int i = static_cast<int>(ext_dim) - 1; i >= 0; --i) {
            double pivot = A[i * (ext_dim + N_out_) + i];
            if (std::fabs(pivot) < 1e-12) continue;
            for (size_t k = 0; k < N_out_; ++k) {
                double sum = A[i * (ext_dim + N_out_) + ext_dim + k];
                for (size_t j = i + 1; j < ext_dim; ++j) {
                    sum -= A[i * (ext_dim + N_out_) + j] * solution[j * N_out_ + k];
                }
                solution[i * N_out_ + k] = sum / pivot;
            }
        }

        // Store as W_out (output_dim x ext_dim) — transpose of solution
        W_out_.resize(N_out_ * ext_dim);
        for (size_t i = 0; i < N_out_; ++i) {
            for (size_t j = 0; j < ext_dim; ++j) {
                W_out_[i * ext_dim + j] = solution[j * N_out_ + i];
            }
        }
    }

    void reset_state() { std::fill(state_.begin(), state_.end(), 0.0); }

    const std::vector<double>& state() const { return state_; }
    size_t reservoir_dim() const { return N_res_; }

private:
    size_t N_in_, N_res_, N_out_;
    double rho_;
    std::mt19937 rng_;
    std::vector<double> W_in_, W_res_, W_out_;
    std::vector<double> state_;

    void step_internal(const std::vector<double>& input) {
        std::vector<double> new_state(N_res_);
        for (size_t i = 0; i < N_res_; ++i) {
            double sum = 0;
            for (size_t j = 0; j < N_in_; ++j)
                sum += W_in_[i * N_in_ + j] * input[j];
            for (size_t j = 0; j < N_res_; ++j)
                sum += W_res_[i * N_res_ + j] * state_[j];
            new_state[i] = std::tanh(sum);
        }
        state_ = new_state;
    }

    void scale_to_spectral_radius() {
        // Power iteration to estimate spectral radius
        std::vector<double> v(N_res_, 1.0);
        double norm = std::sqrt(static_cast<double>(N_res_));
        for (auto& x : v) x /= norm;

        double lambda = 1.0;
        for (int iter = 0; iter < 100; ++iter) {
            std::vector<double> w(N_res_, 0);
            for (size_t i = 0; i < N_res_; ++i) {
                for (size_t j = 0; j < N_res_; ++j) {
                    w[i] += W_res_[i * N_res_ + j] * v[j];
                }
            }
            lambda = 0;
            for (auto x : w) lambda += x * x;
            lambda = std::sqrt(lambda);
            if (lambda > 1e-12) {
                for (auto& x : w) x /= lambda;
            }
            v = w;
        }

        // Scale
        if (lambda > 1e-12) {
            double scale = rho_ / lambda;
            for (auto& w : W_res_) w *= scale;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DeepTreeEchoSystem — Unified system combining all components
// ─────────────────────────────────────────────────────────────────────────────
class DeepTreeEchoSystem {
public:
    struct Config {
        size_t tree_depth;      // A000081 parameter index
        size_t input_dim;
        size_t output_dim;
        size_t surface_dim;
        size_t num_membranes;
        uint32_t seed;

        Config() : tree_depth(8), input_dim(3), output_dim(1),
                   surface_dim(3), num_membranes(4), seed(42) {}
    };

    explicit DeepTreeEchoSystem(const Config& cfg = Config())
        : cfg_(cfg), a81_(cfg.tree_depth + 2),
          surface_(cfg.surface_dim),
          rng_(cfg.seed)
    {
        // Derive reservoir size and spectral radius from A000081
        size_t res_dim = std::max<size_t>(16, a81_.reservoir_size(cfg.tree_depth));
        double rho = a81_.spectral_radius(cfg.tree_depth);

        esn_ = std::unique_ptr<EchoStateNetwork>(
            new EchoStateNetwork(cfg.input_dim, res_dim, cfg.output_dim, rho, cfg.seed));

        // Initialize membranes
        for (size_t i = 0; i < cfg.num_membranes; ++i) {
            garden_.add_membrane(garden_.skin_id(), cfg.input_dim);
        }

        // Initialize surface state
        surface_state_.coords.resize(cfg.surface_dim, 0.1);
        surface_state_.temperature = a81_.param(cfg.tree_depth, 0.01, 1.0);
    }

    // Process one timestep
    std::vector<double> step(const std::vector<double>& input) {
        // 1. Feed input to membranes
        auto* skin = garden_.get(garden_.skin_id());
        if (skin && skin->multiset.size() >= input.size()) {
            for (size_t i = 0; i < input.size(); ++i) {
                skin->multiset[i] = input[i];
            }
        }
        garden_.step();

        // 2. Evolve J-Surface
        double dt = a81_.param(cfg_.tree_depth, 0.001, 0.01);
        double noise = a81_.param(cfg_.tree_depth, 0.001, 0.1);
        surface_.step(surface_state_, dt, noise, rng_);

        // 3. Feed to reservoir (input + surface energy as augmented input)
        std::vector<double> aug_input = input;
        // Pad or truncate to match ESN input dim
        aug_input.resize(cfg_.input_dim, surface_state_.energy);

        return esn_->step(aug_input);
    }

    // Train the system
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               double ridge = 1e-6) {
        esn_->train(inputs, targets, ridge);
    }

    const A000081& a81() const { return a81_; }
    const EchoStateNetwork& esn() const { return *esn_; }
    const JSurface& surface() const { return surface_; }
    const MembraneGarden& garden() const { return garden_; }

private:
    Config cfg_;
    A000081 a81_;
    std::unique_ptr<EchoStateNetwork> esn_;
    JSurface surface_;
    JSurface::State surface_state_;
    MembraneGarden garden_;
    std::mt19937 rng_;
};

}} // namespace cog::pilot

#endif // COG_PILOT_HPP
