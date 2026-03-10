// cog/pilot/lorenz_reservoir.hpp — Lorenz-Reservoir Coupling for Expression Dynamics
// Couples the Lorenz attractor chaotic dynamics with ESN reservoir computing
// to generate temporally coherent micro-expression trajectories.
// Header-only, C++11, zero external dependencies
// SPDX-License-Identifier: MIT
//
// The Lorenz attractor drives the reservoir input, and the reservoir's
// echo state provides temporal smoothing and pattern memory for expressions.
// This prevents abrupt AU transitions while maintaining chaotic liveliness.
//
#ifndef COG_PILOT_LORENZ_RESERVOIR_HPP
#define COG_PILOT_LORENZ_RESERVOIR_HPP

#include "../core/core.hpp"
#include "pilot.hpp"
#include <cmath>
#include <cstdint>
#include <array>
#include <vector>
#include <random>

namespace cog { namespace pilot {

// ─────────────────────────────────────────────────────────────────────────────
// LorenzReservoirCoupling — ESN driven by Lorenz attractor for expression
// ─────────────────────────────────────────────────────────────────────────────
class LorenzReservoirCoupling {
public:
    static constexpr size_t INPUT_DIM  = 3;  // Lorenz x, y, z
    static constexpr size_t OUTPUT_DIM = 5;  // AU modulation channels

    // Construct with reservoir size derived from A000081 sequence
    explicit LorenzReservoirCoupling(size_t a000081_n = 6, uint32_t seed = 42)
        : a81_(), rng_(seed)
    {
        // Derive reservoir size from A000081(n)
        res_size_ = a81_.reservoir_size(a000081_n);
        if (res_size_ < 8) res_size_ = 8;
        if (res_size_ > 512) res_size_ = 512;

        // Derive spectral radius from A000081
        spectral_radius_ = static_cast<float>(a81_.spectral_radius(a000081_n));

        // Initialize reservoir state
        state_.resize(res_size_, 0.0f);

        // Initialize input weights Win [res_size x INPUT_DIM]
        win_.resize(res_size_ * INPUT_DIM);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& w : win_) w = dist(rng_);

        // Initialize reservoir weights W [res_size x res_size] (sparse)
        w_.resize(res_size_ * res_size_, 0.0f);
        float sparsity = 0.1f;
        std::uniform_real_distribution<float> prob(0.0f, 1.0f);
        for (size_t i = 0; i < res_size_ * res_size_; ++i) {
            if (prob(rng_) < sparsity) {
                w_[i] = dist(rng_);
            }
        }
        // Scale to spectral radius (approximate)
        float max_abs = 0.0f;
        for (auto v : w_) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs > 1e-6f) {
            float scale = spectral_radius_ / max_abs;
            for (auto& v : w_) v *= scale;
        }

        // Initialize output weights Wout [OUTPUT_DIM x res_size] (identity-like)
        wout_.resize(OUTPUT_DIM * res_size_, 0.0f);
        for (size_t o = 0; o < OUTPUT_DIM; ++o) {
            size_t idx = o * res_size_ + (o % res_size_);
            wout_[idx] = 0.1f;
        }

        // Lorenz attractor
        lx_ = ly_ = lz_ = 1.0f;
        sigma_ = 10.0f; rho_ = 28.0f; beta_ = 8.0f / 3.0f;
        dt_ = 0.01f;
    }

    // Step the coupled system, return OUTPUT_DIM modulation values in [-1, 1]
    std::array<float, OUTPUT_DIM> step() {
        // 1. Step Lorenz attractor
        lorenz_rk4();
        float input[INPUT_DIM] = {
            lx_ / 20.0f,
            ly_ / 30.0f,
            (lz_ - 25.0f) / 25.0f
        };

        // 2. Reservoir update: state = tanh(Win * input + W * state)
        std::vector<float> new_state(res_size_, 0.0f);
        for (size_t i = 0; i < res_size_; ++i) {
            float sum = 0.0f;
            // Win * input
            for (size_t j = 0; j < INPUT_DIM; ++j) {
                sum += win_[i * INPUT_DIM + j] * input[j];
            }
            // W * state
            for (size_t j = 0; j < res_size_; ++j) {
                sum += w_[i * res_size_ + j] * state_[j];
            }
            new_state[i] = std::tanh(sum);
        }
        state_ = new_state;

        // 3. Output: Wout * state
        std::array<float, OUTPUT_DIM> output;
        for (size_t o = 0; o < OUTPUT_DIM; ++o) {
            float sum = 0.0f;
            for (size_t j = 0; j < res_size_; ++j) {
                sum += wout_[o * res_size_ + j] * state_[j];
            }
            output[o] = std::tanh(sum); // Bound to [-1, 1]
        }
        return output;
    }

    // Get reservoir state for introspection
    const std::vector<float>& reservoir_state() const { return state_; }
    size_t reservoir_size() const { return res_size_; }
    float spectral_radius() const { return spectral_radius_; }

    // Lorenz state accessors
    float lorenz_x() const { return lx_; }
    float lorenz_y() const { return ly_; }
    float lorenz_z() const { return lz_; }

private:
    A000081 a81_;
    std::mt19937 rng_;
    size_t res_size_;
    float spectral_radius_;

    // Reservoir weights and state
    std::vector<float> win_;   // Input weights
    std::vector<float> w_;     // Reservoir weights
    std::vector<float> wout_;  // Output weights
    std::vector<float> state_; // Reservoir state

    // Lorenz attractor state
    float lx_, ly_, lz_;
    float sigma_, rho_, beta_, dt_;

    void lorenz_derivatives(float ax, float ay, float az,
                            float& dx, float& dy, float& dz) const {
        dx = sigma_ * (ay - ax);
        dy = ax * (rho_ - az) - ay;
        dz = ax * ay - beta_ * az;
    }

    void lorenz_rk4() {
        float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
        lorenz_derivatives(lx_, ly_, lz_, k1x, k1y, k1z);
        lorenz_derivatives(lx_+0.5f*dt_*k1x, ly_+0.5f*dt_*k1y, lz_+0.5f*dt_*k1z, k2x, k2y, k2z);
        lorenz_derivatives(lx_+0.5f*dt_*k2x, ly_+0.5f*dt_*k2y, lz_+0.5f*dt_*k2z, k3x, k3y, k3z);
        lorenz_derivatives(lx_+dt_*k3x, ly_+dt_*k3y, lz_+dt_*k3z, k4x, k4y, k4z);
        lx_ += (dt_/6.0f) * (k1x + 2*k2x + 2*k3x + k4x);
        ly_ += (dt_/6.0f) * (k1y + 2*k2y + 2*k3y + k4y);
        lz_ += (dt_/6.0f) * (k1z + 2*k2z + 2*k3z + k4z);
    }
};

}} // namespace cog::pilot

#endif // COG_PILOT_LORENZ_RESERVOIR_HPP
