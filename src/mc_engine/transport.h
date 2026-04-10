#ifndef TRANSPORT_H
#define TRANSPORT_H

#include "physics.h"
#include "geometry.h"
#include <vector>
#include <random>

namespace mc {

// ─────────────────────────────────────────────────────────────────────────────
// TransportEngine — original 1D pencil-beam engine (backward compatible)
// ─────────────────────────────────────────────────────────────────────────────

class TransportEngine {
public:
    TransportEngine(const PhysicsModel& phys, const Geometry& geom);

    // Returns 1D dose grid [num_bins], energy deposit normalised by N_histories
    std::vector<double> simulate(int n_histories, double initial_energy_MeV,
                                 double grid_spacing_cm, double step_size_cm,
                                 double energy_cutoff_MeV);

private:
    const PhysicsModel& physics;
    const Geometry& geometry;
    std::mt19937 generator;

    double calculate_straggling(double E_k, double beta2,
                                double step_size, double density,
                                double Z_over_A) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// TransportEngine2D — 2D pencil-beam engine with Highland MCS
// ─────────────────────────────────────────────────────────────────────────────

class TransportEngine2D {
public:
    TransportEngine2D(const PhysicsModel& phys, const Geometry2D& geom);

    // Returns 2D dose grid [ny * nx] (row-major), energy deposit / N_histories
    std::vector<double> simulate(int n_histories, double initial_energy_MeV,
                                 double grid_spacing_cm, double step_size_cm,
                                 double energy_cutoff_MeV);

private:
    const PhysicsModel& physics;
    const Geometry2D& geometry;
    std::mt19937 generator;

    // Highland (1975) RMS projected scattering angle [rad] for one step
    double calculate_highland_angle(double p_MeV, double beta,
                                    double step_cm, double X0_cm) const;

    // Bohr straggling with material-specific Z/A and kappa-regime classification
    double calculate_straggling(double E_k, double beta2,
                                double step_size, double density,
                                double Z_over_A) const;
};

} // namespace mc

#endif // TRANSPORT_H
