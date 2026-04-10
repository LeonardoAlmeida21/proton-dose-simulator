#include "transport.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mc {

// ═════════════════════════════════════════════════════════════════════════════
// Shared physics helper — Bohr straggling (material-aware, kappa-regime)
// ═════════════════════════════════════════════════════════════════════════════

// Generic straggling function used by both 1D and 2D engines.
// Accepts the local Z/A to avoid the previous hardcoded water value.
static double compute_straggling(double E_k, double beta2,
                                 double step_size, double density, double Z_over_A,
                                 std::mt19937& gen) {
    // Bohr variance [MeV^2]:  Omega^2 = K * (Z/A) * rho * dx * m_e
    double omega2 = phys::K_const * Z_over_A * density * step_size * phys::m_e;

    // kappa = xi / W_max  (regime classifier)
    double xi    = phys::K_const * Z_over_A * density * step_size / beta2;
    double gamma = 1.0 / std::sqrt(1.0 - beta2);
    double me_mp = phys::m_e / phys::m_p;
    double W_max = (2.0 * phys::m_e * beta2 * gamma * gamma) /
                   (1.0 + 2.0 * gamma * me_mp + me_mp * me_mp);
    double kappa = (W_max > 1e-12) ? (xi / W_max) : 100.0;

    double sigma2;
    if      (kappa > 10.0) sigma2 = omega2;                  // Gaussian
    else if (kappa > 0.01) sigma2 = omega2 * (1.0 + 1.0 / kappa); // Vavilov
    else                   sigma2 = omega2 * 3.0;            // Landau

    if (sigma2 <= 0.0) return 0.0;
    std::normal_distribution<double> dist(0.0, std::sqrt(sigma2));
    return dist(gen);
}

// ═════════════════════════════════════════════════════════════════════════════
// TransportEngine (1D — backward compatible)
// ═════════════════════════════════════════════════════════════════════════════

TransportEngine::TransportEngine(const PhysicsModel& phys, const Geometry& geom)
    : physics(phys), geometry(geom) {
    std::random_device rd; generator.seed(rd());
}

double TransportEngine::calculate_straggling(double E_k, double beta2,
                                             double step_size, double density,
                                             double Z_over_A) const {
    // Delegate to shared helper (const_cast: generator is logically mutable)
    return compute_straggling(E_k, beta2, step_size, density, Z_over_A,
                              const_cast<std::mt19937&>(generator));
}

std::vector<double> TransportEngine::simulate(int n_histories, double initial_energy_MeV,
                                              double grid_spacing_cm, double step_size_cm,
                                              double energy_cutoff_MeV) {
    int num_bins = static_cast<int>(std::ceil(geometry.get_max_depth() / grid_spacing_cm));
    std::vector<double> dose_grid(num_bins, 0.0);

    for (int hist = 0; hist < n_histories; ++hist) {
        double E   = initial_energy_MeV;
        double pos = 0.0;

        while (E > energy_cutoff_MeV && pos < geometry.get_max_depth()) {
            double I_val   = 0.0;
            double density = geometry.get_properties(pos, I_val);
            double Z_A     = geometry.get_Z_over_A(pos);

            double tau   = E / phys::m_p;
            double gamma = tau + 1.0;
            double beta2 = 1.0 - 1.0 / (gamma * gamma);

            double dE_dx = physics.calculate_stopping_power(E, density, I_val);
            double dE    = dE_dx * step_size_cm
                         + calculate_straggling(E, beta2, step_size_cm, density, Z_A);
            dE = std::clamp(dE, 0.0, E);

            E -= dE;

            int bin = static_cast<int>(pos / grid_spacing_cm);
            if (bin >= 0 && bin < num_bins)
                dose_grid[bin] += dE / n_histories;

            pos += step_size_cm;
        }
    }
    return dose_grid;
}

// ═════════════════════════════════════════════════════════════════════════════
// TransportEngine2D — Highland MCS + OpenMP
// ═════════════════════════════════════════════════════════════════════════════

TransportEngine2D::TransportEngine2D(const PhysicsModel& phys, const Geometry2D& geom)
    : physics(phys), geometry(geom) {
    std::random_device rd; generator.seed(rd());
}

double TransportEngine2D::calculate_highland_angle(double p_MeV, double beta,
                                                   double step_cm, double X0_cm) const {
    // Highland (1975) / PDG formula for projected RMS scattering angle:
    //   theta_RMS = (13.6 MeV / (p * beta)) * sqrt(dx/X0) * [1 + 0.038 * ln(dx/X0)]
    if (X0_cm <= 0.0 || step_cm <= 0.0) return 0.0;
    double t = step_cm / X0_cm;              // Material thickness in radiation lengths
    if (t <= 0.0) return 0.0;
    double ln_t = std::log(t);
    double theta_rms = (13.6 / (p_MeV * beta)) * std::sqrt(t) * (1.0 + 0.038 * ln_t);
    return std::abs(theta_rms);              // [rad]
}

double TransportEngine2D::calculate_straggling(double E_k, double beta2,
                                               double step_size, double density,
                                               double Z_over_A) const {
    return compute_straggling(E_k, beta2, step_size, density, Z_over_A,
                              const_cast<std::mt19937&>(generator));
}

std::vector<double> TransportEngine2D::simulate(int n_histories, double initial_energy_MeV,
                                                double grid_spacing_cm, double step_size_cm,
                                                double energy_cutoff_MeV) {
    int nx = geometry.get_nx();
    int ny = geometry.get_ny();
    double max_depth   = geometry.get_max_depth();
    double max_lateral = geometry.get_max_lateral();
    double y_center    = max_lateral / 2.0;  // Beam enters at lateral midpoint

    std::vector<double> dose_grid(static_cast<size_t>(nx) * ny, 0.0);

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    // Thread-local dose accumulators to avoid race conditions
    std::vector<std::vector<double>> local_doses(n_threads,
        std::vector<double>(static_cast<size_t>(nx) * ny, 0.0));

    // Base seed for thread-local PRNGs
    unsigned int base_seed = static_cast<unsigned int>(generator());

#pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::mt19937 local_gen(base_seed + static_cast<unsigned int>(tid) * 1000007u);
        std::vector<double>& local_dose = local_doses[tid];

#pragma omp for schedule(dynamic, 64)
        for (int hist = 0; hist < n_histories; ++hist) {
            double E     = initial_energy_MeV;
            double pos_x = 0.0;
            double pos_y = y_center;
            double theta = 0.0;   // Proton direction angle w.r.t. depth axis [rad]

            while (E > energy_cutoff_MeV && pos_x < max_depth) {
                double density = geometry.get_density(pos_x, pos_y);
                double I_val   = geometry.get_I_value(pos_x, pos_y);
                double Z_A     = geometry.get_Z_over_A(pos_x, pos_y);

                // Radiation length for Highland MCS
                double X0 = (density > 1e-6) ? (phys::X0_water / density) : 1e6;

                // Relativistic kinematics
                double p    = std::sqrt(E * E + 2.0 * E * phys::m_p); // [MeV/c]
                double Etot = E + phys::m_p;
                double beta = p / Etot;
                double beta2 = beta * beta;

                // ── 1. Energy loss ──────────────────────────────────────────
                double dE_dx   = physics.calculate_stopping_power(E, density, I_val);
                double dE_mean = dE_dx * step_size_cm;

                // Interface-aware straggling: check for material transition
                double density_next = geometry.get_density(
                    pos_x + step_size_cm,
                    pos_y + std::tan(theta) * step_size_cm);
                double dE_strag;
                if (std::abs(density_next - density) / (density + 1e-9) > 0.3) {
                    // Interface detected: split step at midpoint
                    double half = step_size_cm * 0.5;
                    double Z_A_next = geometry.get_Z_over_A(
                        pos_x + step_size_cm,
                        pos_y + std::tan(theta) * step_size_cm);
                    dE_strag = compute_straggling(E, beta2, half, density,      Z_A,      local_gen)
                             + compute_straggling(E, beta2, half, density_next, Z_A_next, local_gen);
                } else {
                    dE_strag = compute_straggling(E, beta2, step_size_cm, density, Z_A, local_gen);
                }

                double dE = std::clamp(dE_mean + dE_strag, 0.0, E);

                // ── 2. Multiple Coulomb Scattering (Highland) ───────────────
                double theta_rms = calculate_highland_angle(p, beta, step_size_cm, X0);
                std::normal_distribution<double> angle_dist(0.0, theta_rms);
                theta += angle_dist(local_gen);

                // ── 3. Update position ──────────────────────────────────────
                pos_y += std::tan(theta) * step_size_cm;
                pos_x += step_size_cm;

                // Reflect at lateral boundaries
                if (pos_y < 0.0) {
                    pos_y  = -pos_y;
                    theta  = -theta;
                } else if (pos_y >= max_lateral) {
                    pos_y  = 2.0 * max_lateral - pos_y;
                    theta  = -theta;
                }

                E -= dE;

                // ── 4. Score dose ───────────────────────────────────────────
                int ix = static_cast<int>(pos_x / grid_spacing_cm);
                int iy = static_cast<int>(pos_y / grid_spacing_cm);
                if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
                    local_dose[static_cast<size_t>(iy) * nx + ix] += dE / n_histories;
                }
            }
        } // end omp for

        // Reduce thread-local accumulators
#pragma omp critical
        {
            for (size_t i = 0; i < dose_grid.size(); ++i)
                dose_grid[i] += local_dose[i];
        }
    } // end omp parallel

    return dose_grid;
}

} // namespace mc
