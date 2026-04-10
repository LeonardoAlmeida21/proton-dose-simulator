#include "physics.h"
#include "constants.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mc {

PhysicsModel::PhysicsModel() {
    load_nist_pstar_data();
    compute_cubic_splines();
}

void PhysicsModel::load_nist_pstar_data() {
    // NIST PSTAR electronic stopping power for protons in liquid water.
    // Source: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
    // Units: E_k [MeV], S_el [MeV cm^2/g]
    //
    // 26-point table covering the clinical proton therapy range (1–250 MeV).
    // The log-log distribution of points provides denser sampling where the
    // stopping power gradient is steepest (low energies near the Bragg Peak).

    nist_energies = {
        1.0,   1.5,   2.0,   3.0,   4.0,   5.0,
        7.0,   10.0,  15.0,  20.0,  25.0,  30.0,
        40.0,  50.0,  60.0,  70.0,  80.0,  90.0,
        100.0, 120.0, 150.0, 175.0, 200.0, 225.0,
        250.0, 300.0
    };

    nist_stopping_powers = {
        269.0, 204.0, 166.0, 123.0, 99.5,  84.1,
        65.8,  49.6,  36.2,  28.8,  24.1,  20.8,
        16.3,  13.5,  11.6,  10.2,  9.17,  8.35,
        7.69,  6.65,  5.58,  5.01,  4.56,  4.20,
        3.90,  3.43
    };
}

void PhysicsModel::compute_cubic_splines() {
    // Natural cubic spline interpolation via the Thomas algorithm.
    // Given n data points (x_i, y_i), we solve for n-2 interior second
    // derivatives using a tridiagonal system, then derive the polynomial
    // coefficients b_i, c_i, d_i for each interval.
    //
    // S_i(x) = y_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
    //
    // Natural boundary conditions: S''(x_0) = S''(x_{n-1}) = 0.

    int n = static_cast<int>(nist_energies.size());
    if (n < 3) {
        throw std::runtime_error("Need at least 3 NIST data points for cubic spline.");
    }

    // Step sizes h_i = x_{i+1} - x_i
    std::vector<double> h(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i] = nist_energies[i + 1] - nist_energies[i];
    }

    // Build the tridiagonal system for second derivatives (c coefficients)
    // System size: (n-2) interior points
    std::vector<double> alpha(n - 2);
    for (int i = 0; i < n - 2; ++i) {
        alpha[i] = (3.0 / h[i + 1]) * (nist_stopping_powers[i + 2] - nist_stopping_powers[i + 1])
                 - (3.0 / h[i])     * (nist_stopping_powers[i + 1] - nist_stopping_powers[i]);
    }

    // Thomas algorithm: forward sweep
    std::vector<double> l(n - 2, 0.0);
    std::vector<double> mu(n - 2, 0.0);
    std::vector<double> z(n - 2, 0.0);

    l[0] = 2.0 * (h[0] + h[1]);
    mu[0] = h[1] / l[0];
    z[0] = alpha[0] / l[0];

    for (int i = 1; i < n - 2; ++i) {
        l[i] = 2.0 * (h[i] + h[i + 1]) - h[i] * mu[i - 1];
        mu[i] = (i < n - 3) ? h[i + 1] / l[i] : 0.0;
        z[i] = (alpha[i] - h[i] * z[i - 1]) / l[i];
    }

    // Thomas algorithm: back substitution -> compute c_i (second derivatives / 2)
    std::vector<double> c_coeff(n, 0.0);  // natural BC: c[0]=0, c[n-1]=0
    for (int j = n - 3; j >= 0; --j) {
        c_coeff[j + 1] = z[j] - mu[j] * c_coeff[j + 2];
    }

    // Derive b and d coefficients for each interval
    spline_b.resize(n - 1);
    spline_c.resize(n);
    spline_d.resize(n - 1);

    for (int i = 0; i < n; ++i) {
        spline_c[i] = c_coeff[i];
    }

    for (int i = 0; i < n - 1; ++i) {
        spline_b[i] = (nist_stopping_powers[i + 1] - nist_stopping_powers[i]) / h[i]
                     - h[i] * (spline_c[i + 1] + 2.0 * spline_c[i]) / 3.0;
        spline_d[i] = (spline_c[i + 1] - spline_c[i]) / (3.0 * h[i]);
    }
}

double PhysicsModel::interpolate_nist_pstar(double E_k) const {
    // Boundary clamping: return edge values for out-of-range energies
    if (E_k <= nist_energies.front()) return nist_stopping_powers.front();
    if (E_k >= nist_energies.back())  return nist_stopping_powers.back();
    
    // Binary search O(log n) to find the bracketing interval
    auto it = std::lower_bound(nist_energies.begin(), nist_energies.end(), E_k);
    int idx = static_cast<int>(std::distance(nist_energies.begin(), it)) - 1;
    
    // Evaluate the cubic spline polynomial: S(x) = y + b*dx + c*dx^2 + d*dx^3
    double dx = E_k - nist_energies[idx];
    return nist_stopping_powers[idx]
         + spline_b[idx] * dx
         + spline_c[idx] * dx * dx
         + spline_d[idx] * dx * dx * dx;
}

double PhysicsModel::base_bethe_bloch(double E_k, double density, double I_value) const {
    // Relativistic kinematics
    double tau   = E_k / phys::m_p;        // Kinetic energy in units of proton rest mass
    double gamma = tau + 1.0;              // Lorentz factor
    double beta2 = 1.0 - 1.0 / (gamma * gamma);
    
    if (beta2 <= 0.0) return 0.0;  // Particle at rest
    
    // Maximum energy transfer in a single collision (relativistic)
    double me_over_mp = phys::m_e / phys::m_p;
    double W_max = (2.0 * phys::m_e * beta2 * gamma * gamma) / 
                   (1.0 + 2.0 * gamma * me_over_mp + me_over_mp * me_over_mp);

    // Bethe logarithm — all quantities must be in consistent units [MeV and eV]
    // I_value is in eV, so convert to MeV: I_MeV = I_value * 1e-6
    double I_MeV = I_value * 1e-6;
    double ln_arg = (2.0 * phys::m_e * beta2 * gamma * gamma * W_max) / (I_MeV * I_MeV);
    
    if (ln_arg <= 0.0) return 0.0;
    
    double L = 0.5 * std::log(ln_arg) - beta2;

    // Mass stopping power: K * z^2 * (Z/A) * (1/beta^2) * L
    // For water: Z/A ≈ 0.5551, z=1 (proton)
    double dE_dx_mass = phys::K_const * 0.5551 * (1.0 / beta2) * L;
    
    return dE_dx_mass * density; // Convert to linear stopping power [MeV/cm]
}

double PhysicsModel::calculate_stopping_power(double E_k_MeV, double density, double I_value) const {
    // Primary path: NIST PSTAR cubic-spline interpolation for water.
    // The table values are mass stopping power [MeV cm^2/g].
    // Multiply by local density to obtain linear stopping power [MeV/cm].
    //
    // This approach inherently captures the shell corrections, Barkas effect,
    // and density effect that the analytical Bethe-Bloch misses at low energies.
    
    double nist_mass_sp = interpolate_nist_pstar(E_k_MeV);
    return nist_mass_sp * density;
}

} // namespace mc
