#ifndef PHYSICS_H
#define PHYSICS_H

#include <vector>

namespace mc {

class PhysicsModel {
public:
    PhysicsModel();
    
    // Core function: returns the stopping power (-dE/dx) in MeV/cm
    // Uses cubic-spline-interpolated NIST PSTAR data for water, scaled by local density.
    double calculate_stopping_power(double kinetic_energy_MeV, double density, double I_value) const;

private:
    // Analytical Bethe-Bloch (fallback for energies outside the NIST table range)
    double base_bethe_bloch(double E_k, double density, double I_value) const;

    // Cubic spline interpolation of the NIST PSTAR table
    double interpolate_nist_pstar(double E_k) const;

    // NIST PSTAR table data and cubic spline coefficients
    std::vector<double> nist_energies;        // Kinetic energy [MeV]
    std::vector<double> nist_stopping_powers; // Mass stopping power [MeV cm^2/g]
    std::vector<double> spline_b;             // Spline linear coefficients
    std::vector<double> spline_c;             // Spline quadratic coefficients
    std::vector<double> spline_d;             // Spline cubic coefficients
    
    // Initialisation routines
    void load_nist_pstar_data();
    void compute_cubic_splines();
};

} // namespace mc

#endif // PHYSICS_H
