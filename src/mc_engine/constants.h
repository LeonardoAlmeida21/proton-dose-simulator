#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace phys {

// Fundamental Constants (CODATA 2018 / PDG)
constexpr double c        = 299792458.0;       // Speed of light [m/s]
constexpr double m_e      = 0.51099895000;     // Electron mass [MeV/c^2]
constexpr double m_p      = 938.27208943;      // Proton mass [MeV/c^2]
constexpr double e_charge = 1.602176634e-19;   // Elementary charge [C]
constexpr double N_A      = 6.02214076e23;     // Avogadro's number [mol^-1]
constexpr double r_e      = 2.8179403262e-13;  // Classical electron radius [cm]

// Derived constants
constexpr double K_const  = 0.307075;          // 4*pi*N_A*r_e^2*m_e*c^2 [MeV cm^2/mol]

// Radiation lengths (PDG)
constexpr double X0_water = 36.08;             // Radiation length of liquid water [cm]

} // namespace phys

#endif // CONSTANTS_H
