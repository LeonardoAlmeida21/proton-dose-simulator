#ifndef DENSITY_MAP_H
#define DENSITY_MAP_H

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mc {

// ─────────────────────────────────────────────────────────────────────────────
// DensityMap — 2D material property grid loaded from disk
//
// Layout: row-major [ny][nx] where
//   x = depth axis (beam direction), nx voxels of size dx [cm]
//   y = lateral axis (perpendicular to beam), ny voxels of size dy [cm]
//
// Binary format written by generate_phantom.py (v2 protocol):
//   int32  nx, ny
//   double dx, dy
//   double density[ny*nx]
//   double I_value[ny*nx]
//   double Z_over_A[ny*nx]
// ─────────────────────────────────────────────────────────────────────────────

struct DensityMap {
    int    nx = 0, ny = 0;           // Depth × lateral grid dimensions
    double dx = 0.1, dy = 0.1;      // Voxel spacing [cm]

    std::vector<double> density;     // [g/cm^3], row-major [ny*nx]
    std::vector<double> I_value;     // Mean excitation potential [eV]
    std::vector<double> Z_over_A;    // Atomic number / mass number ratio

    // ── Bounds ───────────────────────────────────────────────────────────────
    double get_max_depth()   const { return nx * dx; }
    double get_max_lateral() const { return ny * dy; }

    // ── Direct index access ──────────────────────────────────────────────────
    double density_at  (int ix, int iy) const { return density  [iy * nx + ix]; }
    double I_value_at  (int ix, int iy) const { return I_value  [iy * nx + ix]; }
    double Z_over_A_at (int ix, int iy) const { return Z_over_A [iy * nx + ix]; }

    // ── Bilinear interpolation at continuous positions ───────────────────────
    // x ∈ [0, nx*dx], y ∈ [0, ny*dy]
    double get_density  (double x, double y) const;
    double get_I_value  (double x, double y) const;
    double get_Z_over_A (double x, double y) const;

private:
    // Generic bilinear interpolation helper
    double bilinear(const std::vector<double>& field, double x, double y) const;
};

// Load a DensityMap from a binary file produced by generate_phantom.py
DensityMap load_density_map(const std::string& filepath);

} // namespace mc

#endif // DENSITY_MAP_H
