#include "density_map.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>

namespace mc {

// ─────────────────────────────────────────────────────────────────────────────
// Binary loader
// ─────────────────────────────────────────────────────────────────────────────

DensityMap load_density_map(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("DensityMap: cannot open file: " + filepath);
    }

    DensityMap m;

    // Header
    int32_t nx32, ny32;
    f.read(reinterpret_cast<char*>(&nx32), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&ny32), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&m.dx),  sizeof(double));
    f.read(reinterpret_cast<char*>(&m.dy),  sizeof(double));

    m.nx = static_cast<int>(nx32);
    m.ny = static_cast<int>(ny32);

    if (m.nx <= 0 || m.ny <= 0) {
        throw std::runtime_error("DensityMap: invalid grid dimensions in file: " + filepath);
    }

    size_t n_voxels = static_cast<size_t>(m.nx) * static_cast<size_t>(m.ny);

    // Read three property arrays
    m.density.resize(n_voxels);
    m.I_value.resize(n_voxels);
    m.Z_over_A.resize(n_voxels);

    f.read(reinterpret_cast<char*>(m.density.data()),   n_voxels * sizeof(double));
    f.read(reinterpret_cast<char*>(m.I_value.data()),   n_voxels * sizeof(double));
    f.read(reinterpret_cast<char*>(m.Z_over_A.data()),  n_voxels * sizeof(double));

    if (!f.good()) {
        throw std::runtime_error("DensityMap: error reading data from: " + filepath);
    }

    std::cout << "DensityMap loaded: " << filepath
              << "  [" << m.nx << " x " << m.ny << "]"
              << "  dx=" << m.dx << " cm  dy=" << m.dy << " cm\n";
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear interpolation (generic)
// ─────────────────────────────────────────────────────────────────────────────

double DensityMap::bilinear(const std::vector<double>& field,
                            double x, double y) const {
    // Convert continuous coordinates → fractional grid indices
    double fi = x / dx;   // depth index (along nx)
    double fj = y / dy;   // lateral index (along ny)

    // Clamp to valid range
    fi = std::max(0.0, std::min(fi, static_cast<double>(nx - 1)));
    fj = std::max(0.0, std::min(fj, static_cast<double>(ny - 1)));

    int i0 = static_cast<int>(fi);
    int j0 = static_cast<int>(fj);
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny - 1);

    double fx = fi - i0;   // fractional part in x
    double fy = fj - j0;   // fractional part in y

    // Bilinear combination of the 4 surrounding voxels
    double v00 = field[j0 * nx + i0];
    double v10 = field[j0 * nx + i1];
    double v01 = field[j1 * nx + i0];
    double v11 = field[j1 * nx + i1];

    return (1.0 - fx) * (1.0 - fy) * v00
         +        fx  * (1.0 - fy) * v10
         + (1.0 - fx) *        fy  * v01
         +        fx  *        fy  * v11;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public accessors (delegate to bilinear helper)
// ─────────────────────────────────────────────────────────────────────────────

double DensityMap::get_density(double x, double y) const {
    return bilinear(density, x, y);
}

double DensityMap::get_I_value(double x, double y) const {
    return bilinear(I_value, x, y);
}

double DensityMap::get_Z_over_A(double x, double y) const {
    return bilinear(Z_over_A, x, y);
}

} // namespace mc
