#include "export.h"
#include <fstream>
#include <iostream>
#include <cstdint>

namespace mc {

// ─────────────────────────────────────────────────────────────────────────────
// 1D export — v1 binary format (original, unchanged)
//
// Layout:
//   double  grid_size_cm   (8 bytes)
//   double  step_size_cm   (8 bytes)
//   int32   n_histories    (4 bytes)
//   int32   num_bins       (4 bytes)
//   double  dose_data[num_bins]
// ─────────────────────────────────────────────────────────────────────────────

void ExportModule::save_binary(const std::string& filepath,
                               const std::vector<double>& dose_grid,
                               double grid_size_cm,
                               double step_size_cm,
                               int n_histories) {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Export] Cannot open: " << filepath << "\n";
        return;
    }
    int32_t num_bins  = static_cast<int32_t>(dose_grid.size());
    int32_t n_hist    = static_cast<int32_t>(n_histories);

    f.write(reinterpret_cast<const char*>(&grid_size_cm), sizeof(double));
    f.write(reinterpret_cast<const char*>(&step_size_cm), sizeof(double));
    f.write(reinterpret_cast<const char*>(&n_hist),       sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&num_bins),     sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(dose_grid.data()),
            num_bins * static_cast<std::streamsize>(sizeof(double)));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2D export — v2 binary format (versioned header)
//
// Layout:
//   int32   version = 2    (4 bytes)
//   int32   nx             (4 bytes)
//   int32   ny             (4 bytes)
//   double  grid_size_cm   (8 bytes)
//   double  step_size_cm   (8 bytes)
//   int32   n_histories    (4 bytes)
//   double  dose_data[ny*nx]  (row-major)
// ─────────────────────────────────────────────────────────────────────────────

void ExportModule::save_binary_2d(const std::string& filepath,
                                  const std::vector<double>& dose_grid_2d,
                                  int nx, int ny,
                                  double grid_size_cm,
                                  double step_size_cm,
                                  int n_histories) {
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Export] Cannot open: " << filepath << "\n";
        return;
    }
    int32_t version = 2;
    int32_t nx32    = static_cast<int32_t>(nx);
    int32_t ny32    = static_cast<int32_t>(ny);
    int32_t n_hist  = static_cast<int32_t>(n_histories);

    f.write(reinterpret_cast<const char*>(&version),      sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&nx32),         sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&ny32),         sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&grid_size_cm), sizeof(double));
    f.write(reinterpret_cast<const char*>(&step_size_cm), sizeof(double));
    f.write(reinterpret_cast<const char*>(&n_hist),       sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(dose_grid_2d.data()),
            static_cast<std::streamsize>(dose_grid_2d.size() * sizeof(double)));
}

} // namespace mc
