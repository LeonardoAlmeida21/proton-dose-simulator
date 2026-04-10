#ifndef EXPORT_H
#define EXPORT_H

#include <vector>
#include <string>

namespace mc {

class ExportModule {
public:
    // ── 1D export (v1 — original format, backward compatible) ────────────────
    static void save_binary(const std::string& filepath,
                            const std::vector<double>& dose_grid,
                            double grid_size_cm,
                            double step_size_cm,
                            int n_histories);

    // ── 2D export (v2 — versioned header with nx, ny) ────────────────────────
    static void save_binary_2d(const std::string& filepath,
                               const std::vector<double>& dose_grid_2d,
                               int nx, int ny,
                               double grid_size_cm,
                               double step_size_cm,
                               int n_histories);
};

} // namespace mc

#endif // EXPORT_H
