#include "physics.h"
#include "geometry.h"
#include "transport.h"
#include "export.h"

#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>

int main(int argc, char* argv[]) {
    std::cout << "--- High Precision Proton Therapy MC Engine (2D) ---\n";

    // Default simulation parameters
    double      energy           = 150.0;
    double      setup_shift      = 0.0;
    std::string output_dir       = "data";
    std::string density_map_path = "";    // Empty = use 1D backward-compatible mode
    bool        use_2d           = false;

    // Parse CLI arguments:
    //   [energy_MeV] [setup_shift_cm] [output_dir] [--density-map path]
    if (argc > 1) energy      = std::stod(argv[1]);
    if (argc > 2) setup_shift = std::stod(argv[2]);
    if (argc > 3) output_dir  = argv[3];
    for (int i = 4; i < argc - 1; ++i) {
        std::string flag(argv[i]);
        if (flag == "--density-map") {
            density_map_path = argv[i + 1];
            use_2d           = true;
        }
    }

    mc::PhysicsModel physics;

    auto t_start = std::chrono::steady_clock::now();

    if (use_2d) {
        // ── 2D mode: load external density map ───────────────────────────────
        std::cout << "Mode: 2D | E=" << energy << " MeV"
                  << " | map='" << density_map_path << "'"
                  << " | output='" << output_dir << "'\n";

        mc::Geometry2D geometry(density_map_path);
        mc::TransportEngine2D engine(physics, geometry);

        int    nx           = geometry.get_nx();
        int    ny           = geometry.get_ny();
        double grid_cm      = geometry.get_dx();
        double step_cm      = grid_cm / 2.0;          // Step = half a voxel
        double cutoff_MeV   = 0.1;

        // High-statistics reference (100k histories)
        int n_ref = 100000;
        std::cout << "  Simulating reference (" << n_ref << " histories)...\n";
        auto ref_grid = engine.simulate(n_ref, energy, grid_cm, step_cm, cutoff_MeV);
        mc::ExportModule::save_binary_2d(output_dir + "/reference_output.bin",
                                         ref_grid, nx, ny, grid_cm, step_cm, n_ref);

        // Low-statistics noisy input (100 histories)
        int n_noisy = 100;
        std::cout << "  Simulating noisy    (" << n_noisy << " histories)...\n";
        auto noisy_grid = engine.simulate(n_noisy, energy, grid_cm, step_cm, cutoff_MeV);
        mc::ExportModule::save_binary_2d(output_dir + "/noisy_output.bin",
                                         noisy_grid, nx, ny, grid_cm, step_cm, n_noisy);

        // Physics verification: Bragg Peak from laterally-integrated depth-dose profile.
        // Summing over all lateral rows gives reliable statistics regardless of scatter width.
        double max_dose = -1.0;
        int    max_bin  = -1;
        for (int ix = 0; ix < nx; ++ix) {
            double lateral_sum = 0.0;
            for (int iy = 0; iy < ny; ++iy)
                lateral_sum += ref_grid[static_cast<size_t>(iy) * nx + ix];
            if (lateral_sum > max_dose) { max_dose = lateral_sum; max_bin = ix; }
        }
        std::cout << "  Bragg Peak (lateral integral) @ "
                  << (max_bin * grid_cm) << " cm\n";

    } else {
        // ── 1D backward-compatible mode ───────────────────────────────────────
        std::cout << "Mode: 1D | E=" << energy << " MeV"
                  << " | shift=" << setup_shift << " cm"
                  << " | output='" << output_dir << "'\n";

        mc::Geometry geometry(setup_shift);
        geometry.build_default_thoracic();
        mc::TransportEngine engine(physics, geometry);

        double grid_cm    = 0.1;
        double step_cm    = 0.05;
        double cutoff_MeV = 0.1;

        int n_ref = 100000;
        auto ref_grid = engine.simulate(n_ref, energy, grid_cm, step_cm, cutoff_MeV);
        mc::ExportModule::save_binary(output_dir + "/reference_output.bin",
                                      ref_grid, grid_cm, step_cm, n_ref);

        int n_noisy = 100;
        auto noisy_grid = engine.simulate(n_noisy, energy, grid_cm, step_cm, cutoff_MeV);
        mc::ExportModule::save_binary(output_dir + "/noisy_output.bin",
                                      noisy_grid, grid_cm, step_cm, n_noisy);

        // Bragg Peak verification
        double max_dose = -1.0; int max_bin = -1;
        for (size_t i = 0; i < ref_grid.size(); ++i)
            if (ref_grid[i] > max_dose) { max_dose = ref_grid[i]; max_bin = (int)i; }
        std::cout << "  Bragg Peak @ " << (max_bin * grid_cm) << " cm\n";
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Done in " << elapsed << " s\n";

    return 0;
}
