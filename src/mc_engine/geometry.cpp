#include "geometry.h"

namespace mc {

// ─────────────────────────────────────────────────────────────────────────────
// Geometry (1D slab phantom)
// ─────────────────────────────────────────────────────────────────────────────

Geometry::Geometry(double setup_uncertainty_cm)
    : global_shift(setup_uncertainty_cm), max_depth(0.0) {}

void Geometry::build_default_thoracic() {
    layers.clear();

    // Thoracic phantom layers (ICRU Report 37 / ICRU 49):
    //   Layer 0: Anterior soft tissue (chest wall)   ρ=1.04, I=75.0 eV, Z/A≈0.5500
    //   Layer 1: Lung parenchyma (inflated)          ρ=0.26, I=85.7 eV, Z/A≈0.5494
    //   Layer 2: Posterior soft tissue (mediastinum) ρ=1.04, I=75.0 eV, Z/A≈0.5500
    layers.push_back({0.0,  5.0,  1.04, 75.0, 0.5500});
    layers.push_back({5.0,  15.0, 0.26, 85.7, 0.5494});
    layers.push_back({15.0, 30.0, 1.04, 75.0, 0.5500});

    max_depth = 30.0;
}

void Geometry::apply_setup_uncertainty(double shift_cm) {
    global_shift = shift_cm;
}

// Internal helper shared by get_density and get_properties
static const mc::MaterialLayer* find_layer(const std::vector<mc::MaterialLayer>& layers,
                                           double x, double global_shift) {
    double ex = x - global_shift;  // Effective position in anatomy frame
    for (const auto& L : layers) {
        if (ex >= L.start_pos && ex < L.end_pos) return &L;
    }
    return nullptr;
}

double Geometry::get_density(double x, double& out_I_value) const {
    const MaterialLayer* L = find_layer(layers, x, global_shift);
    if (L) { out_I_value = L->I_value; return L->density; }
    // Air / beyond phantom
    out_I_value = 85.7;
    return 0.001205;
}

double Geometry::get_properties(double x, double& out_I_value) const {
    return get_density(x, out_I_value);
}

double Geometry::get_Z_over_A(double x) const {
    const MaterialLayer* L = find_layer(layers, x, global_shift);
    if (L) return L->Z_over_A;
    return 0.4992; // Dry air Z/A (PDG)
}

double Geometry::get_max_depth() const { return max_depth; }

// ─────────────────────────────────────────────────────────────────────────────
// Geometry2D — wraps DensityMap
// ─────────────────────────────────────────────────────────────────────────────

Geometry2D::Geometry2D(const std::string& density_map_path)
    : map(load_density_map(density_map_path)) {}

double Geometry2D::get_density  (double x, double y) const { return map.get_density(x, y);   }
double Geometry2D::get_I_value  (double x, double y) const { return map.get_I_value(x, y);   }
double Geometry2D::get_Z_over_A (double x, double y) const { return map.get_Z_over_A(x, y);  }
double Geometry2D::get_max_depth()   const { return map.get_max_depth();   }
double Geometry2D::get_max_lateral() const { return map.get_max_lateral(); }

} // namespace mc
