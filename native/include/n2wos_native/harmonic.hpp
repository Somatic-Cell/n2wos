#pragma once

#include <array>

namespace n2wos_native {

struct Vec3d {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

inline double harmonic_x2_minus_y2(double x, double y, double /*z*/) {
  return x * x - y * y;
}

inline double harmonic_x2_minus_y2(const Vec3d& p) {
  return harmonic_x2_minus_y2(p.x, p.y, p.z);
}

inline double harmonic_x2_minus_y2(const std::array<double, 3>& p) {
  return harmonic_x2_minus_y2(p[0], p[1], p[2]);
}

} // namespace n2wos_native
