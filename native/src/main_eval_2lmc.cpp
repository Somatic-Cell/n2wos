#include <fcpw/fcpw.h>
#if N2WOS_ENABLE_FCPW_GPU
#include <fcpw/fcpw_gpu.h>
#endif

#include <n2wos_native/harmonic.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef N2WOS_ENABLE_TCNN
#define N2WOS_ENABLE_TCNN 0
#endif

namespace n2wos {

using Clock = std::chrono::steady_clock;

struct Args {
  std::string backend = "cuda";
  std::string cache = "exact"; // exact, scaled_exact, zero, tcnn_stub, tcnn
  std::string json_path;
  std::string dump_samples_path;
  std::string fcpw_dir = N2WOS_FCPW_SOURCE_DIR;
  int lat_segments = 128;
  int lon_segments = 256;
  float radius = 1.0f;
  float bump_amplitude = 0.08f;
  float bump_frequency = 7.0f;
  int m = 8;
  int max_steps = 512;
  int n_pure = 65536;
  int n_coarse = 65536;
  int n_residual = 65536;
  int warmup_queries = 65536;
  float eps = 1.0e-4f;
  float safety = 0.99f;
  uint32_t seed = 12345u;
  double x0 = 0.10;
  double y0 = 0.05;
  double z0 = 0.00;
  double cache_scale = 1.0;
  double cache_bias = 0.0;
  bool print_logs = false;
  bool skip_pure = false;
  bool skip_coarse = false;
  bool skip_residual = false;
};

struct Mesh {
  std::vector<fcpw::Vector3> positions;
  std::vector<fcpw::Vector3i> indices;
};

struct QueryBatch {
  std::vector<fcpw::Vector3> closest_points;
  std::vector<float> distances;
};

struct Stats {
  int n = 0;
  double mean = 0.0;
  double variance = 0.0;
  double std_error = 0.0;
  double bias = 0.0;
  double mse_model = 0.0;
};

struct RunResult {
  std::string name;
  int n = 0;
  int m = 0;
  int max_steps = 0;
  double total_usec = 0.0;
  std::uint64_t query_count = 0;
  double avg_steps = 0.0;
  int max_observed_steps = 0;
  int boundary_count = 0;
  int truncated_count = 0;
  std::vector<int> active_counts;
  std::vector<double> values;
  std::vector<fcpw::Vector3> cache_points;
  std::vector<int> cache_is_boundary;
  std::vector<int> cache_steps;
};

struct ResidualResult {
  int n = 0;
  int m = 0;
  int max_steps = 0;
  double total_usec = 0.0;
  std::uint64_t query_count = 0;
  double avg_steps = 0.0;
  int max_observed_steps = 0;
  int boundary_count = 0;
  int truncated_count = 0;
  std::vector<int> active_counts;
  std::vector<double> w;
  std::vector<double> c;
  std::vector<double> r;
  std::vector<fcpw::Vector3> cache_points;
  std::vector<int> cache_is_boundary;
  std::vector<int> cache_steps;
};

[[noreturn]] void die(const std::string& msg) { throw std::runtime_error(msg); }

std::string next_value(int& i, int argc, char** argv, const char* name) {
  if (i + 1 >= argc) die(std::string("missing value for ") + name);
  return argv[++i];
}

bool is_gpu_backend(const std::string& backend) {
  return backend == "cuda" || backend == "vulkan" || backend == "default" || backend == "d3d12";
}

std::string json_escape(const std::string& s) {
  std::ostringstream o;
  for (unsigned char c : s) {
    switch (c) {
      case '\\': o << "\\\\"; break;
      case '"': o << "\\\""; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if (c < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec << std::setfill(' ');
        } else {
          o << static_cast<char>(c);
        }
    }
  }
  return o.str();
}

std::string env_value(const char* key) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : std::string();
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--backend") a.backend = next_value(i, argc, argv, "--backend");
    else if (key == "--cache") a.cache = next_value(i, argc, argv, "--cache");
    else if (key == "--json") a.json_path = next_value(i, argc, argv, "--json");
    else if (key == "--dump-samples" || key == "--dump_samples") a.dump_samples_path = next_value(i, argc, argv, "--dump-samples");
    else if (key == "--fcpw-dir" || key == "--fcpw_dir") a.fcpw_dir = next_value(i, argc, argv, "--fcpw-dir");
    else if (key == "--m" || key == "--depth") a.m = std::stoi(next_value(i, argc, argv, "--m"));
    else if (key == "--max-steps" || key == "--max_steps") a.max_steps = std::stoi(next_value(i, argc, argv, "--max-steps"));
    else if (key == "--n-pure" || key == "--n_pure") a.n_pure = std::stoi(next_value(i, argc, argv, "--n-pure"));
    else if (key == "--n-coarse" || key == "--n_coarse") a.n_coarse = std::stoi(next_value(i, argc, argv, "--n-coarse"));
    else if (key == "--n-residual" || key == "--n_residual") a.n_residual = std::stoi(next_value(i, argc, argv, "--n-residual"));
    else if (key == "--warmup-queries" || key == "--warmup_queries") a.warmup_queries = std::stoi(next_value(i, argc, argv, "--warmup-queries"));
    else if (key == "--eps" || key == "--epsilon") a.eps = std::stof(next_value(i, argc, argv, "--eps"));
    else if (key == "--safety") a.safety = std::stof(next_value(i, argc, argv, "--safety"));
    else if (key == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next_value(i, argc, argv, "--seed")));
    else if (key == "--x0") a.x0 = std::stod(next_value(i, argc, argv, "--x0"));
    else if (key == "--y0") a.y0 = std::stod(next_value(i, argc, argv, "--y0"));
    else if (key == "--z0") a.z0 = std::stod(next_value(i, argc, argv, "--z0"));
    else if (key == "--cache-scale" || key == "--cache_scale") a.cache_scale = std::stod(next_value(i, argc, argv, "--cache-scale"));
    else if (key == "--cache-bias" || key == "--cache_bias") a.cache_bias = std::stod(next_value(i, argc, argv, "--cache-bias"));
    else if (key == "--lat-segments" || key == "--lat_segments") a.lat_segments = std::stoi(next_value(i, argc, argv, "--lat-segments"));
    else if (key == "--lon-segments" || key == "--lon_segments") a.lon_segments = std::stoi(next_value(i, argc, argv, "--lon-segments"));
    else if (key == "--radius") a.radius = std::stof(next_value(i, argc, argv, "--radius"));
    else if (key == "--bump-amplitude" || key == "--bump_amplitude") a.bump_amplitude = std::stof(next_value(i, argc, argv, "--bump-amplitude"));
    else if (key == "--bump-frequency" || key == "--bump_frequency") a.bump_frequency = std::stof(next_value(i, argc, argv, "--bump-frequency"));
    else if (key == "--print-logs" || key == "--print_logs") a.print_logs = true;
    else if (key == "--skip-pure" || key == "--skip_pure") a.skip_pure = true;
    else if (key == "--skip-coarse" || key == "--skip_coarse") a.skip_coarse = true;
    else if (key == "--skip-residual" || key == "--skip_residual") a.skip_residual = true;
    else if (key == "--help" || key == "-h") {
      std::cout
        << "Usage: n2wos_eval_2lmc [options]\n\n"
        << "  --backend cpu|cuda|vulkan|default\n"
        << "  --cache exact|scaled_exact|zero|tcnn_stub|tcnn\n"
        << "  --m N --max-steps N\n"
        << "  --n-pure N --n-coarse N --n-residual N\n"
        << "  --warmup-queries N\n"
        << "  --x0 X --y0 Y --z0 Z\n"
        << "  --json PATH\n"
        << "  --dump-samples PATH   CSV dump of coarse/residual cache points for external cache evaluation\n";
      std::exit(0);
    } else {
      die("unknown argument: " + key);
    }
  }
  if (a.backend != "cpu" && !is_gpu_backend(a.backend)) die("unsupported backend");
  if (a.cache != "exact" && a.cache != "scaled_exact" && a.cache != "zero" && a.cache != "tcnn_stub" && a.cache != "tcnn") die("unsupported cache");
  if (a.cache == "tcnn") die("cache=tcnn is reserved for the next training/inference patch; use exact, scaled_exact, zero, or tcnn_stub in 0005");
  if (a.m < 0 || a.max_steps < 0 || a.max_steps < a.m) die("require 0 <= m <= max_steps");
  if (a.n_pure <= 0 || a.n_coarse <= 0 || a.n_residual <= 0) die("sample counts must be positive");
  if (a.warmup_queries < 0) die("warmup_queries must be non-negative");
  if (a.lat_segments < 8 || a.lon_segments < 16) die("mesh resolution too small");
  if (!(a.eps > 0.0f)) die("eps must be positive");
  if (!(a.safety > 0.0f && a.safety <= 1.0f)) die("safety must be in (0, 1]");
  return a;
}

float bumpy_radius(float theta, float phi, const Args& args) {
  const float s = std::sin(theta);
  const float bump = 0.55f * std::sin(args.bump_frequency * theta + 0.31f) * std::sin(args.bump_frequency * phi + 0.17f) * s * s
                   + 0.30f * std::cos(1.7f * args.bump_frequency * phi + 0.73f) * s * s
                   + 0.15f * std::sin(2.3f * args.bump_frequency * theta + 0.41f) * s;
  return args.radius * (1.0f + args.bump_amplitude * bump);
}

Mesh make_bumpy_sphere(const Args& args) {
  Mesh mesh;
  const int lat = args.lat_segments;
  const int lon = args.lon_segments;
  const float pi = 3.14159265358979323846f;
  mesh.positions.reserve(2 + (lat - 1) * lon);
  mesh.indices.reserve(2 * lon * (lat - 1));
  mesh.positions.emplace_back(0.0f, 0.0f, bumpy_radius(0.0f, 0.0f, args));
  auto ring_index = [&](int i, int j) -> int { return 1 + (i - 1) * lon + ((j % lon + lon) % lon); };
  for (int i = 1; i <= lat - 1; ++i) {
    const float theta = pi * static_cast<float>(i) / static_cast<float>(lat);
    for (int j = 0; j < lon; ++j) {
      const float phi = 2.0f * pi * static_cast<float>(j) / static_cast<float>(lon);
      const float r = bumpy_radius(theta, phi, args);
      mesh.positions.emplace_back(r * std::sin(theta) * std::cos(phi), r * std::sin(theta) * std::sin(phi), r * std::cos(theta));
    }
  }
  const int bottom = static_cast<int>(mesh.positions.size());
  mesh.positions.emplace_back(0.0f, 0.0f, -bumpy_radius(pi, 0.0f, args));
  for (int j = 0; j < lon; ++j) mesh.indices.emplace_back(0, ring_index(1, j), ring_index(1, (j + 1) % lon));
  for (int i = 1; i <= lat - 2; ++i) {
    for (int j = 0; j < lon; ++j) {
      const int jn = (j + 1) % lon;
      const int a = ring_index(i, j), b = ring_index(i + 1, j), c = ring_index(i + 1, jn), d = ring_index(i, jn);
      mesh.indices.emplace_back(a, b, c);
      mesh.indices.emplace_back(a, c, d);
    }
  }
  for (int j = 0; j < lon; ++j) mesh.indices.emplace_back(ring_index(lat - 1, (j + 1) % lon), ring_index(lat - 1, j), bottom);
  return mesh;
}

void build_scene(const Mesh& mesh, bool vectorized_cpu_bvh, fcpw::Scene<3>& scene) {
  scene.setObjectCount(1);
  scene.setObjectVertices(mesh.positions, 0);
  scene.setObjectTriangles(mesh.indices, 0);
  scene.build(fcpw::AggregateType::Bvh_SurfaceArea, vectorized_cpu_bvh, false, false);
}

fcpw::Vector3 start_point(const Args& args) { return fcpw::Vector3(static_cast<float>(args.x0), static_cast<float>(args.y0), static_cast<float>(args.z0)); }

double target_value(const fcpw::Vector3& p) { return n2wos_native::harmonic_x2_minus_y2(static_cast<double>(p[0]), static_cast<double>(p[1]), static_cast<double>(p[2])); }
double boundary_value(const fcpw::Vector3& p) { return target_value(p); }

double cache_value(const fcpw::Vector3& p, const Args& args) {
  const double u = target_value(p);
  if (args.cache == "exact" || args.cache == "tcnn_stub") return u;
  if (args.cache == "scaled_exact") return args.cache_scale * u + args.cache_bias;
  if (args.cache == "zero") return 0.0;
  die("unsupported cache");
}

fcpw::Vector3 random_unit_vector(std::mt19937& rng) {
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const float x = normal(rng), y = normal(rng), z = normal(rng);
  const float inv = 1.0f / std::sqrt(std::max(1.0e-30f, x * x + y * y + z * z));
  return fcpw::Vector3(x * inv, y * inv, z * inv);
}

QueryBatch query_cpu(fcpw::Scene<3>& scene, const std::vector<fcpw::Vector3>& points) {
  std::vector<fcpw::BoundingSphere<3>> spheres;
  spheres.reserve(points.size());
  for (const auto& p : points) spheres.emplace_back(p, fcpw::maxFloat);
  std::vector<fcpw::Interaction<3>> interactions;
  scene.findClosestPoints(spheres, interactions);
  QueryBatch out;
  out.closest_points.reserve(interactions.size());
  out.distances.reserve(interactions.size());
  for (const auto& it : interactions) {
    out.closest_points.emplace_back(it.p);
    out.distances.emplace_back(it.d);
  }
  return out;
}

#if N2WOS_ENABLE_FCPW_GPU
QueryBatch query_gpu(fcpw::GPUScene<3>& scene, const std::vector<fcpw::Vector3>& points) {
  using namespace fcpw;
  std::vector<GPUBoundingSphere> spheres;
  spheres.reserve(points.size());
  for (const auto& p : points) spheres.emplace_back(float3{p[0], p[1], p[2]}, maxFloat);
  std::vector<GPUInteraction> interactions;
  scene.findClosestPoints(spheres, interactions);
  QueryBatch out;
  out.closest_points.reserve(interactions.size());
  out.distances.reserve(interactions.size());
  for (const auto& it : interactions) {
    out.closest_points.emplace_back(it.p.x, it.p.y, it.p.z);
    out.distances.emplace_back(it.d);
  }
  return out;
}
#endif

Stats summarize(const std::vector<double>& v, double exact) {
  Stats s;
  s.n = static_cast<int>(v.size());
  if (v.empty()) return s;
  s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  double ss = 0.0;
  for (double x : v) ss += (x - s.mean) * (x - s.mean);
  s.variance = v.size() > 1 ? ss / static_cast<double>(v.size() - 1) : 0.0;
  s.std_error = std::sqrt(s.variance / static_cast<double>(v.size()));
  s.bias = s.mean - exact;
  s.mse_model = s.bias * s.bias + s.variance / static_cast<double>(v.size());
  return s;
}

double checksum(const std::vector<double>& v) {
  double s = 0.0;
  const size_t n = std::min<size_t>(v.size(), 256);
  for (size_t i = 0; i < n; ++i) s += v[i] * (1.0 + 1.0e-4 * static_cast<double>(i));
  return s;
}

int max_step(const std::vector<int>& steps) { return steps.empty() ? 0 : *std::max_element(steps.begin(), steps.end()); }

template <class QueryFn>
void warmup(const Args& args, QueryFn&& query_fn) {
  if (args.warmup_queries <= 0) return;
  std::mt19937 rng(args.seed + 777u);
  std::uniform_real_distribution<float> uni(-0.25f * args.radius, 0.25f * args.radius);
  std::vector<fcpw::Vector3> pts;
  pts.reserve(static_cast<size_t>(args.warmup_queries));
  for (int i = 0; i < args.warmup_queries; ++i) pts.emplace_back(uni(rng), uni(rng), uni(rng));
  QueryBatch q = query_fn(pts);
  if (static_cast<int>(q.distances.size()) != args.warmup_queries) die("warmup query returned wrong size");
}

template <class QueryFn>
RunResult run_pure(const Args& args, QueryFn&& query_fn) {
  RunResult out;
  out.name = "pure_wos";
  out.n = args.n_pure;
  out.m = -1;
  out.max_steps = args.max_steps;
  out.values.assign(static_cast<size_t>(out.n), 0.0);
  std::vector<fcpw::Vector3> x(static_cast<size_t>(out.n), start_point(args));
  std::vector<int> steps(static_cast<size_t>(out.n), 0), active;
  active.reserve(static_cast<size_t>(out.n));
  for (int i = 0; i < out.n; ++i) active.push_back(i);
  std::mt19937 rng(args.seed + 101u);
  const auto t0 = Clock::now();
  for (int step = 0; step < args.max_steps && !active.empty(); ++step) {
    out.active_counts.push_back(static_cast<int>(active.size()));
    out.query_count += static_cast<std::uint64_t>(active.size());
    std::vector<fcpw::Vector3> batch;
    batch.reserve(active.size());
    for (int idx : active) batch.push_back(x[static_cast<size_t>(idx)]);
    QueryBatch q = query_fn(batch);
    if (q.distances.size() != batch.size()) die("pure query returned wrong size");
    std::vector<int> next;
    next.reserve(active.size());
    for (size_t k = 0; k < active.size(); ++k) {
      const int idx = active[k];
      const size_t si = static_cast<size_t>(idx);
      const float d = q.distances[k];
      steps[si] += 1;
      if (!(d > args.eps) || !std::isfinite(d)) {
        out.values[si] = boundary_value(q.closest_points[k]);
        out.boundary_count += 1;
      } else {
        x[si] += (args.safety * d) * random_unit_vector(rng);
        next.push_back(idx);
      }
    }
    active.swap(next);
  }
  for (int idx : active) {
    out.values[static_cast<size_t>(idx)] = target_value(x[static_cast<size_t>(idx)]);
    out.truncated_count += 1;
  }
  const auto t1 = Clock::now();
  out.total_usec = std::chrono::duration<double, std::micro>(t1 - t0).count();
  out.avg_steps = std::accumulate(steps.begin(), steps.end(), 0.0) / static_cast<double>(steps.size());
  out.max_observed_steps = max_step(steps);
  return out;
}

template <class QueryFn>
RunResult run_coarse(const Args& args, QueryFn&& query_fn) {
  RunResult out;
  out.name = "coarse_cache";
  out.n = args.n_coarse;
  out.m = args.m;
  out.max_steps = args.m;
  out.values.assign(static_cast<size_t>(out.n), 0.0);
  out.cache_points.assign(static_cast<size_t>(out.n), start_point(args));
  out.cache_is_boundary.assign(static_cast<size_t>(out.n), 0);
  out.cache_steps.assign(static_cast<size_t>(out.n), 0);
  std::vector<fcpw::Vector3> x(static_cast<size_t>(out.n), start_point(args));
  std::vector<int> steps(static_cast<size_t>(out.n), 0), active;
  active.reserve(static_cast<size_t>(out.n));
  for (int i = 0; i < out.n; ++i) active.push_back(i);
  std::mt19937 rng(args.seed + 202u);
  const auto t0 = Clock::now();
  if (args.m == 0) {
    for (int i = 0; i < out.n; ++i) {
      const size_t si = static_cast<size_t>(i);
      out.cache_points[si] = x[si];
      out.cache_is_boundary[si] = 0;
      out.cache_steps[si] = 0;
      out.values[si] = cache_value(x[si], args);
    }
  } else {
    for (int step = 0; step < args.m && !active.empty(); ++step) {
      out.active_counts.push_back(static_cast<int>(active.size()));
      out.query_count += static_cast<std::uint64_t>(active.size());
      std::vector<fcpw::Vector3> batch;
      batch.reserve(active.size());
      for (int idx : active) batch.push_back(x[static_cast<size_t>(idx)]);
      QueryBatch q = query_fn(batch);
      if (q.distances.size() != batch.size()) die("coarse query returned wrong size");
      std::vector<int> next;
      next.reserve(active.size());
      for (size_t k = 0; k < active.size(); ++k) {
        const int idx = active[k];
        const size_t si = static_cast<size_t>(idx);
        const float d = q.distances[k];
        steps[si] += 1;
        if (!(d > args.eps) || !std::isfinite(d)) {
          out.cache_points[si] = q.closest_points[k];
          out.cache_is_boundary[si] = 1;
          out.cache_steps[si] = steps[si];
          out.values[si] = boundary_value(q.closest_points[k]);
          out.boundary_count += 1;
        } else {
          x[si] += (args.safety * d) * random_unit_vector(rng);
          if (steps[si] >= args.m) {
            out.cache_points[si] = x[si];
            out.cache_is_boundary[si] = 0;
            out.cache_steps[si] = steps[si];
            out.values[si] = cache_value(x[si], args);
          } else {
            next.push_back(idx);
          }
        }
      }
      active.swap(next);
    }
  }
  const auto t1 = Clock::now();
  out.total_usec = std::chrono::duration<double, std::micro>(t1 - t0).count();
  out.avg_steps = std::accumulate(steps.begin(), steps.end(), 0.0) / static_cast<double>(steps.size());
  out.max_observed_steps = max_step(steps);
  return out;
}

template <class QueryFn>
ResidualResult run_residual(const Args& args, QueryFn&& query_fn) {
  ResidualResult out;
  out.n = args.n_residual;
  out.m = args.m;
  out.max_steps = args.max_steps;
  out.w.assign(static_cast<size_t>(out.n), 0.0);
  out.c.assign(static_cast<size_t>(out.n), 0.0);
  out.r.assign(static_cast<size_t>(out.n), 0.0);
  out.cache_points.assign(static_cast<size_t>(out.n), start_point(args));
  out.cache_is_boundary.assign(static_cast<size_t>(out.n), 0);
  out.cache_steps.assign(static_cast<size_t>(out.n), 0);
  std::vector<fcpw::Vector3> x(static_cast<size_t>(out.n), start_point(args));
  std::vector<int> steps(static_cast<size_t>(out.n), 0), active;
  std::vector<unsigned char> captured(static_cast<size_t>(out.n), 0);
  active.reserve(static_cast<size_t>(out.n));
  for (int i = 0; i < out.n; ++i) active.push_back(i);
  std::mt19937 rng(args.seed + 303u);
  const auto t0 = Clock::now();
  if (args.m == 0) {
    for (int i = 0; i < out.n; ++i) {
      const size_t si = static_cast<size_t>(i);
      out.cache_points[si] = x[si];
      out.cache_is_boundary[si] = 0;
      out.cache_steps[si] = 0;
      out.c[si] = cache_value(x[si], args);
      captured[si] = 1;
    }
  }
  for (int step = 0; step < args.max_steps && !active.empty(); ++step) {
    out.active_counts.push_back(static_cast<int>(active.size()));
    out.query_count += static_cast<std::uint64_t>(active.size());
    std::vector<fcpw::Vector3> batch;
    batch.reserve(active.size());
    for (int idx : active) batch.push_back(x[static_cast<size_t>(idx)]);
    QueryBatch q = query_fn(batch);
    if (q.distances.size() != batch.size()) die("residual query returned wrong size");
    std::vector<int> next;
    next.reserve(active.size());
    for (size_t k = 0; k < active.size(); ++k) {
      const int idx = active[k];
      const size_t si = static_cast<size_t>(idx);
      const float d = q.distances[k];
      steps[si] += 1;
      if (!(d > args.eps) || !std::isfinite(d)) {
        out.w[si] = boundary_value(q.closest_points[k]);
        if (!captured[si]) {
          out.cache_points[si] = q.closest_points[k];
          out.cache_is_boundary[si] = 1;
          out.cache_steps[si] = steps[si];
          out.c[si] = out.w[si];
          captured[si] = 1;
        }
        out.boundary_count += 1;
      } else {
        x[si] += (args.safety * d) * random_unit_vector(rng);
        if (!captured[si] && steps[si] >= args.m) {
          out.cache_points[si] = x[si];
          out.cache_is_boundary[si] = 0;
          out.cache_steps[si] = steps[si];
          out.c[si] = cache_value(x[si], args);
          captured[si] = 1;
        }
        next.push_back(idx);
      }
    }
    active.swap(next);
  }
  for (int idx : active) {
    const size_t si = static_cast<size_t>(idx);
    out.w[si] = target_value(x[si]);
    if (!captured[si]) {
      out.cache_points[si] = x[si];
      out.cache_is_boundary[si] = 0;
      out.cache_steps[si] = steps[si];
      out.c[si] = cache_value(x[si], args);
      captured[si] = 1;
    }
    out.truncated_count += 1;
  }
  for (int i = 0; i < out.n; ++i) {
    const size_t si = static_cast<size_t>(i);
    out.r[si] = out.w[si] - out.c[si];
  }
  const auto t1 = Clock::now();
  out.total_usec = std::chrono::duration<double, std::micro>(t1 - t0).count();
  out.avg_steps = std::accumulate(steps.begin(), steps.end(), 0.0) / static_cast<double>(steps.size());
  out.max_observed_steps = max_step(steps);
  return out;
}

void emit_stats(std::ostringstream& o, const Stats& s, int indent) {
  const std::string sp(static_cast<size_t>(indent), ' ');
  o << sp << "\"n\": " << s.n << ",\n";
  o << sp << "\"mean\": " << s.mean << ",\n";
  o << sp << "\"variance\": " << s.variance << ",\n";
  o << sp << "\"std_error\": " << s.std_error << ",\n";
  o << sp << "\"bias\": " << s.bias << ",\n";
  o << sp << "\"mse_model\": " << s.mse_model << "\n";
}

void emit_active(std::ostringstream& o, const std::vector<int>& a) {
  o << "[";
  const size_t n = std::min<size_t>(a.size(), 128);
  for (size_t i = 0; i < n; ++i) {
    if (i) o << ", ";
    o << a[i];
  }
  if (a.size() > n) o << ", \"truncated_after_128_entries\"";
  o << "]";
}

void emit_run(std::ostringstream& o, const RunResult* r, const Stats* s, int indent) {
  const std::string sp(static_cast<size_t>(indent), ' ');
  if (!r || !s) { o << "null"; return; }
  const double usec_per_sample = r->n > 0 ? r->total_usec / static_cast<double>(r->n) : 0.0;
  const double usec_per_query = r->query_count > 0 ? r->total_usec / static_cast<double>(r->query_count) : 0.0;
  o << "{\n";
  o << sp << "  \"name\": \"" << r->name << "\",\n";
  o << sp << "  \"m\": " << r->m << ",\n";
  o << sp << "  \"max_steps\": " << r->max_steps << ",\n";
  o << sp << "  \"total_usec\": " << r->total_usec << ",\n";
  o << sp << "  \"usec_per_sample\": " << usec_per_sample << ",\n";
  o << sp << "  \"usec_per_query\": " << usec_per_query << ",\n";
  o << sp << "  \"query_count\": " << r->query_count << ",\n";
  o << sp << "  \"queries_per_sample\": " << (r->n > 0 ? static_cast<double>(r->query_count) / static_cast<double>(r->n) : 0.0) << ",\n";
  o << sp << "  \"avg_steps\": " << r->avg_steps << ",\n";
  o << sp << "  \"max_observed_steps\": " << r->max_observed_steps << ",\n";
  o << sp << "  \"boundary_count\": " << r->boundary_count << ",\n";
  o << sp << "  \"truncated_count\": " << r->truncated_count << ",\n";
  o << sp << "  \"checksum\": " << checksum(r->values) << ",\n";
  o << sp << "  \"stats\": {\n";
  emit_stats(o, *s, indent + 4);
  o << sp << "  },\n";
  o << sp << "  \"active_count_by_step\": ";
  emit_active(o, r->active_counts);
  o << "\n" << sp << "}";
}

void emit_residual(std::ostringstream& o, const ResidualResult* r, const Stats* ws, const Stats* cs, const Stats* rs, int indent) {
  const std::string sp(static_cast<size_t>(indent), ' ');
  if (!r || !ws || !cs || !rs) { o << "null"; return; }
  const double usec_per_sample = r->n > 0 ? r->total_usec / static_cast<double>(r->n) : 0.0;
  const double usec_per_query = r->query_count > 0 ? r->total_usec / static_cast<double>(r->query_count) : 0.0;
  o << "{\n";
  o << sp << "  \"name\": \"coupled_residual\",\n";
  o << sp << "  \"m\": " << r->m << ",\n";
  o << sp << "  \"max_steps\": " << r->max_steps << ",\n";
  o << sp << "  \"total_usec\": " << r->total_usec << ",\n";
  o << sp << "  \"usec_per_sample\": " << usec_per_sample << ",\n";
  o << sp << "  \"usec_per_query\": " << usec_per_query << ",\n";
  o << sp << "  \"query_count\": " << r->query_count << ",\n";
  o << sp << "  \"queries_per_sample\": " << (r->n > 0 ? static_cast<double>(r->query_count) / static_cast<double>(r->n) : 0.0) << ",\n";
  o << sp << "  \"avg_steps\": " << r->avg_steps << ",\n";
  o << sp << "  \"max_observed_steps\": " << r->max_observed_steps << ",\n";
  o << sp << "  \"boundary_count\": " << r->boundary_count << ",\n";
  o << sp << "  \"truncated_count\": " << r->truncated_count << ",\n";
  o << sp << "  \"checksum\": " << checksum(r->r) << ",\n";
  o << sp << "  \"w_stats\": {\n"; emit_stats(o, *ws, indent + 4); o << sp << "  },\n";
  o << sp << "  \"c_stats\": {\n"; emit_stats(o, *cs, indent + 4); o << sp << "  },\n";
  o << sp << "  \"residual_stats\": {\n"; emit_stats(o, *rs, indent + 4); o << sp << "  },\n";
  o << sp << "  \"active_count_by_step\": "; emit_active(o, r->active_counts); o << "\n" << sp << "}";
}

std::string make_json(const Args& args, const Mesh& mesh, double exact,
                      const RunResult* pure, const Stats* ps,
                      const RunResult* coarse, const Stats* cs,
                      const ResidualResult* residual, const Stats* rws, const Stats* rcs, const Stats* rs) {
  const double pure_cost = pure && ps && pure->n > 0 ? ps->variance * (pure->total_usec / static_cast<double>(pure->n)) : 0.0;
  const double coarse_cost = coarse && cs && coarse->n > 0 ? cs->variance * (coarse->total_usec / static_cast<double>(coarse->n)) : 0.0;
  const double residual_cost = residual && rs && residual->n > 0 ? rs->variance * (residual->total_usec / static_cast<double>(residual->n)) : 0.0;
  const double opt_score = (coarse && cs && residual && rs) ? std::pow(std::sqrt(std::max(0.0, coarse_cost)) + std::sqrt(std::max(0.0, residual_cost)), 2.0) : 0.0;
  const double tl_mean = (cs && rs) ? cs->mean + rs->mean : 0.0;
  const double tl_bias = tl_mean - exact;
  const double tl_var = (coarse && cs && residual && rs) ? cs->variance / coarse->n + rs->variance / residual->n : 0.0;
  const double tl_mse = tl_bias * tl_bias + tl_var;
  const double tl_usec = (coarse ? coarse->total_usec : 0.0) + (residual ? residual->total_usec : 0.0);
  std::ostringstream o;
  o << std::setprecision(10);
  o << "{\n";
  o << "  \"program\": \"n2wos_eval_2lmc\",\n";
  o << "  \"backend\": \"" << json_escape(args.backend) << "\",\n";
  o << "  \"cache\": \"" << json_escape(args.cache) << "\",\n";
  o << "  \"compiled\": {\"N2WOS_ENABLE_FCPW_GPU\": " << (N2WOS_ENABLE_FCPW_GPU ? "true" : "false") << ", \"N2WOS_ENABLE_TCNN\": " << (N2WOS_ENABLE_TCNN ? "true" : "false") << "},\n";
  o << "  \"args\": {\"m\": " << args.m << ", \"max_steps\": " << args.max_steps << ", \"n_pure\": " << args.n_pure << ", \"n_coarse\": " << args.n_coarse << ", \"n_residual\": " << args.n_residual << ", \"warmup_queries\": " << args.warmup_queries << ", \"eps\": " << args.eps << ", \"safety\": " << args.safety << ", \"seed\": " << args.seed << ", \"x0\": [" << args.x0 << ", " << args.y0 << ", " << args.z0 << "], \"cache_scale\": " << args.cache_scale << ", \"cache_bias\": " << args.cache_bias << "},\n";
  o << "  \"target\": {\"name\": \"harmonic_x2_minus_y2\", \"exact_value_at_x0\": " << exact << "},\n";
  o << "  \"mesh\": {\"type\": \"procedural_bumpy_sphere\", \"vertices\": " << mesh.positions.size() << ", \"triangles\": " << mesh.indices.size() << ", \"lat_segments\": " << args.lat_segments << ", \"lon_segments\": " << args.lon_segments << ", \"radius\": " << args.radius << ", \"bump_amplitude\": " << args.bump_amplitude << ", \"bump_frequency\": " << args.bump_frequency << "},\n";
  o << "  \"environment\": {\"CUDA_HOME\": \"" << json_escape(env_value("CUDA_HOME")) << "\", \"CUDA_PATH\": \"" << json_escape(env_value("CUDA_PATH")) << "\", \"LD_LIBRARY_PATH\": \"" << json_escape(env_value("LD_LIBRARY_PATH")) << "\"},\n";
  o << "  \"estimators\": {\n";
  o << "    \"pure_wos\": "; emit_run(o, pure, ps, 4); o << ",\n";
  o << "    \"coarse\": "; emit_run(o, coarse, cs, 4); o << ",\n";
  o << "    \"coupled_residual\": "; emit_residual(o, residual, rws, rcs, rs, 4); o << "\n";
  o << "  },\n";
  o << "  \"two_level\": {\n";
  o << "    \"mean\": " << tl_mean << ",\n";
  o << "    \"bias\": " << tl_bias << ",\n";
  o << "    \"variance_model_current_allocation\": " << tl_var << ",\n";
  o << "    \"mse_model_current_allocation\": " << tl_mse << ",\n";
  o << "    \"total_usec_current_allocation\": " << tl_usec << ",\n";
  o << "    \"mse_time_current_allocation\": " << tl_mse * tl_usec << ",\n";
  o << "    \"pure_var_cost_score\": " << pure_cost << ",\n";
  o << "    \"coarse_var_cost_score\": " << coarse_cost << ",\n";
  o << "    \"residual_var_cost_score\": " << residual_cost << ",\n";
  o << "    \"two_level_optimal_var_cost_score\": " << opt_score << ",\n";
  o << "    \"speedup_score_vs_pure\": " << (opt_score > 0.0 ? pure_cost / opt_score : 0.0) << "\n";
  o << "  }\n";
  o << "}\n";
  return o.str();
}

void write_json(const std::string& json, const std::string& path) {
  if (!path.empty()) {
    std::ofstream out(path);
    if (!out) die("failed to open JSON output: " + path);
    out << json;
  }
  std::cout << json;
}

void write_sample_dump(const Args& args, const RunResult* coarse, const ResidualResult* residual) {
  if (args.dump_samples_path.empty()) return;
  std::ofstream out(args.dump_samples_path);
  if (!out) die("failed to open sample dump: " + args.dump_samples_path);
  out << std::setprecision(10);
  out << "kind,index,x,y,z,is_boundary,cache_steps,c_exact,w\n";
  if (coarse) {
    for (int i = 0; i < coarse->n; ++i) {
      const size_t si = static_cast<size_t>(i);
      const auto& p = coarse->cache_points[si];
      out << "coarse," << i << "," << p[0] << "," << p[1] << "," << p[2] << ","
          << coarse->cache_is_boundary[si] << "," << coarse->cache_steps[si] << ","
          << coarse->values[si] << ",\n";
    }
  }
  if (residual) {
    for (int i = 0; i < residual->n; ++i) {
      const size_t si = static_cast<size_t>(i);
      const auto& p = residual->cache_points[si];
      out << "residual," << i << "," << p[0] << "," << p[1] << "," << p[2] << ","
          << residual->cache_is_boundary[si] << "," << residual->cache_steps[si] << ","
          << residual->c[si] << "," << residual->w[si] << "\n";
    }
  }
}

template <class QueryFn>
std::string evaluate(const Args& args, const Mesh& mesh, QueryFn&& query_fn) {
  warmup(args, query_fn);
  const double exact = n2wos_native::harmonic_x2_minus_y2(args.x0, args.y0, args.z0);
  RunResult pure, coarse;
  ResidualResult residual;
  Stats ps, cs, rws, rcs, rs;
  RunResult *pp = nullptr, *cp = nullptr;
  ResidualResult* rp = nullptr;
  Stats *psp = nullptr, *csp = nullptr, *rwsp = nullptr, *rcsp = nullptr, *rsp = nullptr;
  if (!args.skip_pure) {
    std::cerr << "Running pure WoS n=" << args.n_pure << "\n";
    pure = run_pure(args, query_fn);
    ps = summarize(pure.values, exact);
    pp = &pure; psp = &ps;
  }
  if (!args.skip_coarse) {
    std::cerr << "Running coarse C_m n=" << args.n_coarse << " m=" << args.m << " cache=" << args.cache << "\n";
    coarse = run_coarse(args, query_fn);
    cs = summarize(coarse.values, exact);
    cp = &coarse; csp = &cs;
  }
  if (!args.skip_residual) {
    std::cerr << "Running coupled residual n=" << args.n_residual << " m=" << args.m << " cache=" << args.cache << "\n";
    residual = run_residual(args, query_fn);
    rws = summarize(residual.w, exact);
    rcs = summarize(residual.c, exact);
    rs = summarize(residual.r, 0.0);
    rp = &residual; rwsp = &rws; rcsp = &rcs; rsp = &rs;
  }
  write_sample_dump(args, cp, rp);
  return make_json(args, mesh, exact, pp, psp, cp, csp, rp, rwsp, rcsp, rsp);
}

} // namespace n2wos

int main(int argc, char** argv) {
  try {
    using namespace n2wos;
    Args args = parse_args(argc, argv);
#if !N2WOS_ENABLE_FCPW_GPU
    if (args.backend != "cpu") die("This binary was built without N2WOS_ENABLE_FCPW_GPU");
#endif
    Mesh mesh = make_bumpy_sphere(args);
    std::cerr << "Generated mesh: " << mesh.positions.size() << " vertices, " << mesh.indices.size() << " triangles\n";
    const bool use_gpu = args.backend != "cpu";
    fcpw::Scene<3> scene;
    build_scene(mesh, !use_gpu, scene);
    std::cerr << "Built FCPW CPU scene with vectorized_bvh=" << (!use_gpu ? "true" : "false") << "\n";
    std::string json;
    if (!use_gpu) {
      auto query_fn = [&](const std::vector<fcpw::Vector3>& pts) { return query_cpu(scene, pts); };
      json = evaluate(args, mesh, query_fn);
    } else {
#if N2WOS_ENABLE_FCPW_GPU
      std::cerr << "Creating FCPW GPUScene, fcpw_dir=" << args.fcpw_dir << ", backend=" << args.backend << "\n";
      fcpw::GPUScene<3> gpu_scene(args.fcpw_dir, args.print_logs);
      gpu_scene.transferToGPU(scene, args.backend);
      auto query_fn = [&](const std::vector<fcpw::Vector3>& pts) { return query_gpu(gpu_scene, pts); };
      json = evaluate(args, mesh, query_fn);
#else
      die("GPU path was not compiled");
#endif
    }
    write_json(json, args.json_path);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
