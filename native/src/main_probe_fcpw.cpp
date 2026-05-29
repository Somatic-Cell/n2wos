#include <fcpw/fcpw.h>
#if N2WOS_ENABLE_FCPW_GPU
#include <fcpw/fcpw_gpu.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace n2wos {

using Clock = std::chrono::steady_clock;

struct Args {
    std::string backend = "cpu";       // cpu, cuda, vulkan, default
    std::string mode = "cpq";          // env, cpq, wos, both
    std::string json_path;
    std::string fcpw_dir = N2WOS_FCPW_SOURCE_DIR;
    int lat_segments = 128;
    int lon_segments = 256;
    float radius = 1.0f;
    float bump_amplitude = 0.08f;
    float bump_frequency = 7.0f;
    int n_queries = 1 << 20;
    int n_samples = 1 << 16;
    int max_steps = 512;
    int repeats = 5;
    float eps = 1.0e-4f;
    float safety = 0.99f;
    uint32_t seed = 12345u;
    bool print_logs = false;
};

struct Mesh {
    std::vector<fcpw::Vector3> positions;
    std::vector<fcpw::Vector3i> indices;
};

struct QueryBatch {
    std::vector<fcpw::Vector3> closest_points;
    std::vector<float> distances;
};

struct ProfileResult {
    std::string backend;
    std::string mode;
    int n_queries = 0;
    int n_samples = 0;
    int repeats = 0;
    double total_usec = 0.0;
    double usec_per_query = 0.0;
    double usec_per_sample = 0.0;
    double usec_per_wos_query = 0.0;
    double mean_usec_per_query = 0.0;
    double median_usec_per_query = 0.0;
    double min_usec_per_query = 0.0;
    double max_usec_per_query = 0.0;
    double avg_steps = 0.0;
    int p95_steps = 0;
    int p99_steps = 0;
    int max_steps = 0;
    int max_steps_limit = 0;
    int active_remaining = 0;
    std::uint64_t query_count = 0;
    double checksum = 0.0;
    std::vector<std::uint64_t> active_count_by_step;
};

[[noreturn]] void die(const std::string& msg) {
    throw std::runtime_error(msg);
}

bool is_gpu_backend(const std::string& backend) {
    return backend == "cuda" || backend == "vulkan" || backend == "default" || backend == "d3d12";
}

std::string env_value(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
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
                    o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                      << std::dec << std::setfill(' ');
                } else {
                    o << static_cast<char>(c);
                }
        }
    }
    return o.str();
}

std::string next_value(int& i, int argc, char** argv, const char* name) {
    if (i + 1 >= argc) die(std::string("missing value for ") + name);
    return argv[++i];
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--backend") a.backend = next_value(i, argc, argv, "--backend");
        else if (key == "--mode") a.mode = next_value(i, argc, argv, "--mode");
        else if (key == "--json") a.json_path = next_value(i, argc, argv, "--json");
        else if (key == "--fcpw-dir" || key == "--fcpw_dir") a.fcpw_dir = next_value(i, argc, argv, "--fcpw-dir");
        else if (key == "--lat-segments" || key == "--lat_segments") a.lat_segments = std::stoi(next_value(i, argc, argv, "--lat-segments"));
        else if (key == "--lon-segments" || key == "--lon_segments") a.lon_segments = std::stoi(next_value(i, argc, argv, "--lon-segments"));
        else if (key == "--radius") a.radius = std::stof(next_value(i, argc, argv, "--radius"));
        else if (key == "--bump-amplitude" || key == "--bump_amplitude") a.bump_amplitude = std::stof(next_value(i, argc, argv, "--bump-amplitude"));
        else if (key == "--bump-frequency" || key == "--bump_frequency") a.bump_frequency = std::stof(next_value(i, argc, argv, "--bump-frequency"));
        else if (key == "--n-queries" || key == "--n_queries") a.n_queries = std::stoi(next_value(i, argc, argv, "--n-queries"));
        else if (key == "--n-samples" || key == "--n_samples") a.n_samples = std::stoi(next_value(i, argc, argv, "--n-samples"));
        else if (key == "--max-steps" || key == "--max_steps") a.max_steps = std::stoi(next_value(i, argc, argv, "--max-steps"));
        else if (key == "--repeats") a.repeats = std::stoi(next_value(i, argc, argv, "--repeats"));
        else if (key == "--eps" || key == "--epsilon") a.eps = std::stof(next_value(i, argc, argv, "--eps"));
        else if (key == "--safety") a.safety = std::stof(next_value(i, argc, argv, "--safety"));
        else if (key == "--seed") a.seed = static_cast<uint32_t>(std::stoul(next_value(i, argc, argv, "--seed")));
        else if (key == "--print-logs" || key == "--print_logs") a.print_logs = true;
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: n2wos_probe_fcpw [options]\n\n"
                << "  --backend cpu|cuda|vulkan|default   Backend to use. cpu bypasses GPUScene.\n"
                << "  --mode env|cpq|wos|both             Benchmark mode.\n"
                << "  --json PATH                         Write JSON to PATH.\n"
                << "  --fcpw-dir PATH                     FCPW source directory passed to GPUScene.\n"
                << "  --n-queries N                       Closest-point queries for cpq mode.\n"
                << "  --n-samples N                       Starting particles for wos mode.\n"
                << "  --max-steps N                       WoS-style maximum steps.\n"
                << "  --repeats N                         CPQ timing repeats after one warmup.\n"
                << "  --eps X                             Boundary epsilon for wos mode.\n"
                << "  --safety X                          Step length multiplier for wos mode.\n"
                << "  --lat-segments N --lon-segments N   Procedural mesh resolution.\n"
                << "  --print-logs                        Enable FCPW GPUScene logs where supported.\n";
            std::exit(0);
        } else {
            die("unknown argument: " + key);
        }
    }

    if (a.backend != "cpu" && !is_gpu_backend(a.backend)) {
        die("backend must be one of: cpu, cuda, vulkan, default, d3d12");
    }
    if (a.mode != "env" && a.mode != "cpq" && a.mode != "wos" && a.mode != "both") {
        die("mode must be one of: env, cpq, wos, both");
    }
    if (a.lat_segments < 8 || a.lon_segments < 16) die("mesh resolution too small");
    if (a.n_queries <= 0 || a.n_samples <= 0) die("n_queries and n_samples must be positive");
    if (a.repeats <= 0) die("repeats must be positive");
    if (!(a.eps > 0.0f)) die("eps must be positive");
    if (!(a.safety > 0.0f && a.safety <= 1.0f)) die("safety must be in (0, 1]");
    return a;
}

float bumpy_radius(float theta, float phi, const Args& args) {
    const float s = std::sin(theta);
    const float bump =
        0.55f * std::sin(args.bump_frequency * theta + 0.31f) *
                std::sin(args.bump_frequency * phi + 0.17f) * s * s +
        0.30f * std::cos(1.7f * args.bump_frequency * phi + 0.73f) * s * s +
        0.15f * std::sin(2.3f * args.bump_frequency * theta + 0.41f) * s;
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

    auto ring_index = [&](int i, int j) -> int {
        return 1 + (i - 1) * lon + ((j % lon + lon) % lon);
    };

    for (int i = 1; i <= lat - 1; ++i) {
        const float theta = pi * static_cast<float>(i) / static_cast<float>(lat);
        for (int j = 0; j < lon; ++j) {
            const float phi = 2.0f * pi * static_cast<float>(j) / static_cast<float>(lon);
            const float r = bumpy_radius(theta, phi, args);
            mesh.positions.emplace_back(
                r * std::sin(theta) * std::cos(phi),
                r * std::sin(theta) * std::sin(phi),
                r * std::cos(theta));
        }
    }

    const int bottom = static_cast<int>(mesh.positions.size());
    mesh.positions.emplace_back(0.0f, 0.0f, -bumpy_radius(pi, 0.0f, args));

    for (int j = 0; j < lon; ++j) {
        const int jn = (j + 1) % lon;
        mesh.indices.emplace_back(0, ring_index(1, j), ring_index(1, jn));
    }

    for (int i = 1; i <= lat - 2; ++i) {
        for (int j = 0; j < lon; ++j) {
            const int jn = (j + 1) % lon;
            const int a = ring_index(i, j);
            const int b = ring_index(i + 1, j);
            const int c = ring_index(i + 1, jn);
            const int d = ring_index(i, jn);
            mesh.indices.emplace_back(a, b, c);
            mesh.indices.emplace_back(a, c, d);
        }
    }

    for (int j = 0; j < lon; ++j) {
        const int jn = (j + 1) % lon;
        mesh.indices.emplace_back(ring_index(lat - 1, jn), ring_index(lat - 1, j), bottom);
    }

    return mesh;
}

void build_scene(const Mesh& mesh, bool vectorized_cpu_bvh, fcpw::Scene<3>& scene) {
    scene.setObjectCount(1);
    scene.setObjectVertices(mesh.positions, 0);
    scene.setObjectTriangles(mesh.indices, 0);
    const bool print_stats = false;
    const bool reduce_memory_footprint = false;
    scene.build(fcpw::AggregateType::Bvh_SurfaceArea,
                vectorized_cpu_bvh,
                print_stats,
                reduce_memory_footprint);
}

std::vector<fcpw::Vector3> make_query_points(int n, float extent, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(-extent, extent);
    std::vector<fcpw::Vector3> q;
    q.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        q.emplace_back(uni(rng), uni(rng), uni(rng));
    }
    return q;
}

std::vector<fcpw::Vector3> make_start_points(int n, float radius, std::mt19937& rng) {
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<fcpw::Vector3> x;
    x.reserve(static_cast<std::size_t>(n));
    const float scale = 0.20f * radius;
    while (static_cast<int>(x.size()) < n) {
        const float px = uni(rng);
        const float py = uni(rng);
        const float pz = uni(rng);
        const float r2 = px * px + py * py + pz * pz;
        if (r2 <= 1.0f && r2 > 1.0e-12f) {
            x.emplace_back(scale * px, scale * py, scale * pz);
        }
    }
    return x;
}

fcpw::Vector3 random_unit_vector(std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float x = normal(rng);
    const float y = normal(rng);
    const float z = normal(rng);
    const float inv = 1.0f / std::sqrt(std::max(1.0e-30f, x * x + y * y + z * z));
    return fcpw::Vector3(x * inv, y * inv, z * inv);
}

QueryBatch query_cpu(fcpw::Scene<3>& scene, const std::vector<fcpw::Vector3>& points) {
    std::vector<fcpw::BoundingSphere<3>> spheres;
    spheres.reserve(points.size());
    for (const auto& p : points) {
        spheres.emplace_back(p, fcpw::maxFloat);
    }

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
    for (const auto& p : points) {
        float3 q = float3{p[0], p[1], p[2]};
        spheres.emplace_back(q, maxFloat);
    }

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

double checksum_batch(const QueryBatch& out) {
    double s = 0.0;
    const std::size_t n = std::min<std::size_t>(out.distances.size(), 64);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& p = out.closest_points[i];
        s += static_cast<double>(out.distances[i]) + 1.0e-3 * (p[0] + 2.0 * p[1] + 3.0 * p[2]);
    }
    return s;
}

template <class QueryFn>
ProfileResult profile_cpq(const Args& args, QueryFn&& query_fn) {
    std::mt19937 rng(args.seed);
    auto points = make_query_points(args.n_queries, 1.35f * args.radius, rng);

    const int warmup_n = std::min<int>(static_cast<int>(points.size()), 4096);
    if (warmup_n > 0) {
        std::vector<fcpw::Vector3> warmup(points.begin(), points.begin() + warmup_n);
        QueryBatch warm = query_fn(warmup);
        if (static_cast<int>(warm.distances.size()) != warmup_n) die("warmup query returned wrong size");
    }

    std::vector<double> usec_per_query;
    usec_per_query.reserve(static_cast<std::size_t>(args.repeats));
    double checksum = 0.0;
    double total_usec = 0.0;

    for (int r = 0; r < args.repeats; ++r) {
        const auto t0 = Clock::now();
        QueryBatch out = query_fn(points);
        const auto t1 = Clock::now();
        if (static_cast<int>(out.distances.size()) != args.n_queries) die("FCPW returned an unexpected number of CPQ interactions");
        const double usec = std::chrono::duration<double, std::micro>(t1 - t0).count();
        total_usec += usec;
        usec_per_query.push_back(usec / static_cast<double>(args.n_queries));
        checksum += checksum_batch(out);
    }

    std::vector<double> sorted = usec_per_query;
    std::sort(sorted.begin(), sorted.end());
    const double sum = std::accumulate(usec_per_query.begin(), usec_per_query.end(), 0.0);

    ProfileResult r;
    r.backend = args.backend;
    r.mode = "cpq";
    r.n_queries = args.n_queries;
    r.repeats = args.repeats;
    r.total_usec = total_usec;
    r.usec_per_query = sorted[sorted.size() / 2];
    r.mean_usec_per_query = sum / static_cast<double>(usec_per_query.size());
    r.median_usec_per_query = sorted[sorted.size() / 2];
    r.min_usec_per_query = sorted.front();
    r.max_usec_per_query = sorted.back();
    r.query_count = static_cast<std::uint64_t>(args.n_queries) * static_cast<std::uint64_t>(args.repeats);
    r.checksum = checksum;
    return r;
}

int percentile_steps(std::vector<int> values, double p) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const double pos = p * static_cast<double>(values.size() - 1);
    const auto idx = static_cast<std::size_t>(std::lround(pos));
    return values[std::min<std::size_t>(idx, values.size() - 1)];
}

template <class QueryFn>
ProfileResult profile_wos(const Args& args, QueryFn&& query_fn) {
    std::mt19937 rng(args.seed + 17u);
    std::vector<fcpw::Vector3> x = make_start_points(args.n_samples, args.radius, rng);
    std::vector<int> steps(args.n_samples, 0);
    std::vector<int> active;
    active.reserve(static_cast<std::size_t>(args.n_samples));
    for (int i = 0; i < args.n_samples; ++i) active.push_back(i);

    std::vector<std::uint64_t> active_count_by_step;
    active_count_by_step.reserve(static_cast<std::size_t>(args.max_steps));
    std::uint64_t query_count = 0;
    double checksum = 0.0;

    const auto t0 = Clock::now();
    for (int step = 0; step < args.max_steps && !active.empty(); ++step) {
        active_count_by_step.push_back(static_cast<std::uint64_t>(active.size()));
        query_count += static_cast<std::uint64_t>(active.size());

        std::vector<fcpw::Vector3> batch;
        batch.reserve(active.size());
        for (int idx : active) batch.push_back(x[static_cast<std::size_t>(idx)]);

        QueryBatch q = query_fn(batch);
        if (q.distances.size() != batch.size()) die("FCPW returned an unexpected number of WoS interactions");
        checksum += checksum_batch(q);

        std::vector<int> next_active;
        next_active.reserve(active.size());
        for (std::size_t k = 0; k < active.size(); ++k) {
            const int idx = active[k];
            const float d = q.distances[k];
            steps[static_cast<std::size_t>(idx)] += 1;
            if (!(d > args.eps) || !std::isfinite(d)) {
                continue;
            }
            const fcpw::Vector3 dir = random_unit_vector(rng);
            x[static_cast<std::size_t>(idx)] += (args.safety * d) * dir;
            next_active.push_back(idx);
        }
        active.swap(next_active);
    }
    const auto t1 = Clock::now();

    const double usec = std::chrono::duration<double, std::micro>(t1 - t0).count();
    const double avg = std::accumulate(steps.begin(), steps.end(), 0.0) / static_cast<double>(steps.size());
    const int max_observed = steps.empty() ? 0 : *std::max_element(steps.begin(), steps.end());

    ProfileResult r;
    r.backend = args.backend;
    r.mode = "wos";
    r.n_samples = args.n_samples;
    r.total_usec = usec;
    r.usec_per_sample = usec / static_cast<double>(args.n_samples);
    r.usec_per_wos_query = query_count == 0 ? 0.0 : usec / static_cast<double>(query_count);
    r.avg_steps = avg;
    r.p95_steps = percentile_steps(steps, 0.95);
    r.p99_steps = percentile_steps(steps, 0.99);
    r.max_steps = max_observed;
    r.max_steps_limit = args.max_steps;
    r.active_remaining = static_cast<int>(active.size());
    r.query_count = query_count;
    r.checksum = checksum;
    r.active_count_by_step = std::move(active_count_by_step);
    return r;
}

void append_env_json(std::ostringstream& o) {
    const char* keys[] = {
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "VK_ICD_FILENAMES",
        "VK_LAYER_PATH"
    };
    o << "  \"environment\": {\n";
    for (std::size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        o << "    \"" << keys[i] << "\": \"" << json_escape(env_value(keys[i])) << "\"";
        o << (i + 1 == sizeof(keys) / sizeof(keys[0]) ? "\n" : ",\n");
    }
    o << "  }";
}

void append_result_json(std::ostringstream& o, const ProfileResult& r, int indent) {
    const std::string sp(static_cast<std::size_t>(indent), ' ');
    const std::string sp2(static_cast<std::size_t>(indent + 2), ' ');
    o << sp << "{\n";
    o << sp2 << "\"backend\": \"" << json_escape(r.backend) << "\",\n";
    o << sp2 << "\"mode\": \"" << json_escape(r.mode) << "\",\n";
    o << sp2 << "\"n_queries\": " << r.n_queries << ",\n";
    o << sp2 << "\"usec/query\": " << r.usec_per_query << ",\n";
    o << sp2 << "\"usec_per_query\": " << r.usec_per_query << ",\n";
    o << sp2 << "\"mean_usec_per_query\": " << r.mean_usec_per_query << ",\n";
    o << sp2 << "\"median_usec_per_query\": " << r.median_usec_per_query << ",\n";
    o << sp2 << "\"min_usec_per_query\": " << r.min_usec_per_query << ",\n";
    o << sp2 << "\"max_usec_per_query\": " << r.max_usec_per_query << ",\n";
    o << sp2 << "\"n_samples\": " << r.n_samples << ",\n";
    o << sp2 << "\"usec/sample\": " << r.usec_per_sample << ",\n";
    o << sp2 << "\"usec_per_sample\": " << r.usec_per_sample << ",\n";
    o << sp2 << "\"usec_per_wos_query\": " << r.usec_per_wos_query << ",\n";
    o << sp2 << "\"avg_steps\": " << r.avg_steps << ",\n";
    o << sp2 << "\"p95_steps\": " << r.p95_steps << ",\n";
    o << sp2 << "\"p99_steps\": " << r.p99_steps << ",\n";
    o << sp2 << "\"max_steps\": " << r.max_steps << ",\n";
    o << sp2 << "\"max_steps_limit\": " << r.max_steps_limit << ",\n";
    o << sp2 << "\"active_remaining\": " << r.active_remaining << ",\n";
    o << sp2 << "\"query_count\": " << r.query_count << ",\n";
    o << sp2 << "\"repeats\": " << r.repeats << ",\n";
    o << sp2 << "\"total_usec\": " << r.total_usec << ",\n";
    o << sp2 << "\"checksum\": " << r.checksum << ",\n";
    o << sp2 << "\"active_count_by_step\": [";
    for (std::size_t k = 0; k < r.active_count_by_step.size(); ++k) {
        if (k) o << ", ";
        o << r.active_count_by_step[k];
    }
    o << "]\n";
    o << sp << "}";
}

std::string to_json(const std::vector<ProfileResult>& results, const Args& args, const Mesh* mesh) {
    std::ostringstream o;
    o << std::setprecision(10);
    o << "{\n";
    o << "  \"backend\": \"" << json_escape(args.backend) << "\",\n";
    o << "  \"requested_mode\": \"" << json_escape(args.mode) << "\",\n";
    o << "  \"compiled\": {\n";
    o << "    \"N2WOS_ENABLE_FCPW_GPU\": " << (N2WOS_ENABLE_FCPW_GPU ? "true" : "false") << ",\n";
    o << "    \"N2WOS_FCPW_SOURCE_DIR\": \"" << json_escape(N2WOS_FCPW_SOURCE_DIR) << "\",\n";
    o << "    \"N2WOS_CMAKE_BUILD_TYPE\": \"" << json_escape(N2WOS_CMAKE_BUILD_TYPE) << "\",\n";
    o << "    \"__cplusplus\": " << static_cast<long>(__cplusplus) << "\n";
    o << "  },\n";
    o << "  \"runtime\": {\n";
    o << "    \"fcpw_dir\": \"" << json_escape(args.fcpw_dir) << "\",\n";
    o << "    \"print_logs\": " << (args.print_logs ? "true" : "false") << "\n";
    o << "  },\n";
    if (mesh) {
        o << "  \"mesh\": {\n";
        o << "    \"type\": \"procedural_bumpy_sphere\",\n";
        o << "    \"vertices\": " << mesh->positions.size() << ",\n";
        o << "    \"triangles\": " << mesh->indices.size() << ",\n";
        o << "    \"lat_segments\": " << args.lat_segments << ",\n";
        o << "    \"lon_segments\": " << args.lon_segments << ",\n";
        o << "    \"radius\": " << args.radius << ",\n";
        o << "    \"bump_amplitude\": " << args.bump_amplitude << ",\n";
        o << "    \"bump_frequency\": " << args.bump_frequency << "\n";
        o << "  },\n";
    } else {
        o << "  \"mesh\": null,\n";
    }
    append_env_json(o);
    o << ",\n";
    o << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        append_result_json(o, results[i], 4);
        o << (i + 1 == results.size() ? "\n" : ",\n");
    }
    o << "  ]\n";
    o << "}\n";
    return o.str();
}

void write_or_print_json(const std::string& json, const std::string& path) {
    if (!path.empty()) {
        std::ofstream out(path);
        if (!out) die("failed to open JSON output path: " + path);
        out << json;
    }
    std::cout << json;
}

} // namespace n2wos

int main(int argc, char** argv) {
    try {
        using namespace n2wos;
        Args args = parse_args(argc, argv);

        if (args.mode == "env") {
            const std::vector<ProfileResult> empty;
            write_or_print_json(to_json(empty, args, nullptr), args.json_path);
            return 0;
        }

#if !N2WOS_ENABLE_FCPW_GPU
        if (args.backend != "cpu") {
            die("This binary was built without N2WOS_ENABLE_FCPW_GPU. Reconfigure with -DN2WOS_ENABLE_FCPW_GPU=ON.");
        }
#endif

        Mesh mesh = make_bumpy_sphere(args);
        std::cerr << "Generated mesh: " << mesh.positions.size() << " vertices, "
                  << mesh.indices.size() << " triangles\n";

        const bool use_gpu = (args.backend != "cpu");
        fcpw::Scene<3> scene;
        build_scene(mesh, !use_gpu, scene);
        std::cerr << "Built FCPW CPU scene with vectorized_bvh=" << (!use_gpu ? "true" : "false") << "\n";

        std::vector<ProfileResult> results;

        if (!use_gpu) {
            auto query_fn = [&](const std::vector<fcpw::Vector3>& pts) {
                return query_cpu(scene, pts);
            };
            if (args.mode == "cpq" || args.mode == "both") results.push_back(profile_cpq(args, query_fn));
            if (args.mode == "wos" || args.mode == "both") results.push_back(profile_wos(args, query_fn));
        } else {
#if N2WOS_ENABLE_FCPW_GPU
            std::cerr << "Creating FCPW GPUScene, fcpw_dir=" << args.fcpw_dir
                      << ", backend=" << args.backend
                      << ", print_logs=" << (args.print_logs ? "true" : "false") << "\n";
            fcpw::GPUScene<3> gpu_scene(args.fcpw_dir, args.print_logs);
            gpu_scene.transferToGPU(scene, args.backend);
            std::cerr << "Transferred FCPW scene to GPU backend=" << args.backend << "\n";
            auto query_fn = [&](const std::vector<fcpw::Vector3>& pts) {
                return query_gpu(gpu_scene, pts);
            };
            if (args.mode == "cpq" || args.mode == "both") results.push_back(profile_cpq(args, query_fn));
            if (args.mode == "wos" || args.mode == "both") results.push_back(profile_wos(args, query_fn));
#else
            die("GPU path was not compiled");
#endif
        }

        write_or_print_json(to_json(results, args, &mesh), args.json_path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
