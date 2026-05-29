#include <tiny-cuda-nn/cpp_api.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace n2wos {

[[noreturn]] void die(const std::string& msg) {
    throw std::runtime_error(msg);
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

const char* precision_name(tcnn::cpp::Precision p) {
    return p == tcnn::cpp::Precision::Fp32 ? "fp32" : "fp16";
}

struct Args {
    std::string config_path = "configs/cache_hashgrid_small.json";
    std::string json_path;
    bool jit = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--config") a.config_path = next_value(i, argc, argv, "--config");
        else if (key == "--json") a.json_path = next_value(i, argc, argv, "--json");
        else if (key == "--jit") a.jit = true;
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: n2wos_probe_tcnn [options]\n\n"
                << "  --config PATH   HashGrid/MLP JSON config. Default: configs/cache_hashgrid_small.json\n"
                << "  --json PATH     Write probe JSON to PATH.\n"
                << "  --jit           Request tiny-cuda-nn JIT fusion if supported.\n";
            std::exit(0);
        } else {
            die("unknown argument: " + key);
        }
    }
    return a;
}

void write_or_print(const std::string& text, const std::string& path) {
    if (!path.empty()) {
        std::ofstream out(path);
        if (!out) die("failed to open JSON output path: " + path);
        out << text;
    }
    std::cout << text;
}

} // namespace n2wos

int main(int argc, char** argv) {
    try {
        using namespace n2wos;
        const Args args = parse_args(argc, argv);

        std::ifstream in(args.config_path);
        if (!in) die("failed to open config: " + args.config_path);
        tcnn::cpp::json config = tcnn::cpp::json::parse(in, nullptr, true, true);

        const uint32_t n_input_dims = config.value("n_input_dims", 3u);
        const uint32_t n_output_dims = config.value("n_output_dims", 1u);
        const auto encoding = config.at("encoding");
        const auto network = config.at("network");

        const int device = tcnn::cpp::cuda_device();
        cudaDeviceProp prop{};
        const cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) die(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(err));

        std::unique_ptr<tcnn::cpp::Module> module{
            tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network)};
        if (!module) die("tiny-cuda-nn returned a null module");

        const bool supports_jit = tcnn::cpp::supports_jit_fusion(device);
        module->set_jit_fusion(args.jit && supports_jit);

        std::ostringstream o;
        o << std::setprecision(10);
        o << "{\n";
        o << "  \"config_path\": \"" << json_escape(args.config_path) << "\",\n";
        o << "  \"device\": {\n";
        o << "    \"index\": " << device << ",\n";
        o << "    \"name\": \"" << json_escape(prop.name) << "\",\n";
        o << "    \"compute_capability\": \"" << prop.major << "." << prop.minor << "\"\n";
        o << "  },\n";
        o << "  \"tcnn\": {\n";
        o << "    \"has_networks\": " << (tcnn::cpp::has_networks() ? "true" : "false") << ",\n";
        o << "    \"batch_size_granularity\": " << tcnn::cpp::batch_size_granularity() << ",\n";
        o << "    \"preferred_precision\": \"" << precision_name(tcnn::cpp::preferred_precision()) << "\",\n";
        o << "    \"supports_jit_fusion\": " << (supports_jit ? "true" : "false") << "\n";
        o << "  },\n";
        o << "  \"module\": {\n";
        o << "    \"name\": \"" << json_escape(module->name()) << "\",\n";
        o << "    \"n_input_dims\": " << module->n_input_dims() << ",\n";
        o << "    \"n_output_dims_padded\": " << module->n_output_dims() << ",\n";
        o << "    \"n_params\": " << module->n_params() << ",\n";
        o << "    \"param_precision\": \"" << precision_name(module->param_precision()) << "\",\n";
        o << "    \"output_precision\": \"" << precision_name(module->output_precision()) << "\",\n";
        o << "    \"jit_fusion_enabled\": " << (module->jit_fusion() ? "true" : "false") << "\n";
        o << "  }\n";
        o << "}\n";

        write_or_print(o.str(), args.json_path);
        tcnn::cpp::free_temporary_memory();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
