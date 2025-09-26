/*
 * CUB DeviceReduce::Sum Benchmark Tool
 *
 * Purpose: Provides ground truth CUB reduction for Python benchmark comparison
 * License: MIT/Public Domain
 *
 * Usage: ./cub_reduce --dtype=float32 --n=SIZE --in=FILE.bin [--device=0]
 * Output: sum=<result> (stdout only, for automated parsing)
 */

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
    std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
    << " - " << cudaGetErrorString(err) << std::endl; \
    return EXIT_FAILURE; \
} \
} while(0)

// CUB error checking (CUB returns cudaError_t)
#define CUB_CHECK(call) CUDA_CHECK(call)

struct Config {
    std::string dtype;
    size_t n = 0;
    std::string input_file;
    int device = 0;
    int passes = 1;

    bool parse_args(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);

            if (arg.find("--dtype=") == 0) {
                dtype = arg.substr(8);
            } else if (arg.find("--n=") == 0) {
                n = std::stoull(arg.substr(4));
            } else if (arg.find("--in=") == 0) {
                input_file = arg.substr(5);
            } else if (arg.find("--device=") == 0) {
                device = std::stoi(arg.substr(9));
            } else if (arg.find("--passes=") == 0) {
                passes = std::max(1, std::stoi(arg.substr(9)));
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                return false;
            }
        }

        // Validate required arguments
        if (dtype.empty() || n == 0 || input_file.empty()) {
            std::cerr << "Missing required arguments. Usage:" << std::endl;
            std::cerr << "  --dtype=float32 --n=SIZE --in=FILE.bin [--device=0] [--passes=1]" << std::endl;
            return false;
        }

        // Only support float32 for now
        if (dtype != "float32") {
            std::cerr << "Error: Only --dtype=float32 is supported" << std::endl;
            return false;
        }

        return true;
    }
};

int main(int argc, char* argv[]) {
    Config config;

    if (!config.parse_args(argc, argv)) {
        return EXIT_FAILURE;
    }

    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(config.device));

    // [B] NVML initialization early after argument parsing
    nvmlReturn_t nvres = nvmlInit_v2();
    nvmlDevice_t nvdev{};
    int cuDev = 0;
    CUDA_CHECK(cudaGetDevice(&cuDev));
    if (nvres == NVML_SUCCESS) {
        nvres = nvmlDeviceGetHandleByIndex_v2(cuDev, &nvdev);
    }

    // Read input file
    std::ifstream file(config.input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file: " << config.input_file << std::endl;
        return EXIT_FAILURE;
    }

    // Check file size
    std::streamsize file_size = file.tellg();
    std::streamsize expected_size = config.n * sizeof(float);

    if (file_size != expected_size) {
        std::cerr << "Error: File size mismatch. Expected " << expected_size
        << " bytes for " << config.n << " float32 elements, got "
        << file_size << " bytes" << std::endl;
        return EXIT_FAILURE;
    }

    file.seekg(0, std::ios::beg);

    // Allocate host memory and read data
    std::unique_ptr<float[]> h_data(new float[config.n]);
    file.read(reinterpret_cast<char*>(h_data.get()), expected_size);

    if (!file.good()) {
        std::cerr << "Error: Failed to read " << expected_size << " bytes from file" << std::endl;
        return EXIT_FAILURE;
    }
    file.close();

    // Allocate device memory
    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, config.n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_data.get(), config.n * sizeof(float), cudaMemcpyHostToDevice));

    // CUB DeviceReduce::Sum - Two-phase call setup
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Phase 1: Determine temporary storage requirements
    CUB_CHECK(cub::DeviceReduce::Sum(
        nullptr,
        temp_storage_bytes,
        d_in,
        d_out,
        config.n
    ));

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // CUDA Events for precise kernel timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Accumulator for multiple passes
    float* d_accumulator = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accumulator, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_accumulator, 0, sizeof(float))); // Initialize to 0

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    // [C] Energy measurement start (directly after cudaEventRecord(start))
    unsigned long long e_start_mJ = 0ULL;
    if (nvres == NVML_SUCCESS) {
        nvmlDeviceGetTotalEnergyConsumption(nvdev, &e_start_mJ);
    }

    // Run passes with accumulation
    for (int r = 0; r < config.passes; ++r) {
        // Phase 2: Run the actual reduction into d_out
        CUB_CHECK(cub::DeviceReduce::Sum(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            config.n
        ));

        // Accumulate result (d_accumulator += d_out[0])
        float temp_result;
        CUDA_CHECK(cudaMemcpy(&temp_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        float current_acc;
        CUDA_CHECK(cudaMemcpy(&current_acc, d_accumulator, sizeof(float), cudaMemcpyDeviceToHost));
        current_acc += temp_result;
        CUDA_CHECK(cudaMemcpy(d_accumulator, &current_acc, sizeof(float), cudaMemcpyHostToDevice));
    }

    // [D] Stop timing and energy measurement (stop + sync for exact window alignment)
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));  // ensures time window exactly encompasses kernel
    unsigned long long e_stop_mJ = 0ULL;
    if (nvres == NVML_SUCCESS) {
        nvmlDeviceGetTotalEnergyConsumption(nvdev, &e_stop_mJ);
    }

    // Get precise kernel timing
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // [E] Calculate energy/power and print
    double energy_J = (nvres == NVML_SUCCESS && e_stop_mJ >= e_start_mJ) ? (e_stop_mJ - e_start_mJ) * 1e-3 : NAN;
    double power_W = (std::isfinite(energy_J) && kernel_ms > 0.0f) ? energy_J / (kernel_ms * 1e-3) : NAN;

    std::cout << std::fixed << std::setprecision(6)
    << "energy_J=" << energy_J << " "
    << "power_W="  << std::setprecision(3) << power_W << std::endl;


    // Copy final accumulated result to host
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_accumulator, sizeof(float), cudaMemcpyDeviceToHost));

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Output results (for Python parser)
    std::cout << "sum=" << std::scientific << result << std::endl;
    std::cout << "kernel_ms=" << std::fixed << std::setprecision(3) << kernel_ms << std::endl;

    // Cleanup
    cudaFree(d_accumulator);
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    cudaFree(d_in);

    if (nvres == NVML_SUCCESS) nvmlShutdown();

    return EXIT_SUCCESS;
}
