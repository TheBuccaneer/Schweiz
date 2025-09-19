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
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <memory>

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
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                return false;
            }
        }

        // Validate required arguments
        if (dtype.empty() || n == 0 || input_file.empty()) {
            std::cerr << "Missing required arguments. Usage:" << std::endl;
            std::cerr << "  --dtype=float32 --n=SIZE --in=FILE.bin [--device=0]" << std::endl;
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

    // CUB DeviceReduce::Sum - Two-phase call
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Phase 1: Determine temporary storage requirements
    CUB_CHECK(cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        config.n
    ));

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Phase 2: Run the actual reduction
    CUB_CHECK(cub::DeviceReduce::Sum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        config.n
    ));

    // Copy result back to host
    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Output result (ONLY this line for Python parser)
    std::cout << "sum=" << std::scientific << result << std::endl;

    // Cleanup
    cudaFree(d_temp_storage);
    cudaFree(d_out);
    cudaFree(d_in);

    return EXIT_SUCCESS;
}
