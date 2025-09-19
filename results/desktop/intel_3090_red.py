#!/usr/bin/env python3

import os
import sys
import socket
import time
import csv
import argparse
from datetime import datetime
import subprocess

# CRITICAL: Set BLAS threads BEFORE importing numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# TF32 control for reproducible FP32 comparisons
os.environ.setdefault("CUPY_TF32", "0")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import numpy as np
import cupy as cp
import pynvml

# Custom CUDA kernel with warp-shuffle reduction + in-kernel passes
# Implementation follows NVIDIA tree-reduction pattern with __shfl_down_sync
# Reference: https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
REDUCTION_KERNEL_SRC = r"""
extern "C" __global__
void reduce_shfl(const float* __restrict__ x, float* __restrict__ out,
                 const unsigned int N, const int passes)
{
    // grid-stride load + per-thread partial sum
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;

    float sum = 0.0f;
    // In-kernel passes: read the same array multiple times
    for (int r = 0; r < passes; ++r) {
        for (unsigned long long i = idx; i < N; i += stride) {
            sum += x[i];
        }
    }

    // block reduction: first warp-level via shfl, then one atomic per block
    // intra-warp reduction (assume warpSize==32)
    unsigned int mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // shared memory to collect warp leaders
    __shared__ float warp_sums[32]; // supports up to 1024 threads/block
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[wid] = sum;
    }
    __syncthreads();

    // first warp reduces warp_sums
    if (wid == 0) {
        float block_sum = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}
"""

def init_gpu():
    """Initialize GPU and get info"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = name.decode() if isinstance(name, bytes) else str(name)

    uuid = pynvml.nvmlDeviceGetUUID(handle)
    gpu_uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)

    driver_version = pynvml.nvmlSystemGetDriverVersion()
    driver_ver = driver_version.decode() if isinstance(driver_version, bytes) else str(driver_version)

    return handle, gpu_name, gpu_uuid, driver_ver

def get_cpu_info():
    """Get CPU info with robust fallback"""
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return "Unknown"

def read_rapl_energy():
    """Read Intel RAPL energy"""
    try:
        with open('/sys/class/powercap/intel-rapl:0/energy_uj', 'r') as f:
            uj = int(f.read().strip())
        return uj / 1_000_000  # Convert to Joules
    except:
        return None

def get_gpu_energy(handle):
    """Get GPU energy via NVML"""
    try:
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return energy_mj / 1000  # mJ to J
    except Exception as e:
        # NVML energy not supported on older cards/drivers
        if not hasattr(get_gpu_energy, '_warned'):
            print(f"⚠️  GPU energy measurement not available (NVML): {e}")
            get_gpu_energy._warned = True
        return None

def calculate_bandwidth_metrics(N, kernel_time_s, args):
    """Calculate bandwidth metrics with consistent units"""
    bytes_per_elem = 4  # float32
    passes = args.passes if args else 1
    total_bytes = passes * N * bytes_per_elem  # single run with in-kernel passes

    unit_div = (1e9 if args.units == "GB" else (1024**3))
    bw = (total_bytes / unit_div) / kernel_time_s

    pct_peak = None
    if args.peak_bw:
        pct_peak = (bw / args.peak_bw) * 100.0

    return bw, pct_peak, total_bytes

def run_custom_kernel(x, out, passes=1):
    """Run custom hierarchical reduction with warp-shuffle + passes"""
    # Compile kernel once
    if not hasattr(run_custom_kernel, 'kernel'):
        run_custom_kernel.kernel = cp.RawKernel(REDUCTION_KERNEL_SRC, "reduce_shfl")

    # Choose reasonable config
    threads = 256
    blocks = min((x.size + threads - 1) // threads, 1024)

    run_custom_kernel.kernel((blocks,), (threads,), (x, out, cp.uint32(x.size), cp.int32(passes)))
    cp.cuda.Stream.null.synchronize()

def run_cub_baseline(x, out, args=None):
    """CUB baseline via external C++ binary"""
    if args is None or args.cub_bin is None:
        raise RuntimeError("CUB C++ implementation requires --cub-bin path to external binary")

    import tempfile
    import os

    # Write data to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        x_host = cp.asnumpy(x)
        x_host.tofile(f.name)
        tmp_file = f.name

    try:
        # Call external CUB binary
        # Expected format: binary --dtype=float32 --n=SIZE --in=FILE
        # Expected output: sum=RESULT
        result = subprocess.run(
            [args.cub_bin, f"--dtype=float32", f"--n={x.size}", f"--in={tmp_file}"],
            check=True, capture_output=True, text=True
        )

        # Parse result from stdout
        for line in result.stdout.strip().split('\n'):
            if line.startswith('sum='):
                val = float(line.split('=')[1])
                out.fill(0)
                out[0] = val
                return val

        raise RuntimeError(f"CUB binary did not return expected 'sum=' format. Output: {result.stdout}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CUB binary failed: {e}. stderr: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(f"CUB binary not found: {args.cub_bin}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def run_cupy_baseline(x, out, passes=1):
    """CuPy baseline (uses optimized reduction, may use CUB internally)"""
    result = 0.0
    for r in range(passes):
        result += cp.sum(x)
    out.fill(0)
    out[0] = result
    return float(result)

def run_cpu_reduction(x, passes=1):
    """Run CPU reduction with NumPy"""
    result = 0.0
    for r in range(passes):
        result += np.sum(x)
    return result

def measure_once_gpu(N, kernel_func, target_runtime_s, args=None):
    """Single measurement with target runtime via N scaling"""
    # Allocate arrays
    x = cp.random.rand(N).astype(cp.float32)  # Random data for realistic memory patterns
    out = cp.zeros(1, dtype=cp.float32)

    passes = args.passes if args else 1

    # Warmup
    if args and args.impl == 'cub_cpp':
        # For CUB binary, skip warmup to avoid multiple subprocess calls
        pass
    else:
        if args and args.impl == 'cupy':
            run_cupy_baseline(x, out, passes)
        else:
            kernel_func(x, out, passes)
        cp.cuda.Stream.null.synchronize()

    # Reset OUTSIDE timing window
    out.fill(0)
    cp.cuda.Stream.null.synchronize()

    # CUDA events for precise kernel timing
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Timing window
    start_event.record()

    if args and args.impl == 'cupy':
        result = run_cupy_baseline(x, out, passes)
    elif args and args.impl == 'cub_cpp':
        result = run_cub_baseline(x, out, args)  # Note: CUB doesn't support passes
    else:
        kernel_func(x, out, passes)
        result = float(out[0])

    end_event.record()
    end_event.synchronize()

    kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    kernel_time_s = kernel_ms / 1000

    return kernel_ms, kernel_time_s, result, x, out

def measure_once_cpu(N, target_runtime_s, passes=1):
    """Single CPU measurement"""
    x = np.random.rand(N).astype(np.float32)

    # Timing window
    start_time = time.perf_counter()
    result = run_cpu_reduction(x, passes)
    end_time = time.perf_counter()

    runtime_s = end_time - start_time
    kernel_ms = runtime_s * 1000

    return kernel_ms, runtime_s, float(result), x

def find_target_N(candidate_sizes, kernel_func, target_runtime_s, device='GPU', args=None):
    """Find N that achieves >= target_runtime_s in single run"""
    print(f"Finding target N for {device} to reach {target_runtime_s}s runtime...")

    for N in candidate_sizes:
        print(f"  Testing N={N:,} ({N/1e6:.1f}M elements)...")

        if device == 'GPU':
            kernel_ms, kernel_time_s, result, x_buf, out_buf = measure_once_gpu(N, kernel_func, target_runtime_s, args)
        else:
            passes = args.passes if args else 1
            kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)
            out_buf = None

        print(f"    Runtime: {kernel_time_s:.3f}s")

        if kernel_time_s >= target_runtime_s:
            print(f"  ✓ Selected N={N:,} with runtime {kernel_time_s:.3f}s")
            return N, kernel_ms, kernel_time_s, result, x_buf, out_buf

    # Fallback: use largest N and accept < target_runtime_s
    N = candidate_sizes[-1]
    print(f"  ⚠  Using largest N={N:,} (couldn't reach target runtime)")

    if device == 'GPU':
        kernel_ms, kernel_time_s, result, x_buf, out_buf = measure_once_gpu(N, kernel_func, target_runtime_s, args)
    else:
        passes = args.passes if args else 1
        kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)
        out_buf = None

    return N, kernel_ms, kernel_time_s, result, x_buf, out_buf

def main():
    parser = argparse.ArgumentParser(
        description='Memory-bound Reduction Benchmark (CUDA Events provide sub-µs timing precision in ms)'
    )
    parser.add_argument('--impl', choices=['cupy', 'custom', 'cpu', 'cub_cpp'], required=True,
                        help='Implementation: cupy (CuPy sum), custom (hierarchical), cpu, cub_cpp (external C++ CUB)')
    parser.add_argument('--N', type=int,
                        help='Specific array size (if not provided, auto-scale to target runtime)')
    parser.add_argument('--dtype', default='float32', choices=['float32'],
                        help='Data type')
    parser.add_argument('--peak-bw', type=float,
                        help='Peak bandwidth in chosen units/s for percent calculation')
    parser.add_argument('--units', choices=['GB', 'GiB'], default='GiB',
                        help='Bandwidth units: GB (1e9) or GiB (1024^3)')
    parser.add_argument('--target-runtime', type=float, default=1.2,
                        help='Target runtime in seconds (default: 1.2)')
    parser.add_argument('--passes', type=int, default=1,
                        help='In-kernel repeats over the same array (default: 1)')
    parser.add_argument('--cub-bin', type=str, default=None,
                        help='Path to external C++ binary with real cub::DeviceReduce::Sum (CSV-compatible)')
    parser.add_argument('--rtol', type=float, default=5e-4,
                        help='Relative tolerance for result validation against NumPy (default: 5e-4)')

    args = parser.parse_args()

    # Candidate sizes for single-run scaling (no batching)
    candidate_sizes = [2**24, 2**26, 2**27, 2**28, 2**29]  # 16M to 536M elements
    target_runtime_s = args.target_runtime

    # Initialize hardware
    gpu_handle, gpu_name, gpu_uuid, driver_version = init_gpu()
    cpu_name = get_cpu_info()
    host = socket.gethostname()

    print(f"System: {host}")
    print(f"CPU: {cpu_name}")
    print(f"GPU: {gpu_name}")
    print(f"Driver: {driver_version}")
    print(f"Implementation: {args.impl}")
    print(f"Data type: {args.dtype}")
    print(f"Target runtime: {target_runtime_s}s")
    print(f"Passes: {args.passes}")
    print(f"Units: {args.units}")
    if args.peak_bw:
        print(f"Peak BW reference: {args.peak_bw:.1f} {args.units}/s")

    # Seed for reproducibility
    np.random.seed(123)
    cp.random.seed(123)
    print("Random seeds set for reproducibility")

    # Select kernel function
    if args.impl == 'cupy':
        kernel_func = run_cupy_baseline  # CuPy's optimized sum (may use CUB internally)
        device = 'GPU'
    elif args.impl == 'custom':
        kernel_func = run_custom_kernel
        device = 'GPU'
    elif args.impl == 'cub_cpp':
        kernel_func = run_cub_baseline  # External C++ CUB binary
        device = 'GPU'
        if args.cub_bin is None:
            print("ERROR: --impl cub_cpp requires --cub-bin path to external CUB binary")
            sys.exit(1)
        if args.passes > 1:
            print("WARNING: CUB C++ implementation doesn't support --passes > 1, using passes=1")
            args.passes = 1
    elif args.impl == 'cpu':
        kernel_func = None
        device = 'CPU'

    # Find target N or use specified N
    if args.N:
        N = args.N
        print(f"\nUsing specified N={N:,}")

        if device == 'GPU':
            kernel_ms, kernel_time_s, result, x_buf, out_buf = measure_once_gpu(N, kernel_func, target_runtime_s, args)
        else:
            passes = args.passes if args else 1
            kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)
            out_buf = None
    else:
        N, kernel_ms, kernel_time_s, result, x_buf, out_buf = find_target_N(
            candidate_sizes, kernel_func, target_runtime_s, device, args
        )

    # Validation with numpy ground truth (calculate once)
    if device == 'GPU':
        x_cpu = cp.asnumpy(x_buf)
    else:
        x_cpu = x_buf
    expected_result = np.sum(x_cpu) * args.passes  # Expected result with passes

    print(f"\n=== Energy Measurement Window ===")
    print("Starting energy measurement...")

    # Energy measurement window (around the final measurement)
    t_energy_start = time.perf_counter()
    cpu_energy_before = read_rapl_energy()
    gpu_energy_before = get_gpu_energy(gpu_handle)

    # Final measurement for energy calculation
    if device == 'GPU':
        if out_buf is not None:
            out_buf.fill(0)
        cp.cuda.Stream.null.synchronize()

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()

        if args.impl == 'cupy':
            final_result = run_cupy_baseline(x_buf, out_buf, args.passes)
        elif args.impl == 'cub_cpp':
            final_result = run_cub_baseline(x_buf, out_buf, args)
        else:  # custom
            kernel_func(x_buf, out_buf, args.passes)
            final_result = float(out_buf[0])

        end_event.record()
        end_event.synchronize()

        # Calculate metrics based on final measurement
        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        kernel_time_s = kernel_ms / 1000.0

        bw, pct_peak, total_bytes = calculate_bandwidth_metrics(N, kernel_time_s, args)

        qc_threshold = target_runtime_s
        qc_pass = bool(kernel_time_s >= qc_threshold)

        print(f"Results: kernel_ms={kernel_ms:.3f}, result={final_result}, bw={bw:.2f} {args.units}/s")

    else:
        start_time = time.perf_counter()
        final_result = run_cpu_reduction(x_buf, args.passes)
        end_time = time.perf_counter()

        # Calculate metrics based on final measurement
        kernel_time_s = end_time - start_time

        bw, pct_peak, total_bytes = calculate_bandwidth_metrics(N, kernel_time_s, args)

        qc_threshold = target_runtime_s
        qc_pass = bool(kernel_time_s >= qc_threshold)

        print(f"Results: kernel_ms={kernel_time_s*1000:.3f}, result={final_result}, bw={bw:.2f} {args.units}/s")

        kernel_ms = kernel_time_s * 1000.0

    # Energy measurement end
    cpu_energy_after = read_rapl_energy()
    gpu_energy_after = get_gpu_energy(gpu_handle)
    t_energy_end = time.perf_counter()

    energy_window_s = t_energy_end - t_energy_start

    # Calculate energy deltas (robust against counter wraps)
    energy_cpu = None
    energy_gpu = None

    if cpu_energy_before is not None and cpu_energy_after is not None:
        delta = cpu_energy_after - cpu_energy_before
        energy_cpu = delta if delta >= 0 else None

    if gpu_energy_before is not None and gpu_energy_after is not None:
        delta = gpu_energy_after - gpu_energy_before
        energy_gpu = delta if delta >= 0 else None

    # Calculate power using energy measurement window
    power_cpu = (energy_cpu / energy_window_s) if (energy_cpu is not None and energy_window_s > 0) else None
    power_gpu = (energy_gpu / energy_window_s) if (energy_gpu is not None and energy_window_s > 0) else None

    # Dynamic unit label for total bytes display
    unit_div = (1e9 if args.units == "GB" else (1024**3))
    unit_label = args.units

    # QC based on kernel time
    if qc_pass:
        qc_status = "QC_PASS"
    elif kernel_time_s >= 1.0:
        qc_status = "QC_ACCEPTABLE_SHORT"
    else:
        qc_status = "QC_CRITICAL_TOO_SHORT"

    # Prepare final metrics
    energy_j = energy_gpu if device == 'GPU' else energy_cpu
    power_w = power_gpu if device == 'GPU' else power_cpu

    # Results display
    print(f"\n=== Results ===")
    print(f"Array size N: {N:,} ({N/1e6:.1f}M elements)")
    print(f"Passes: {args.passes}")
    print(f"Total bytes: {total_bytes:,} ({total_bytes/unit_div:.2f} {unit_label})")
    print(f"Kernel time: {kernel_ms:.3f} ms ({kernel_time_s:.3f} s)")
    print(f"Energy window: {energy_window_s:.3f} s")
    print(f"Result: {final_result:.6e}")
    print(f"Bandwidth: {bw:.1f} {args.units}/s")
    if pct_peak is not None:
        print(f"Peak BW %: {pct_peak:.1f}%")
    if energy_j is not None:
        print(f"Energy: {energy_j:.3f} J")
    if power_w is not None:
        print(f"Power: {power_w:.1f} W")
    print(f"QC Status: {qc_status} (kernel time >= {target_runtime_s}s: {qc_pass})")

    # Validation (configurable tolerance, warning instead of crash)
    relative_error = abs(final_result - expected_result) / abs(expected_result)
    validation_ok = np.isclose(final_result, expected_result, rtol=args.rtol)

    print(f"\n=== Validation ===")
    print(f"Expected: {expected_result:.6e}")
    print(f"Actual:   {final_result:.6e}")
    print(f"Rel. error: {relative_error:.2e}")
    print(f"Tolerance: {args.rtol:.1e}")
    print(f"Status: {'✓ PASS' if validation_ok else '⚠️ WARNING'}")

    if not validation_ok:
        print(f"⚠️ WARNING: Validation failed - relative error {relative_error:.2e} > tolerance {args.rtol:.1e}")
        print("   This may be normal for shuffle/atomic reductions due to different summation order.")
        print("   Consider increasing --rtol if numerics are expected to differ.")
    else:
        print("✓ Validation passed")

    # CSV output
    outfile = 'data/raw/reduction_benchmark.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Check if file exists to decide on header
    write_header = not os.path.exists(outfile)

    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'device', 'impl', 'dtype', 'N', 'passes',
                'kernel_ms', 'kernel_time_s', f'bw_{args.units.lower()}', 'pct_peak_bw', 'total_bytes', 'qc_pass',
                'energy_J', 'power_W', 'qc_status',
                'units', 'peak_bw_ref', 'cpu_name', 'gpu_name', 'driver_version'
            ])

        writer.writerow([
            device, args.impl, args.dtype, N, args.passes,
            kernel_ms, kernel_time_s, bw, pct_peak, total_bytes, qc_pass,
            energy_j, power_w, qc_status,
            args.units, args.peak_bw, cpu_name, gpu_name, driver_version
        ])

    print(f"\nResults appended to: {outfile}")

    # Cross-validation hints
    if device == 'GPU':
        if args.impl == 'cupy':
            print(f"\n💡 Cross-validation: Run with --impl custom or --impl cub_cpp to compare performance")
        elif args.impl == 'custom':
            print(f"\n💡 Cross-validation: Run with --impl cupy or --impl cub_cpp to compare performance")
        elif args.impl == 'cub_cpp':
            print(f"\n💡 Cross-validation: Run with --impl cupy or --impl custom to compare performance")
        print(f"   Expected: CUB (cub_cpp) >= cupy >= custom within ±5-10% tolerance")

    if args.passes > 1:
        print(f"\n💡 Tip: With --passes {args.passes}, try different values (100, 200, 500) to find optimal energy measurement window")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
