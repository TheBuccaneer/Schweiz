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

def get_gpu_energy(handle):
    """Get GPU energy via NVML - Total energy consumption in mJ since driver reload"""
    try:
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return energy_mj / 1000  # mJ to J
    except Exception as e:
        # NVML energy not supported on older cards/drivers
        if not hasattr(get_gpu_energy, '_warned'):
            print(f"Warning: GPU energy measurement not available (NVML): {e}")
            get_gpu_energy._warned = True
        return None

def calculate_bandwidth_metrics(N, kernel_time_s, args):
    """Calculate bandwidth metrics with consistent units"""
    bytes_per_elem = 4  # float32
    passes = args.passes
    total_bytes = passes * N * bytes_per_elem  # total bytes processed

    unit_div = (1e9 if args.units == "GB" else (1024**3))
    bw = (total_bytes / unit_div) / kernel_time_s

    pct_peak = None
    if args.peak_bw:
        pct_peak = (bw / args.peak_bw) * 100.0

    return bw, pct_peak, total_bytes

def run_cupy_reduction(x, passes=1):
    """CuPy reduction - uses optimized reduction internally (may use CUB)"""
    result = 0.0
    for r in range(passes):
        result += cp.sum(x)
    return float(result)

def measure_once_gpu(N, target_runtime_s, args):
    """Single GPU measurement with CuPy cp.sum()"""
    # Allocate arrays - Use NumPy random + transfer to avoid CURAND issues
    x_cpu = np.random.rand(N).astype(np.float32)
    x = cp.array(x_cpu)

    passes = args.passes

    # Warmup (single call for CUDA context if needed)
    _ = cp.sum(x)
    cp.cuda.Stream.null.synchronize()

    # CUDA events for precise kernel timing (sub-μs precision)
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    # Timing window
    start_event.record()

    result = run_cupy_reduction(x, passes)

    end_event.record()
    end_event.synchronize()

    # Calculate timing - CuPy Events capture only compute time
    kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    kernel_time_s = kernel_ms / 1000

    return kernel_ms, kernel_time_s, result, x

def find_target_N(candidate_sizes, target_runtime_s, args):
    """Find N that achieves >= target_runtime_s in single run"""
    print(f"Finding target N for GPU to reach {target_runtime_s}s runtime...")

    for N in candidate_sizes:
        print(f"  Testing N={N:,} ({N/1e6:.1f}M elements)...")

        kernel_ms, kernel_time_s, result, x_buf = measure_once_gpu(N, target_runtime_s, args)

        print(f"    Runtime: {kernel_time_s:.3f}s")

        if kernel_time_s >= target_runtime_s:
            print(f"  ✓ Selected N={N:,} with runtime {kernel_time_s:.3f}s")
            return N, kernel_ms, kernel_time_s, result, x_buf

    # Fallback: use largest N and accept < target_runtime_s
    N = candidate_sizes[-1]
    print(f"  ⚠ Using largest N={N:,} (couldn't reach target runtime)")

    kernel_ms, kernel_time_s, result, x_buf = measure_once_gpu(N, target_runtime_s, args)

    return N, kernel_ms, kernel_time_s, result, x_buf

def main():
    parser = argparse.ArgumentParser(
        description='CuPy-Only GPU Reduction Benchmark (CUDA Events provide sub-μs timing precision)'
    )
    parser.add_argument('--N', type=int,
                        help='Specific array size (if not provided, auto-scale to target runtime)')
    parser.add_argument('--peak-bw', type=float,
                        help='Peak bandwidth in chosen units/s for percent calculation')
    parser.add_argument('--units', choices=['GB', 'GiB'], default='GiB',
                        help='Bandwidth units: GB (1e9) or GiB (1024^3)')
    parser.add_argument('--target-runtime', type=float, default=1.2,
                        help='Target runtime in seconds (default: 1.2)')
    parser.add_argument('--passes', type=int, default=1,
                        help='Number of reductions over the same array (default: 1)')

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
    print(f"Implementation: CuPy cp.sum()")
    print(f"Target runtime: {target_runtime_s}s")
    print(f"Passes: {args.passes}")
    print(f"Units: {args.units}")
    if args.peak_bw:
        print(f"Peak BW reference: {args.peak_bw:.1f} {args.units}/s")

    # Seed for reproducibility
    np.random.seed(123)
    # Skip cp.random.seed() to avoid CURAND issues - use NumPy random instead
    print("Random seeds set for reproducibility (NumPy)")

    # Find target N or use specified N
    if args.N:
        N = args.N
        print(f"\nUsing specified N={N:,}")
        kernel_ms, kernel_time_s, result, x_buf = measure_once_gpu(N, target_runtime_s, args)
    else:
        N, kernel_ms, kernel_time_s, result, x_buf = find_target_N(
            candidate_sizes, target_runtime_s, args
        )

    # Validation with numpy ground truth
    x_cpu = cp.asnumpy(x_buf)
    expected_result = np.sum(x_cpu) * args.passes  # Expected result with passes

    print(f"\n=== Energy Measurement Window ===")
    print("Starting energy measurement...")

    # Energy measurement window (around the final measurement)
    t_energy_start = time.perf_counter()
    gpu_energy_before = get_gpu_energy(gpu_handle)

    # Final measurement for energy calculation
    # Use NumPy random + transfer to avoid CURAND issues
    x_cpu_final = np.random.rand(N).astype(np.float32)
    x_buf_final = cp.array(x_cpu_final)

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    final_result = run_cupy_reduction(x_buf_final, args.passes)
    end_event.record()
    end_event.synchronize()

    # Calculate metrics based on final measurement
    kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    kernel_time_s = kernel_ms / 1000.0

    gpu_energy_after = get_gpu_energy(gpu_handle)
    t_energy_end = time.perf_counter()

    energy_window_s = t_energy_end - t_energy_start

    # Calculate energy deltas (robust against counter wraps)
    energy_gpu = None
    if gpu_energy_before is not None and gpu_energy_after is not None:
        delta = gpu_energy_after - gpu_energy_before
        energy_gpu = delta if delta >= 0 else None

    # Calculate power using energy measurement window
    power_gpu = (energy_gpu / energy_window_s) if (energy_gpu is not None and energy_window_s > 0) else None

    # Calculate bandwidth metrics
    bw, pct_peak, total_bytes = calculate_bandwidth_metrics(N, kernel_time_s, args)

    # Dynamic unit label for total bytes display
    unit_div = (1e9 if args.units == "GB" else (1024**3))
    unit_label = args.units

    # QC based on kernel time
    qc_threshold = target_runtime_s
    qc_pass = bool(kernel_time_s >= qc_threshold)

    if qc_pass:
        qc_status = "QC_PASS"
    elif kernel_time_s >= 1.0:
        qc_status = "QC_ACCEPTABLE_SHORT"
    else:
        qc_status = "QC_CRITICAL_TOO_SHORT"

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
    if energy_gpu is not None:
        print(f"Energy: {energy_gpu:.3f} J")
    if power_gpu is not None:
        print(f"Power: {power_gpu:.1f} W")
    print(f"QC Status: {qc_status} (kernel time >= {target_runtime_s}s: {qc_pass})")

    # Validation
    relative_error = abs(final_result - expected_result) / abs(expected_result)
    validation_ok = np.isclose(final_result, expected_result, rtol=5e-4)

    print(f"\n=== Validation ===")
    print(f"Expected: {expected_result:.6e}")
    print(f"Actual:   {final_result:.6e}")
    print(f"Rel. error: {relative_error:.2e}")
    print(f"Status: {'✓ PASS' if validation_ok else '⚠ WARNING'}")

    if not validation_ok:
        print(f"⚠ WARNING: Validation failed - relative error {relative_error:.2e} > tolerance 5e-4")

    # CSV output - same structure as original but new filename
    outfile = 'data/raw/reduction_benchmark_gpu_cupy.csv'
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
            'GPU', 'cupy', 'float32', N, args.passes,
            kernel_ms, kernel_time_s, bw, pct_peak, total_bytes, qc_pass,
            energy_gpu, power_gpu, qc_status,
            args.units, args.peak_bw, cpu_name, gpu_name, driver_version
        ])

    print(f"\nResults appended to: {outfile}")

    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
