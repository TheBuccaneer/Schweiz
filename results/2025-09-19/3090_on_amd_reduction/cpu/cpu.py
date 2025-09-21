#!/usr/bin/env python3

import os
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

import numpy as np

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

def calculate_bandwidth_metrics(N, kernel_time_s, args):
    """Calculate bandwidth metrics with consistent units"""
    bytes_per_elem = 4  # float32
    passes = args.passes
    total_bytes = passes * N * bytes_per_elem

    unit_div = (1e9 if args.units == "GB" else (1024**3))
    bw = (total_bytes / unit_div) / kernel_time_s

    pct_peak = None
    if args.peak_bw:
        pct_peak = (bw / args.peak_bw) * 100.0

    return bw, pct_peak, total_bytes

def run_cpu_reduction(x, passes=1):
    """Run CPU reduction with NumPy"""
    result = 0.0
    for r in range(passes):
        result += np.sum(x)
    return result

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

def find_target_N(candidate_sizes, target_runtime_s, args):
    """Find N that achieves >= target_runtime_s in single run"""
    print(f"Finding target N for CPU to reach {target_runtime_s}s runtime...")

    for N in candidate_sizes:
        print(f"  Testing N={N:,} ({N/1e6:.1f}M elements)...")

        kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, args.passes)

        print(f"    Runtime: {kernel_time_s:.3f}s")

        if kernel_time_s >= target_runtime_s:
            print(f"  ✓ Selected N={N:,} with runtime {kernel_time_s:.3f}s")
            return N, kernel_ms, kernel_time_s, result, x_buf

    # Fallback: use largest N and accept < target_runtime_s
    N = candidate_sizes[-1]
    print(f"  ⚠ Using largest N={N:,} (couldn't reach target runtime)")

    kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, args.passes)

    return N, kernel_ms, kernel_time_s, result, x_buf

def main():
    parser = argparse.ArgumentParser(
        description='CPU-Only Reduction Benchmark (NumPy with multiple passes)'
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

    # Initialize system info
    cpu_name = get_cpu_info()
    host = socket.gethostname()

    print(f"System: {host}")
    print(f"CPU: {cpu_name}")
    print(f"Implementation: NumPy CPU reduction")
    print(f"Target runtime: {target_runtime_s}s")
    print(f"Passes: {args.passes}")
    print(f"Units: {args.units}")
    if args.peak_bw:
        print(f"Peak BW reference: {args.peak_bw:.1f} {args.units}/s")

    # Seed for reproducibility
    np.random.seed(123)
    print("Random seeds set for reproducibility")

    # Find target N or use specified N
    if args.N:
        N = args.N
        print(f"\nUsing specified N={N:,}")
        kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, args.passes)
    else:
        N, kernel_ms, kernel_time_s, result, x_buf = find_target_N(
            candidate_sizes, target_runtime_s, args
        )

    # Validation with numpy ground truth (calculate once)
    x_cpu = x_buf
    expected_result = np.sum(x_cpu) * args.passes  # Expected result with passes

    print(f"\n=== Final Measurement ===")
    print("Running final measurement...")

    # Final measurement
    start_time = time.perf_counter()
    final_result = run_cpu_reduction(x_buf, args.passes)
    end_time = time.perf_counter()

    # Calculate metrics based on final measurement
    kernel_time_s = end_time - start_time
    kernel_ms = kernel_time_s * 1000.0

    bw, pct_peak, total_bytes = calculate_bandwidth_metrics(N, kernel_time_s, args)

    qc_threshold = target_runtime_s
    qc_pass = bool(kernel_time_s >= qc_threshold)

    print(f"Results: kernel_ms={kernel_ms:.3f}, result={final_result}, bw={bw:.2f} {args.units}/s")

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

    # Results display
    print(f"\n=== Results ===")
    print(f"Array size N: {N:,} ({N/1e6:.1f}M elements)")
    print(f"Passes: {args.passes}")
    print(f"Total bytes: {total_bytes:,} ({total_bytes/unit_div:.2f} {unit_label})")
    print(f"Kernel time: {kernel_ms:.3f} ms ({kernel_time_s:.3f} s)")
    print(f"Result: {final_result:.6e}")
    print(f"Bandwidth: {bw:.1f} {args.units}/s")
    if pct_peak is not None:
        print(f"Peak BW %: {pct_peak:.1f}%")
    print(f"QC Status: {qc_status} (kernel time >= {target_runtime_s}s: {qc_pass})")

    # Validation (configurable tolerance, warning instead of crash)
    relative_error = abs(final_result - expected_result) / abs(expected_result)
    validation_ok = np.isclose(final_result, expected_result, rtol=5e-4)

    print(f"\n=== Validation ===")
    print(f"Expected: {expected_result:.6e}")
    print(f"Actual:   {final_result:.6e}")
    print(f"Rel. error: {relative_error:.2e}")
    print(f"Tolerance: 5e-4")
    print(f"Status: {'✓ PASS' if validation_ok else '⚠ WARNING'}")

    if not validation_ok:
        print(f"⚠ WARNING: Validation failed - relative error {relative_error:.2e} > tolerance 5e-4")
    else:
        print("✓ Validation passed")

    # CSV output
    outfile = 'data/raw/reduction_benchmark_cpu.csv'
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
                'units', 'peak_bw_ref', 'cpu_name'
            ])

        writer.writerow([
            'CPU', 'numpy', 'float32', N, args.passes,
            kernel_ms, kernel_time_s, bw, pct_peak, total_bytes, qc_pass,
            None, None, qc_status,  # No energy/power measurement
            args.units, args.peak_bw, cpu_name
        ])

    print(f"\nResults appended to: {outfile}")

if __name__ == "__main__":
    main()
