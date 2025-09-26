#!/usr/bin/env python3

import os
import sys
import socket
import time
import csv
import argparse
import glob
from datetime import datetime
import subprocess

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

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

def list_rapl_package_zones():
    """
    Find all RAPL package zones (Intel & AMD) with wrap-around support.
    Returns list of (energy_file, max_range_file) tuples.
    """
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return []

    # Search both intel-rapl and amd-rapl trees, including subzones
    roots = glob.glob(os.path.join(base, "*rapl*"))
    all_zones = roots[:]
    for root in roots:
        all_zones.extend(glob.glob(root + ":*"))
        all_zones.extend(glob.glob(root + ":*:*"))

    zones = []
    for zone_path in all_zones:
        if not os.path.isdir(zone_path):
            continue

        # Check if this is a package zone
        name_file = os.path.join(zone_path, "name")
        try:
            with open(name_file, 'r') as f:
                name = f.read().strip().lower()
            if "package" not in name:
                continue
        except Exception:
            continue

        # Check for required files
        energy_file = os.path.join(zone_path, "energy_uj")
        max_range_file = os.path.join(zone_path, "max_energy_range_uj")

        try:
            # Test readability
            with open(energy_file, 'r') as f:
                f.read(1)
            with open(max_range_file, 'r') as f:
                f.read(1)
            zones.append((energy_file, max_range_file))
        except Exception:
            continue

    return zones

def rapl_sample(zones):
    """
    Sample all RAPL zones, returning list of (energy_uj, max_range_uj) tuples.
    Returns (None, None) for zones that can't be read.
    """
    samples = []
    for energy_file, max_range_file in zones:
        try:
            with open(energy_file, 'r') as f:
                energy_uj = int(f.read().strip())
            with open(max_range_file, 'r') as f:
                max_range_uj = int(f.read().strip())
            samples.append((energy_uj, max_range_uj))
        except Exception:
            samples.append((None, None))
    return samples

def rapl_delta_j(before_samples, after_samples):
    """
    Calculate total energy delta in Joules with wrap-around handling.
    before_samples and after_samples are lists of (energy_uj, max_range_uj) tuples.
    """
    if len(before_samples) != len(after_samples):
        return None

    total_uj = 0
    valid_zones = 0

    for (before_uj, before_range), (after_uj, after_range) in zip(before_samples, after_samples):
        if (before_uj is None or after_uj is None or
            before_range is None or after_range is None):
            continue

        # Calculate delta with wrap-around handling
        delta_uj = after_uj - before_uj
        if delta_uj < 0:  # Wrap-around detected
            delta_uj = (after_uj + before_range) - before_uj

        total_uj += delta_uj
        valid_zones += 1

    if valid_zones == 0:
        return None

    return total_uj / 1_000_000.0  # Convert µJ to J

# Parallele Reduktion: Numba JIT, prange = echte Parallelisierung/Reduktion
# (Numba erkennt die Summe als Reduktion bei parallel=True)
@njit(parallel=True, fastmath=True)
def numba_sum(x):
    s = 0.0
    for i in prange(x.size):
        s += x[i]
    return s

def calculate_bandwidth_metrics(N, kernel_time_s, args):
    """Calculate bandwidth metrics with consistent units"""
    bytes_per_elem = 4  # float32
    passes = args.passes if args else 1
    total_bytes = passes * N * bytes_per_elem

    unit_div = (1e9 if args.units == "GB" else (1024**3))
    bw = (total_bytes / unit_div) / kernel_time_s

    pct_peak = None
    if args.peak_bw:
        pct_peak = (bw / args.peak_bw) * 100.0

    return bw, pct_peak, total_bytes

def run_cpu_reduction(x, passes=1):
    """Run CPU reduction (Numba-parallel, immer)"""
    result = 0.0
    for _ in range(passes):
        result += numba_sum(x)
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

def find_target_N(candidate_sizes, target_runtime_s, args=None):
    """Find N that achieves >= target_runtime_s in single run"""
    print(f"Finding target N for CPU to reach {target_runtime_s}s runtime...")

    for N in candidate_sizes:
        print(f"  Testing N={N:,} ({N/1e6:.1f}M elements)...")

        passes = args.passes if args else 1
        kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)

        print(f"    Runtime: {kernel_time_s:.3f}s")

        if kernel_time_s >= target_runtime_s:
            print(f"  ✓ Selected N={N:,} with runtime {kernel_time_s:.3f}s")
            return N, kernel_ms, kernel_time_s, result, x_buf

    # Fallback: use largest N and accept < target_runtime_s
    N = candidate_sizes[-1]
    print(f"  ⚠  Using largest N={N:,} (couldn't reach target runtime)")

    passes = args.passes if args else 1
    kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)

    return N, kernel_ms, kernel_time_s, result, x_buf

def main():
    parser = argparse.ArgumentParser(
        description='CPU Memory-bound Reduction Benchmark with RAPL Energy Measurement'
    )
    parser.add_argument('--N', type=int,
                        help='Specific array size (if not provided, auto-scale to target runtime)')
    parser.add_argument('--dtype', default='float32', choices=['float32'],
                        help='Data type')
    parser.add_argument('--peak-bw', type=float,
                        help='Peak bandwidth in chosen units/s for percent calculation')
    parser.add_argument('--units', choices=['GB', 'GiB'], default='GiB',
                        help='Bandwidth units: GB (1e9) or GiB (1024^3)')
    parser.add_argument('--target-runtime', type=float, default=1.0,
                        help='Target runtime in seconds (default: 1.0)')
    parser.add_argument('--passes', type=int, default=1,
                        help='Repeats over the same array (default: 1)')
    parser.add_argument('--rtol', type=float, default=5e-4,
                        help='Relative tolerance for result validation against NumPy (default: 5e-4)')
    parser.add_argument('--macro-repeats', type=int, default=5,
                        help='Number of macro measurement repetitions (default: 5)')
    parser.add_argument('--threads', type=int,
                        help='Number of threads for Numba backend (optional)')

    args = parser.parse_args()

    # Candidate sizes for single-run scaling (no batching)
    candidate_sizes = [2**24, 2**25, 2**26, 2**27, 2**28, 2**29, 2**30]  # 16M to 1G elements
    target_runtime_s = args.target_runtime

    # Initialize hardware
    cpu_name = get_cpu_info()
    host = socket.gethostname()

    print(f"System: {host}")
    print(f"CPU: {cpu_name}")
    print(f"Implementation: CPU reduction (Numba parallel)")
    print(f"Data type: {args.dtype}")
    print(f"Target runtime: {target_runtime_s}s")
    print(f"Passes: {args.passes}")
    print(f"Macro repeats: {args.macro_repeats}")
    print(f"Units: {args.units}")
    if args.peak_bw:
        print(f"Peak BW reference: {args.peak_bw:.1f} {args.units}/s")

    # Numba-Threadzahl (falls gesetzt). Hinweis: darf NUMBA_NUM_THREADS nicht überschreiten.
    if args.threads:
        set_num_threads(args.threads)
    print(f"Backend threads: {get_num_threads()}")

    # Initialize RAPL
    rapl_zones = list_rapl_package_zones()
    print(f"RAPL zones found: {len(rapl_zones)}")
    for i, (efile, rfile) in enumerate(rapl_zones):
        zone_name = os.path.dirname(efile).split('/')[-1]
        print(f"  Zone {i}: {zone_name}")

    if not rapl_zones:
        print("⚠️  No RAPL package zones found. Energy measurement will be unavailable.")
        print("    Check RAPL permissions: sudo chmod 644 /sys/class/powercap/*/energy_uj")

    # Seed for reproducibility
    np.random.seed(123)
    print("Random seeds set for reproducibility")

    # Find target N or use specified N
    if args.N:
        N = args.N
        print(f"\nUsing specified N={N:,}")

        passes = args.passes if args else 1
        kernel_ms, kernel_time_s, result, x_buf = measure_once_cpu(N, target_runtime_s, passes)
    else:
        N, kernel_ms, kernel_time_s, result, x_buf = find_target_N(
            candidate_sizes, target_runtime_s, args
        )

    # Validation with numpy ground truth (calculate once)
    expected_result = np.sum(x_buf) * args.passes

    # CSV output
    outfile = 'data/raw/cpu_reduction_benchmark.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Check if file exists to decide on header
    write_header = not os.path.exists(outfile)

    with open(outfile, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'timestamp', 'host', 'device', 'impl', 'dtype', 'N', 'passes',
                'kernel_ms', 'kernel_time_s', f'bw_{args.units.lower()}', 'pct_peak_bw', 'total_bytes',
                'energy_j_cpu', 'power_w_cpu', 'qc_status', 'qc_pass',
                'units', 'peak_bw_ref', 'cpu_name', 'macro_repeat'
            ])

        # Run macro repeats
        energy_measurements = []
        power_measurements = []
        runtime_measurements = []

        for macro_rep in range(args.macro_repeats):
            print(f"\n=== Macro Repeat {macro_rep + 1}/{args.macro_repeats} ===")

            # Energy measurement window
            if rapl_zones:
                rapl_before = rapl_sample(rapl_zones)
            else:
                rapl_before = None

            t_start = time.perf_counter()
            final_result = run_cpu_reduction(x_buf, args.passes)
            t_end = time.perf_counter()

            if rapl_zones:
                rapl_after = rapl_sample(rapl_zones)
            else:
                rapl_after = None

            # Calculate metrics
            kernel_time_s = t_end - t_start
            kernel_ms = kernel_time_s * 1000.0

            # Energy calculation with wrap-around handling
            energy_cpu = None
            power_cpu = None
            if rapl_before and rapl_after:
                energy_cpu = rapl_delta_j(rapl_before, rapl_after)
                if energy_cpu is not None and kernel_time_s > 0:
                    power_cpu = energy_cpu / kernel_time_s

            # Store measurements
            if energy_cpu is not None:
                energy_measurements.append(energy_cpu)
            if power_cpu is not None:
                power_measurements.append(power_cpu)
            runtime_measurements.append(kernel_time_s)

            # Calculate bandwidth
            bw, pct_peak, total_bytes = calculate_bandwidth_metrics(N, kernel_time_s, args)

            # QC status matching GEMM script logic
            if kernel_time_s >= 1.2:
                qc_status = "QC_PASS"
                qc_pass = True
            elif kernel_time_s >= 1.0:
                qc_status = "QC_ACCEPTABLE_SHORT"
                qc_pass = True
            else:
                qc_status = "QC_CRITICAL_TOO_SHORT"
                qc_pass = False

            # Progress display
            runtime_check = "✓" if qc_pass else "⚠"
            energy_str = f"E={energy_cpu:.3f}J" if energy_cpu is not None else "E=N/A"
            power_str = f"P={power_cpu:.0f}W" if power_cpu is not None else "P=N/A"

            print(f"  {runtime_check} Runtime: {kernel_time_s:.3f}s | BW: {bw:.1f} {args.units}/s | {energy_str} | {power_str} | [{qc_status}]")

            # Write to CSV
            timestamp = datetime.now().isoformat()
            writer.writerow([
                timestamp, host, 'CPU', 'numba_reduction', args.dtype, N, args.passes,
                kernel_ms, kernel_time_s, bw, pct_peak, total_bytes,
                energy_cpu, power_cpu, qc_status, qc_pass,
                args.units, args.peak_bw, cpu_name, macro_rep + 1
            ])
            f.flush()

        # Final summary
        print(f"\n=== Final Results ===")
        print(f"Array size N: {N:,} ({N/1e6:.1f}M elements)")
        print(f"Passes: {args.passes}")

        # Calculate statistics
        if runtime_measurements:
            median_runtime = np.median(runtime_measurements)
            print(f"Runtime (median): {median_runtime:.3f}s")

        if energy_measurements:
            median_energy = np.median(energy_measurements)
            print(f"Energy (median): {median_energy:.3f}J")
        else:
            print("Energy: N/A (RAPL unavailable)")

        if power_measurements:
            median_power = np.median(power_measurements)
            print(f"Power (median): {median_power:.0f}W")
        else:
            print("Power: N/A (RAPL unavailable)")

        # Validation
        relative_error = abs(final_result - expected_result) / abs(expected_result)
        validation_ok = np.isclose(final_result, expected_result, rtol=args.rtol)

        print(f"\n=== Validation ===")
        print(f"Expected: {expected_result:.6e}")
        print(f"Actual:   {final_result:.6e}")
        print(f"Rel. error: {relative_error:.2e}")
        print(f"Status: {'✓ PASS' if validation_ok else '⚠️ WARNING'}")

        if not validation_ok:
            print(f"⚠️ WARNING: Validation failed - relative error {relative_error:.2e} > tolerance {args.rtol:.1e}")

    print(f"\nResults saved to: {outfile}")

    if not rapl_zones:
        print(f"\n⚠️  RAPL energy measurement failed. Try:")
        print(f"   sudo chmod 644 /sys/class/powercap/*/energy_uj")
        print(f"   sudo chmod 644 /sys/class/powercap/*/max_energy_range_uj")

if __name__ == "__main__":
    main()
