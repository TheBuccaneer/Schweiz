#!/usr/bin/env python3

import os
import socket
import time
import csv
import subprocess
import glob
from datetime import datetime

# CRITICAL: Set BLAS threads BEFORE importing numpy
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")

# Configuration - Intel CPU only
TARGET_RUNTIME_S = 1.0
MAX_BATCH_SIZE = 60000
MACRO_REPEATS = 5

# GEMM sizes only - same as Intel script
GEMM_SIZES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]

# Import libraries
import numpy as np

def get_cpu_info():
    """Get CPU info - AMD Threadripper"""
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':', 1)[1].strip()
    except:
        return "Unknown"

def read_powercap_energy():
    """
    Summe der CPU-Package-Energie (J) über alle RAPL-Zonen unter /sys/class/powercap.
    Liefert None, wenn nichts lesbar ist.
    """
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None

    total_j, found = 0.0, False

    # Zonen sammeln: intel-rapl:* / amd-rapl:* inkl. Subzonen (:0:0 etc.)
    candidates = glob.glob(os.path.join(base, "*rapl*"))
    candidates += glob.glob(os.path.join(base, "*rapl*:*"))
    candidates += glob.glob(os.path.join(base, "*rapl*:*:*"))

    for z in candidates:
        if not os.path.isdir(z):
            continue
        name_fp = os.path.join(z, "name")
        try:
            name = open(name_fp).read().strip().lower()
            if "package" not in name:   # nur Package-Zonen
                continue
        except Exception:
            continue
        efile = os.path.join(z, "energy_uj")
        try:
            uj = int(open(efile).read().strip())
            total_j += uj / 1_000_000.0  # µJ → J
            found = True
        except Exception:
            pass

    return total_j if found else None

def run_cpu_gemm_fixed(A, B, batch_size):
    """Run CPU GEMM for a fixed batch size; return wall time."""
    start_time = time.perf_counter()
    for _ in range(batch_size):
        _ = A.dot(B)
    return time.perf_counter() - start_time

def run_cpu_gemm(A, B, target_runtime):
    """Run CPU GEMM with batch size adaptation"""
    batch_size = 1

    while batch_size <= MAX_BATCH_SIZE:
        start_time = time.perf_counter()

        # Execute batch
        for _ in range(batch_size):
            C = A.dot(B)

        elapsed_time = time.perf_counter() - start_time

        if elapsed_time >= target_runtime:
            return batch_size, elapsed_time

        if batch_size >= MAX_BATCH_SIZE:
            return batch_size, elapsed_time  # Failed to reach target

        batch_size = min(batch_size * 2, MAX_BATCH_SIZE)

    return batch_size, elapsed_time

def measure_once(size):
    """Measure CPU GEMM once and return data dict"""
    # Prepare matrices
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # 1) Ermittele passende batch_size & ungefähre runtime (Dry-Run, ohne Energie)
    batch_size, runtime_est = run_cpu_gemm(A, B, TARGET_RUNTIME_S)

    # 2) RAPL-Messung (Δ energy_uj um die Messausführung)
    cpu_energy_before = read_powercap_energy()
    t_energy_start = time.perf_counter()
    runtime = run_cpu_gemm_fixed(A, B, batch_size)
    cpu_energy_after = read_powercap_energy()
    t_energy_end = time.perf_counter()
    energy_window_s = t_energy_end - t_energy_start

    energy_cpu = None
    power_cpu = None
    if (cpu_energy_before is not None) and (cpu_energy_after is not None):
        delta = cpu_energy_after - cpu_energy_before
        if delta >= 0:
            energy_cpu = delta
            power_cpu = delta / energy_window_s if energy_window_s > 0 else None

    # QC tiers
    qc_status = ("QC_PASS" if runtime >= 1.2
                else "QC_ACCEPTABLE_SHORT" if runtime >= 1.0
                else "QC_CRITICAL_TOO_SHORT") if runtime is not None else "QC_CRITICAL_TOO_SHORT"

    # Power reliability flag
    power_reliable = (energy_cpu is not None) or (runtime >= 1.2 if runtime is not None else False)

    return {
        'batch_size': batch_size,
        'runtime': runtime,
        'energy_cpu': energy_cpu,
        'power_cpu': power_cpu,
        'qc_status': qc_status,
        'power_reliable': power_reliable
    }

def main():
    # Initialize
    cpu_name = get_cpu_info()
    host = socket.gethostname()

    print(f"System: {host}")
    print(f"CPU: {cpu_name}")
    print(f"Target Runtime: {TARGET_RUNTIME_S}s per measurement")

    # Test energy measurement methods
    powercap_test = read_powercap_energy()
    print(f"Powercap available: {powercap_test is not None}")

    # CSV output - same structure as Intel script
    outfile = 'data/raw/energy_benchmark_intel_cpu.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'host', 'device', 'device_model',
            'matrix_size', 'batch_size', 'runtime_s', 'gpu_kernel_time_s',
            'energy_j_gpu', 'energy_j_cpu', 'power_w_gpu', 'power_w_cpu',
            'qc_status', 'power_reliable'
        ])

        for size in GEMM_SIZES:
            print(f"\nGEMM size {size}x{size}")

            for rep in range(1, MACRO_REPEATS + 1):
                timestamp = datetime.now().isoformat()

                # Measure CPU
                result = measure_once(size)

                # Output to CSV - fill only CPU columns, GPU columns remain empty/None
                writer.writerow([
                    timestamp, host, 'CPU', cpu_name,
                    size, result['batch_size'], result['runtime'], None,  # gpu_kernel_time_s = None
                    None, result['energy_cpu'], None, result['power_cpu'],  # GPU energy/power = None
                    result['qc_status'], result['power_reliable']
                ])
                f.flush()

                # Progress display - CPU-First-Ausgabe (prioritize CPU values)
                runtime_check = "✓" if (result['runtime'] is not None and result['runtime'] >= TARGET_RUNTIME_S) else "⚠ "
                qc_print = "QC_PASS" if (result['runtime'] is not None and result['runtime'] >= TARGET_RUNTIME_S) else "QC_ISSUE"

                # CPU-Run: CPU zuerst anzeigen
                energy_str = f"E={result['energy_cpu']:.1f}J" if result['energy_cpu'] is not None else "E=N/A"
                power_str = f"P={result['power_cpu']:.0f}W" if result['power_cpu'] is not None else "P=N/A"
                runtime_str = f"{result['runtime']:.3f}s" if result['runtime'] is not None else "N/A"

                print(f"  CPU run {rep}/{MACRO_REPEATS}: {runtime_check}{runtime_str} (batch={result['batch_size']}) {energy_str} {power_str} [{qc_print}]")

    print(f"\nResults saved to: {outfile}")

if __name__ == "__main__":
    main()
