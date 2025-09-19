#!/usr/bin/env python3

import os
import socket
import platform
import time
import csv
from datetime import datetime
import subprocess
import threading

# CRITICAL: Set BLAS threads BEFORE importing numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# TF32 control for reproducible FP32 comparisons (Ampere vs older GPUs)
os.environ.setdefault("CUPY_TF32", "0")          # CuPy: TF32 off by default
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")  # NVIDIA libs: prefer FP32

# Configuration - simplified for i9-7900X + RTX 3090
TARGET_RUNTIME_S = 1.2
MAX_BATCH_SIZE = 50000
MACRO_REPEATS = 3
POWER_SAMPLE_HZ = 20

# GEMM sizes only
GEMM_SIZES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]

# Import libraries - assume they're available for this specific setup
import numpy as np
import cupy as cp
import pynvml

def init_gpu():
    """Initialize GPU - assume RTX 3090 setup"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # GPU info - handle both string and bytes return types
    name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = name.decode() if isinstance(name, bytes) else str(name)

    uuid = pynvml.nvmlDeviceGetUUID(handle)
    gpu_uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)

    driver_version = pynvml.nvmlSystemGetDriverVersion()
    driver_ver = driver_version.decode() if isinstance(driver_version, bytes) else str(driver_version)

    return handle, gpu_name, gpu_uuid, driver_ver

def get_cpu_info():
    """Get CPU info - assume Intel with RAPL"""
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':', 1)[1].strip()
    except:
        return "Unknown"

def read_rapl_energy():
    """Read Intel RAPL energy - assume permissions are set"""
    try:
        with open('/sys/class/powercap/intel-rapl:0/energy_uj', 'r') as f:
            uj = int(f.read().strip())
        return uj / 1_000_000  # Convert to Joules
    except:
        return None

def get_gpu_energy(handle):
    """Get GPU energy via NVML - RTX 3090 supports this"""
    try:
        energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return energy_mj / 1000  # mJ to J
    except:
        return None

def get_gpu_state(handle):
    """Get basic GPU state"""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        return util.gpu, pstate, temp, power_limit
    except:
        return None, None, None, None

def run_gpu_gemm(A_gpu, B_gpu, target_runtime):
    """Run GPU GEMM with batch size adaptation to reach target runtime"""
    batch_size = 1

    while batch_size <= MAX_BATCH_SIZE:
        # CUDA events for precise GPU timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Wall time for energy calculation
        wall_start = time.perf_counter()
        start_event.record()

        # Execute batch
        for _ in range(batch_size):
            C_gpu = cp.dot(A_gpu, B_gpu)

        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter()

        gpu_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000  # ms to s
        wall_time = wall_end - wall_start

        if gpu_time >= target_runtime:
            return batch_size, gpu_time, wall_time

        if batch_size >= MAX_BATCH_SIZE:
            return batch_size, gpu_time, wall_time  # Failed to reach target

        batch_size = min(batch_size * 2, MAX_BATCH_SIZE)

    return batch_size, gpu_time, wall_time

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

def main():
    # Initialize
    gpu_handle, gpu_name, gpu_uuid, driver_version = init_gpu()
    cpu_name = get_cpu_info()
    host = socket.gethostname()

    print(f"System: {host}")
    print(f"CPU: {cpu_name}")
    print(f"GPU: {gpu_name}")
    print(f"Driver: {driver_version}")
    print(f"Target Runtime: {TARGET_RUNTIME_S}s per measurement")

    # CSV output
    outfile = 'data/raw/energy_benchmark_simple.csv'
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

            # Prepare matrices
            A = np.random.rand(size, size).astype(np.float32)
            B = np.random.rand(size, size).astype(np.float32)
            A_gpu = cp.array(A)
            B_gpu = cp.array(B)

            for device in ['CPU', 'GPU']:
                for rep in range(1, MACRO_REPEATS + 1):
                    timestamp = datetime.now().isoformat()

                    # Measure energy before
                    t_energy_start = time.perf_counter()
                    cpu_energy_before = read_rapl_energy()
                    gpu_energy_before = get_gpu_energy(gpu_handle)

                    if device == 'CPU':
                        batch_size, runtime = run_cpu_gemm(A, B, TARGET_RUNTIME_S)
                        gpu_kernel_time = None
                        wall_time = runtime
                    else:  # GPU
                        batch_size, gpu_kernel_time, wall_time = run_gpu_gemm(A_gpu, B_gpu, TARGET_RUNTIME_S)
                        runtime = wall_time

                    # Measure energy after
                    cpu_energy_after = read_rapl_energy()
                    gpu_energy_after = get_gpu_energy(gpu_handle)
                    t_energy_end = time.perf_counter()

                    energy_window_s = t_energy_end - t_energy_start

                    # Calculate energy deltas - robust calculation without truthiness bug
                    energy_cpu = None
                    energy_gpu = None

                    if cpu_energy_before is not None and cpu_energy_after is not None:
                        energy_cpu = cpu_energy_after - cpu_energy_before
                        if energy_cpu < 0:
                            energy_cpu = None

                    if gpu_energy_before is not None and gpu_energy_after is not None:
                        energy_gpu = gpu_energy_after - gpu_energy_before
                        if energy_gpu < 0:
                            energy_gpu = None

                    # Calculate power using energy measurement window - robust calculation
                    power_cpu = (energy_cpu / energy_window_s) if (energy_cpu is not None and energy_window_s > 0) else None
                    power_gpu = (energy_gpu / energy_window_s) if (energy_gpu is not None and energy_window_s > 0) else None

                    # QC tiers
                    qc_status = ("QC_PASS" if wall_time >= 1.2
                                else "QC_ACCEPTABLE_SHORT" if wall_time >= 1.0
                                else "QC_CRITICAL_TOO_SHORT")

                    # Power-Verlässlichkeitsflag: NVML-Energie vorhanden ODER Δt >= 1.2s
                    power_reliable = (energy_gpu is not None) or (energy_window_s >= 1.2)

                    # Output
                    writer.writerow([
                        timestamp, host, device,
                        gpu_name if device == 'GPU' else cpu_name,
                        size, batch_size, runtime, gpu_kernel_time,
                        energy_gpu, energy_cpu, power_gpu, power_cpu,
                        qc_status, power_reliable
                    ])
                    f.flush()

                    # Progress display with corrected labels
                    runtime_check = "✓" if runtime >= TARGET_RUNTIME_S else "⚠"
                    qc_print = "QC_PASS" if runtime >= TARGET_RUNTIME_S else "QC_ISSUE"

                    # Correct energy/power labels based on available measurements
                    energy_str = (
                        f"E={energy_gpu:.1f}J" if energy_gpu is not None
                        else (f"E={energy_cpu:.1f}J" if energy_cpu is not None else "E=N/A")
                    )
                    power_str = (
                        f"P={power_gpu:.0f}W" if power_gpu is not None
                        else (f"P={power_cpu:.0f}W" if power_cpu is not None else "P=N/A")
                    )

                    print(f"  {device} run {rep}/{MACRO_REPEATS}: {runtime_check}{runtime:.3f}s (batch={batch_size}) {energy_str} {power_str} [{qc_print}]")

    pynvml.nvmlShutdown()
    print(f"\nResults saved to: {outfile}")

if __name__ == "__main__":
    main()
