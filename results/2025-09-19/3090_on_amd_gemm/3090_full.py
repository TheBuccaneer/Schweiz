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

# Configuration - RTX 3090 only
TARGET_RUNTIME_S = 1.0
MAX_BATCH_SIZE = 60000
MACRO_REPEATS = 5

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
    """Get CPU info for logging purposes only"""
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':', 1)[1].strip()
    except:
        return "Unknown"

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
    outfile = 'data/raw/energy_benchmark_gpu_only.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'host', 'device', 'device_model',
            'matrix_size', 'batch_size', 'runtime_s', 'gpu_kernel_time_s',
            'energy_j_gpu', 'power_w_gpu',
            'qc_status', 'power_reliable', 'energy_window_s'
        ])

        for size in GEMM_SIZES:
            print(f"\nGEMM size {size}x{size}")

            for rep in range(1, MACRO_REPEATS + 1):
                timestamp = datetime.now().isoformat()

                # Prepare matrices fresh for each measurement (CPU work - outside timing)
                A = np.random.rand(size, size).astype(np.float32)
                B = np.random.rand(size, size).astype(np.float32)

                # E2E Energy measurement starts HERE
                t_energy_start = time.perf_counter()
                gpu_energy_before = get_gpu_energy(gpu_handle)

                # Host→Device transfers are now INSIDE the measurement
                A_gpu = cp.array(A)
                B_gpu = cp.array(B)

                # Run GPU GEMM
                batch_size, gpu_kernel_time, wall_time = run_gpu_gemm(A_gpu, B_gpu, TARGET_RUNTIME_S)
                runtime = wall_time

                # Device→Host transfer (für realistischen Use Case)
                result = cp.asnumpy(A_gpu)  # ← DAS fehlt noch!
                # Ensure all GPU work is complete before measuring energy
                cp.cuda.Device().synchronize()

                # E2E Energy measurement ends HERE
                gpu_energy_after = get_gpu_energy(gpu_handle)
                t_energy_end = time.perf_counter()

                energy_window_s = t_energy_end - t_energy_start

                # Calculate energy deltas
                energy_gpu = None
                if gpu_energy_before is not None and gpu_energy_after is not None:
                    energy_gpu = gpu_energy_after - gpu_energy_before
                    if energy_gpu < 0:
                        energy_gpu = None

                # Calculate power using energy measurement window
                power_gpu = (energy_gpu / energy_window_s) if (energy_gpu is not None and energy_window_s > 0) else None

                # QC tiers
                qc_status = ("QC_PASS" if wall_time >= 1.2
                            else "QC_ACCEPTABLE_SHORT" if wall_time >= 1.0
                            else "QC_CRITICAL_TOO_SHORT")

                # Power reliability flag: NVML energy available OR Δt >= 1.2s
                power_reliable = (energy_gpu is not None) or (energy_window_s >= 1.2)

                # Output
                writer.writerow([
                    timestamp, host, 'GPU', gpu_name,
                    size, batch_size, runtime, gpu_kernel_time,
                    energy_gpu, power_gpu,
                    qc_status, power_reliable, energy_window_s
                ])
                f.flush()

                # Progress display
                runtime_check = "✓" if runtime >= TARGET_RUNTIME_S else "⚠ "
                qc_print = "QC_PASS" if runtime >= TARGET_RUNTIME_S else "QC_ISSUE"

                energy_str = f"E={energy_gpu:.1f}J" if energy_gpu is not None else "E=N/A"
                power_str = f"P={power_gpu:.0f}W" if power_gpu is not None else "P=N/A"

                print(f"  GPU run {rep}/{MACRO_REPEATS}: {runtime_check}{runtime:.3f}s (batch={batch_size}) {energy_str} {power_str} [{qc_print}]")

    pynvml.nvmlShutdown()
    print(f"\nResults saved to: {outfile}")

if __name__ == "__main__":
    main()
