#!/usr/bin/env python3

import os

# CRITICAL: Set BLAS threads BEFORE importing numpy for reproducible CPU measurements
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import csv
from datetime import datetime
import subprocess

# Versuche NVML / pynvml
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Funktion: Energie via NVML (GPU)
def init_nvml():
    if NVML_AVAILABLE:
        pynvml.nvmlInit()
        # ggf. Version holen
        driver_version = pynvml.nvmlSystemGetDriverVersion()
    else:
        driver_version = None
    return driver_version

def shutdown_nvml():
    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()

def get_gpu_handle(device_idx=0):
    if NVML_AVAILABLE:
        return pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    else:
        return None

def get_gpu_name(handle):
    if NVML_AVAILABLE and handle:
        name = pynvml.nvmlDeviceGetName(handle)
        # Handle bytes/string conversion for different pynvml versions
        return name.decode() if isinstance(name, bytes) else str(name)
    else:
        return None

def get_gpu_energy(handle):
    """
    Versucht GPU-Energie zu messen.
    Fallback: None (PowerÃ—Time wird dann im Hauptprogramm berechnet)
    """
    if NVML_AVAILABLE and handle:
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            return energy_mj / 1000  # mJ â†' J
        except Exception:
            # TotalEnergyConsumption nicht verfÃ¼gbar
            return None
    else:
        return None

def get_gpu_power(handle):
    """
    Liest aktuelle GPU-Leistung in Watt.
    """
    if NVML_AVAILABLE and handle:
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return power_mw / 1000  # mW â†' W
        except Exception:
            return None
    else:
        return None

# Funktion: CPU Energie via RAPL (Linux)
def read_rapl_energy_joules():
    """
    Liest die CPU Energie via RAPL (Package domain, energy_uj).
    Gibt Joule zurÃ¼ck oder None, falls nicht verfÃ¼gbar.
    """
    rapl_path = '/sys/class/powercap/intel-rapl:0/energy_uj'
    if os.path.isfile(rapl_path):
        try:
            with open(rapl_path, 'r') as f:
                uj = int(f.read().strip())
            # Umwandlung: Mikro-Joule â†' Joule
            return uj / 1_000_000
        except Exception:
            return None
    else:
        return None

def get_cpu_model():
    # z. B. via lscpu
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':',1)[1].strip()
    except Exception:
        pass
    return None

def calculate_batch_size(size):
    """
    Calculate optimal batch size to get 5-50ms execution windows
    for stable energy measurements
    """
    if size <= 2000:
        return 50  # Small matrices: many repetitions
    elif size <= 4000:
        return 20  # Medium matrices: moderate repetitions
    else:
        return 5   # Large matrices: few repetitions

def main():
    driver_version = init_nvml()
    gpu_handle = get_gpu_handle(0) if NVML_AVAILABLE else None
    gpu_name = get_gpu_name(gpu_handle)

    cpu_model = get_cpu_model()

    # GrÃ¶ÃŸen & Wiederholungen definieren
    workloads = ['GEMM']
    sizes = [2000, 4000, 8000]  # Beispiel: klein, mittel, groÃŸ
    repeats = 10

    # Output-Datei
    outfile = 'data/raw/pilot_full_robust.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Improved timer function
    now = time.perf_counter

    # Check if CuPy is available for proper CUDA event timing
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
        print("CuPy available - using CUDA events for GPU timing")
    except ImportError:
        CUPY_AVAILABLE = False
        print("CuPy not available - using CPU timing with synchronization")

    with open(outfile, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'workload',
            'size',
            'device',
            'repeat',
            'batch_size',           # Track batching for energy stability
            'energy_j',
            'energy_j_per_op',     # Energy per single operation
            'time_s',
            'time_s_per_op',       # Time per single operation
            'gpu_kernel_time_s',   # Total GPU kernel time
            'gpu_kernel_time_s_per_op',  # GPU kernel time per operation
            'cpu_model',
            'gpu_model',
            'driver_version',
            'rapl_available',
            'timing_method',       # Track which timing method was used
            'energy_method',       # Track which energy measurement method was used
            'notes'
        ])
        f.flush()

        for workload in workloads:
            for size in sizes:
                # Calculate optimal batch size for stable measurements
                batch_size = calculate_batch_size(size)

                # Erzeuge Daten
                import numpy as np

                A = np.random.rand(size, size).astype(np.float32)
                B = np.random.rand(size, size).astype(np.float32)

                # Warm-up runs for GPU (important for accurate benchmarking)
                if CUPY_AVAILABLE:
                    print(f"Warming up GPU for {workload} {size}x{size} (batch_size={batch_size})...")
                    A_gpu = cp.array(A)
                    B_gpu = cp.array(B)
                    for _ in range(3):  # 3 warm-up runs
                        _ = cp.dot(A_gpu, B_gpu)
                        cp.cuda.Stream.null.synchronize()

                for device in ['CPU', 'GPU']:
                    for rep in range(1, repeats + 1):
                        timestamp = datetime.now().isoformat()
                        notes = ''
                        timing_method = ''
                        gpu_kernel_time = None

                        # GPU Power-Messung fÃ¼r Fallback (vor Workload)
                        gpu_power_samples = []

                        # Vor-Energie
                        if device == 'GPU':
                            e_before = get_gpu_energy(gpu_handle)
                        else:
                            e_before = read_rapl_energy_joules()

                        # GPU execution with proper timing and batching
                        if device == 'GPU' and CUPY_AVAILABLE:
                            # Ensure GPU is idle before measurement
                            cp.cuda.Stream.null.synchronize()

                            # Setup CUDA events for accurate GPU timing
                            start_event = cp.cuda.Event()
                            end_event = cp.cuda.Event()

                            # Always measure power for fallback (even if energy counter available)
                            power_before = get_gpu_power(gpu_handle)

                            # CPU timing start (for comparison)
                            t_start = now()

                            # Record start event
                            start_event.record()

                            # Execute kernel batch for stable energy measurement window
                            for batch_i in range(batch_size):
                                C_gpu = cp.dot(A_gpu, B_gpu)

                            # Record end event
                            end_event.record()

                            # Synchronize to ensure completion
                            end_event.synchronize()

                            # CPU timing end
                            t_end = now()
                            time_elapsed = t_end - t_start

                            # Get accurate GPU kernel time
                            gpu_kernel_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000  # ms to seconds
                            timing_method = 'CUDA_events_batched'

                            # Always measure power after for fallback
                            power_after = get_gpu_power(gpu_handle)

                            # Prepare power fallback data
                            if power_before is not None and power_after is not None:
                                avg_power = (power_before + power_after) / 2
                                gpu_power_samples = [avg_power]

                        elif device == 'GPU' and gpu_handle and not CUPY_AVAILABLE:
                            notes = 'CuPy not installed, using CPU fallback with CPU timing'
                            timing_method = 'CPU_fallback_batched'

                            # Fallback zu CPU mit improved timing and batching:
                            t_start = now()
                            for batch_i in range(batch_size):
                                C = A.dot(B)
                            t_end = now()
                            time_elapsed = t_end - t_start

                        elif device == 'GPU':
                            notes = 'GPU not available, using CPU fallback'
                            timing_method = 'CPU_fallback_batched'

                            # Fallback zu CPU:
                            t_start = now()
                            for batch_i in range(batch_size):
                                C = A.dot(B)
                            t_end = now()
                            time_elapsed = t_end - t_start

                        else:
                            # CPU execution with improved timing and batching
                            timing_method = 'CPU_perf_counter_batched'

                            t_start = now()
                            for batch_i in range(batch_size):
                                C = A.dot(B)
                            t_end = now()
                            time_elapsed = t_end - t_start

                        # Nach-Energie
                        if device == 'GPU':
                            e_after = get_gpu_energy(gpu_handle)

                            # Berechne Energie mit intelligenter Fallback-Strategie
                            energy_used = None
                            energy_method = ''

                            if (e_before is not None) and (e_after is not None):
                                # Energy Counter verfügbar - prüfe Plausibilität
                                energy_delta = e_after - e_before

                                if energy_delta < 0:
                                    # Negative Energie - Counter-Wraparound
                                    notes = (notes + '; ' if notes else '') + 'negative_energy_delta'
                                    energy_used = None
                                    energy_method = 'nvml_total_energy_negative'
                                elif energy_delta == 0.0 and gpu_power_samples:
                                    # 0.0 J aber Power verfügbar - Counter zu träge für kurze Jobs
                                    avg_power = sum(gpu_power_samples) / len(gpu_power_samples)
                                    timing_for_energy = gpu_kernel_time if gpu_kernel_time is not None else time_elapsed
                                    energy_used = avg_power * timing_for_energy
                                    energy_method = 'power_time_fallback_zero_counter'
                                    notes = (notes + '; ' if notes else '') + 'energy_counter_zero_fallback_to_power'
                                elif energy_delta > 0:
                                    # Plausible Energie vom Counter
                                    energy_used = energy_delta
                                    energy_method = 'nvml_total_energy'
                                else:
                                    # 0.0 J und keine Power-Daten
                                    energy_method = 'nvml_total_energy_zero_no_power'
                                    notes = (notes + '; ' if notes else '') + 'energy_counter_zero_no_power'

                            elif gpu_power_samples:
                                # Kein Energy Counter - direkt Power×Time
                                avg_power = sum(gpu_power_samples) / len(gpu_power_samples)
                                timing_for_energy = gpu_kernel_time if gpu_kernel_time is not None else time_elapsed
                                energy_used = avg_power * timing_for_energy
                                energy_method = 'power_time_primary'
                                notes = (notes + '; ' if notes else '') + 'no_energy_counter_using_power'
                            else:
                                # Weder Energy Counter noch Power verfügbar
                                energy_used = None
                                energy_method = 'energy_unavailable'
                                notes = (notes + '; ' if notes else '') + 'energy_unavailable'
                        else:
                            e_after = read_rapl_energy_joules()

                            # CPU Energie
                            if (e_before is not None) and (e_after is not None):
                                energy_used = e_after - e_before
                                if energy_used < 0:
                                    energy_used = None
                                    energy_method = 'rapl_negative'
                                    notes = (notes + '; ' if notes else '') + 'negative_energy_delta'
                                else:
                                    energy_method = 'rapl'
                            else:
                                energy_used = None
                                energy_method = 'rapl_unavailable'
                                notes = (notes + '; ' if notes else '') + 'energy_unavailable'

                        # Calculate per-operation metrics
                        energy_per_op = energy_used / batch_size if energy_used is not None else None
                        time_per_op = time_elapsed / batch_size
                        gpu_kernel_time_per_op = gpu_kernel_time / batch_size if gpu_kernel_time is not None else None

                        # RAPL VerfÃ¼gbarkeit prÃ¼fen
                        rapl_available = 'yes' if read_rapl_energy_joules() is not None else 'no'

                        writer.writerow([
                            timestamp,
                            workload,
                            size,
                            device,
                            rep,
                            batch_size,
                            energy_used,
                            energy_per_op,
                            time_elapsed,
                            time_per_op,
                            gpu_kernel_time,
                            gpu_kernel_time_per_op,
                            cpu_model,
                            gpu_name,
                            driver_version,
                            rapl_available,
                            timing_method,
                            energy_method,  # Fixed: Always use computed energy_method
                            notes
                        ])
                        f.flush()

                        # Progress feedback
                        if rep % 5 == 0 or rep == 1:
                            print(f"Completed {workload} {size}x{size} {device} run {rep}/{repeats} (batch={batch_size})")

    shutdown_nvml()
    print(f"Done. Robust pilot data in {outfile}")

if __name__ == "__main__":
    main()
