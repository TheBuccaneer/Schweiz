#!/usr/bin/env python3

import os
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
        return pynvml.nvmlDeviceGetName(handle)
    else:
        return None

def get_gpu_energy(handle):
    """
    Versucht GPU-Energie zu messen.
    Fallback: None (Power×Time wird dann im Hauptprogramm berechnet)
    """
    if NVML_AVAILABLE and handle:
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            return energy_mj / 1000  # mJ → J
        except Exception:
            # TotalEnergyConsumption nicht verfügbar
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
            return power_mw / 1000  # mW → W
        except Exception:
            return None
    else:
        return None

# Funktion: CPU Energie via RAPL (Linux)
def read_rapl_energy_joules():
    """
    Liest die CPU Energie via RAPL (Package domain, energy_uj).
    Gibt Joule zurück oder None, falls nicht verfügbar.
    """
    rapl_path = '/sys/class/powercap/intel-rapl:0/energy_uj'
    if os.path.isfile(rapl_path):
        try:
            with open(rapl_path, 'r') as f:
                uj = int(f.read().strip())
            # Umwandlung: Mikro-Joule → Joule
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

def main():
    driver_version = init_nvml()
    gpu_handle = get_gpu_handle(0) if NVML_AVAILABLE else None
    gpu_name = get_gpu_name(gpu_handle)

    cpu_model = get_cpu_model()

    # Größen & Wiederholungen definieren
    workloads = ['GEMM']
    sizes = [2000, 4000, 8000]  # Beispiel: klein, mittel, groß
    repeats = 10

    # Output-Datei
    outfile = 'data/raw/pilot_full.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    with open(outfile, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp',
            'workload',
            'size',
            'device',
            'repeat',
            'energy_j',
            'time_s',
            'cpu_model',
            'gpu_model',
            'driver_version',
            'rapl_available',
            'notes'
        ])
        f.flush()

        for workload in workloads:
            for size in sizes:
                # Erzeuge Daten
                import numpy as np

                A = np.random.rand(size, size).astype(np.float32)
                B = np.random.rand(size, size).astype(np.float32)

                for device in ['CPU', 'GPU']:
                    for rep in range(1, repeats + 1):
                        timestamp = datetime.now().isoformat()
                        notes = ''  # Initialisiere notes

                        # GPU Power-Messung für Fallback (vor Workload)
                        gpu_power_samples = []

                        # Vor-Energie
                        if device == 'GPU':
                            e_before = get_gpu_energy(gpu_handle)
                            # Falls Energy Counter nicht verfügbar, nutze Power-Sampling
                            if e_before is None and gpu_handle:
                                # Sammle Power-Samples während der Ausführung
                                gpu_power_samples = []
                        else:
                            e_before = read_rapl_energy_joules()

                        t_start = time.time()

                        # Workload ausführen
                        if device == 'GPU' and gpu_handle:
                            # Optional: GPU Ausführung; z. B. mit CuPy falls vorhanden
                            try:
                                import cupy as cp
                                A_gpu = cp.array(A)
                                B_gpu = cp.array(B)

                                # Wenn Energy Counter nicht verfügbar, messe Power während Ausführung
                                if e_before is None:
                                    # Start Power Sampling
                                    C_gpu = cp.dot(A_gpu, B_gpu)
                                    cp.cuda.Stream.null.synchronize()
                                    # Nimm Power-Sample nach der Operation
                                    power_sample = get_gpu_power(gpu_handle)
                                    if power_sample is not None:
                                        gpu_power_samples.append(power_sample)
                                else:
                                    C_gpu = cp.dot(A_gpu, B_gpu)
                                    cp.cuda.Stream.null.synchronize()

                            except ImportError:
                                notes = 'CuPy not installed, using CPU fallback'
                                # Fallback zu CPU:
                                C = A.dot(B)
                        else:
                            # CPU
                            C = A.dot(B)

                        t_end = time.time()
                        time_elapsed = t_end - t_start

                        # Nach-Energie
                        if device == 'GPU':
                            e_after = get_gpu_energy(gpu_handle)

                            # Berechne Energie
                            if (e_before is not None) and (e_after is not None):
                                # Energy Counter verfügbar
                                energy_used = e_after - e_before
                                if energy_used < 0:
                                    energy_used = None
                                    notes = (notes + '; ' if notes else '') + 'negative_energy_delta'
                            elif gpu_power_samples:
                                # Fallback: Power × Time
                                avg_power = sum(gpu_power_samples) / len(gpu_power_samples)
                                energy_used = avg_power * time_elapsed
                                notes = (notes + '; ' if notes else '') + 'power_time_estimate'
                            else:
                                energy_used = None
                                notes = (notes + '; ' if notes else '') + 'energy_unavailable'
                        else:
                            e_after = read_rapl_energy_joules()

                            # CPU Energie
                            if (e_before is not None) and (e_after is not None):
                                energy_used = e_after - e_before
                                if energy_used < 0:
                                    energy_used = None
                                    notes = (notes + '; ' if notes else '') + 'negative_energy_delta'
                            else:
                                energy_used = None
                                notes = (notes + '; ' if notes else '') + 'energy_unavailable'

                        # RAPL Verfügbarkeit prüfen
                        rapl_available = 'yes' if read_rapl_energy_joules() is not None else 'no'

                        writer.writerow([
                            timestamp,
                            workload,
                            size,
                            device,
                            rep,
                            energy_used,
                            time_elapsed,
                            cpu_model,
                            gpu_name,
                            driver_version,
                            rapl_available,
                            notes
                        ])
                        f.flush()

    shutdown_nvml()
    print(f"Done. Pilot data in {outfile}")

if __name__ == "__main__":
    main()
