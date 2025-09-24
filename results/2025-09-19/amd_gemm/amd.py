#!/usr/bin/env python3

import os
import socket
import time
import csv
import tempfile
import shutil
import subprocess
import glob
import re
import math
from datetime import datetime

# CRITICAL: Set BLAS threads BEFORE importing numpy
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "32")

# Configuration - AMD CPU only
TARGET_RUNTIME_S = 1.0
MAX_BATCH_SIZE = 60000
MACRO_REPEATS = 5

# GEMM sizes only - same as Intel script
GEMM_SIZES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536]

# Import libraries
import numpy as np

# uProf-Header variieren: "socket0-package-power", "Socket0 Package Power (W)", "Socket 0 Package Power"
SOCKET_COL_RX = re.compile(r"^socket\s*\d+\s*[- ]?\s*package\s*power", re.I)

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

def resolve_uprof_cli():
    """Find AMDuProfCLI - use full path as specified"""
    # Try the specified full path first
    full_path = "/opt/AMDuProf_5.1-701/bin/AMDuProfCLI"
    if os.path.isfile(full_path):
        return full_path

    # Fallback to PATH
    p = shutil.which("AMDuProfCLI")
    if p:
        return p

    # Try other typical installation locations
    for cand in sorted(glob.glob("/opt/AMDuProf_*/bin/AMDuProfCLI")):
        if os.path.isfile(cand):
            return cand
    return None

def parse_uprof_timechart(csv_path, *, interval_ms):
    """
    Parse uProf timechart.csv and return (energy_j, avg_power_w).
    Header tolerant gegenüber Schreibweisen/Einheiten.
    Energy = Sum_i( sum_socketPowers_i * Δt ), Δt = interval_ms/1000.
    """
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            rows = list(rdr)

        # Header finden (erste Zeile, die eine Socket-Power-Spalte enthält)
        header_idx = None
        for i, row in enumerate(rows[:100]):
            cols = [c.strip() for c in row]
            if any(SOCKET_COL_RX.search(c) for c in cols):
                header_idx = i
                break
        if header_idx is None:
            return None, None

        header = [c.strip() for c in rows[header_idx]]
        sock_cols = [j for j, c in enumerate(header) if SOCKET_COL_RX.search(c)]
        if not sock_cols:
            return None, None

        dt = interval_ms / 1000.0
        power_samples = []
        for row in rows[header_idx + 1:]:
            if not row or len(row) <= max(sock_cols):
                continue
            s = 0.0
            ok = False
            for j in sock_cols:
                try:
                    cell = (row[j] or "").strip().replace(",", ".")
                    v = float(cell)
                    if math.isfinite(v):
                        s += v
                        ok = True
                except Exception:
                    pass
            if ok:
                power_samples.append(s)

        if not power_samples:
            return None, None

        energy_j = sum(p * dt for p in power_samples)
        avg_power_w = sum(power_samples) / len(power_samples)
        return energy_j, avg_power_w
    except Exception:
        return None, None

def start_uprof_sampler(duration_s, interval_ms=50):
    """
    Start uProf timechart sampler asynchronously for given duration.
    Returns (proc, outdir) or (None, None) if not available.
    """
    ucli = resolve_uprof_cli()
    if not ucli:
        return None, None
    outdir = tempfile.mkdtemp(prefix="uprof_")
    duration_s = max(2.0, float(duration_s))
    # keep sampler alive with a trivial command so we can overlap with our workload
    cmd = [ucli, "timechart", "--event", "power",
           "--interval", str(interval_ms), "--duration", str(duration_s),
           "-o", outdir, "--format", "csv", "--", "/bin/sleep", str(duration_s)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc, outdir
    except Exception:
        if os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
        return None, None

def find_timechart_csv(outdir):
    """Locate uProf timechart.csv within output directory."""
    cand = os.path.join(outdir, "timechart.csv")
    if os.path.isfile(cand):
        return cand
    alts = glob.glob(os.path.join(outdir, "**", "timechart.csv"), recursive=True)
    return alts[0] if alts else None

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

    # 2) Messweg A: powercap, wenn vorhanden (Δ energy_uj um die Messausführung)
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

    # 3) Messweg B: uProf als Fallback – uProf muss parallel zum Workload laufen
    if energy_cpu is None:
        # Starte Sampler etwas länger als die erwartete Laufzeit
        margin = 0.3
        proc, outdir = start_uprof_sampler(runtime_est + margin, interval_ms=50)
        if proc and outdir:
            # Führe den Messlauf (gleiche batch_size) aus, solange uProf sammelt
            runtime = run_cpu_gemm_fixed(A, B, batch_size)
            # warte bis uProf-Fenster endet
            try:
                proc.wait(timeout=max(2.0, runtime_est + margin + 2.0))
            except Exception:
                proc.kill()
            # CSV suchen & parsen
            csv_path = find_timechart_csv(outdir)
            if csv_path:
                energy_cpu, power_cpu = parse_uprof_timechart(csv_path, interval_ms=50)
            shutil.rmtree(outdir, ignore_errors=True)
        else:
            # Fallback fehlgeschlagen
            energy_cpu, power_cpu = None, None

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
    uprof_test = resolve_uprof_cli()
    print(f"Powercap available: {powercap_test is not None}")
    print(f"uProf available: {uprof_test is not None}")

    # CSV output - same structure as Intel script
    outfile = 'data/raw/energy_benchmark_amd_cpu.csv'
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
