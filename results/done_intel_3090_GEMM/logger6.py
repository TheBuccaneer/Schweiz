#!/usr/bin/env python3

import os
import socket
import platform

# CRITICAL: Set BLAS threads BEFORE importing numpy for reproducible CPU measurements
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import csv
from datetime import datetime
import subprocess
import threading
from collections import deque

# Configuration Constants
TARGET_RUNTIME_S = 1.2          # Hard minimum runtime for stable measurements
TARGET_RUNTIME_RETRY_FACTOR = 1.2  # Multiply factor for single retry
GPU_INDEX = 0                   # GPU device index to use
MACRO_REPEATS = 3              # Number of macro runs per {device, workload, size}
MAX_BATCH_SIZE = 50000         # Maximum batch size limit (increased)
POWER_SAMPLE_HZ = 20           # Power sampling frequency for integration

# QC Power validation thresholds (Watts)
GPU_POWER_MIN = 10
GPU_POWER_MAX = 450
CPU_POWER_MIN = 5
CPU_POWER_MAX = 200

# Utilization change thresholds for stability detection
GPU_UTIL_VARIANCE_THRESHOLD = 15  # % points acceptable variance

# Workload definitions
GEMM_SIZES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792]
REDUCTION_SIZES = [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 4096, 8192, 16384, 32768, 65536]

# Try NVML / nvidia-ml-py (official replacement for deprecated pynvml)
try:
    # Suppress deprecation warnings from transitional packages
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*deprecated.*")
        import pynvml  # This should now be from nvidia-ml-py package
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: nvidia-ml-py not available. GPU measurements will be limited.")
    print("Install with: pip install nvidia-ml-py")
except Exception as e:
    NVML_AVAILABLE = False
    print(f"Warning: NVML initialization issue: {e}")

# NVML/GPU Functions
def init_nvml():
    if NVML_AVAILABLE:
        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        return driver_version.decode() if isinstance(driver_version, bytes) else str(driver_version)
    return None

def shutdown_nvml():
    if NVML_AVAILABLE:
        pynvml.nvmlShutdown()

def get_gpu_handle(device_idx=GPU_INDEX):
    if NVML_AVAILABLE:
        return pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    return None

def get_gpu_metadata(handle):
    """Get comprehensive GPU metadata"""
    if not NVML_AVAILABLE or not handle:
        return None, None, None, None, None

    try:
        # GPU name
        name = pynvml.nvmlDeviceGetName(handle)
        gpu_name = name.decode() if isinstance(name, bytes) else str(name)

        # GPU UUID
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        gpu_uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)

        # Power limit
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000  # mW to W

        # Persistence mode
        try:
            persistence = pynvml.nvmlDeviceGetPersistenceMode(handle)
            persistence_enabled = "yes" if persistence == pynvml.NVML_FEATURE_ENABLED else "no"
        except:
            persistence_enabled = "unknown"

        # CUDA version (from runtime if available)
        cuda_version = "unknown"
        try:
            import cupy as cp
            cuda_version = f"{cp.cuda.runtime.runtimeGetVersion()}"
        except:
            pass

        return gpu_name, gpu_uuid, power_limit, persistence_enabled, cuda_version
    except Exception as e:
        return None, None, None, None, None

def supports_total_energy_consumption(handle):
    """
    Test if GPU actually supports NVML TotalEnergyConsumption by attempting the call
    More robust than name-based heuristics which can fail with OEM variants
    """
    if not NVML_AVAILABLE or not handle:
        return False

    try:
        # Try to get the energy reading - if it works, it's supported
        _ = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return True
    except Exception:
        # Function not available or not supported on this GPU
        return False

def get_gpu_utilization_and_state(handle):
    """Get comprehensive GPU state for stability monitoring"""
    if not NVML_AVAILABLE or not handle:
        return None, None, None, None

    try:
        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = util.gpu

        # Performance state
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)

        # Power limit (enforced)
        power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000  # mW to W

        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        return gpu_util, pstate, power_limit, temp
    except Exception:
        return None, None, None, None

def get_gpu_energy(handle):
    """Get GPU energy using NVML TotalEnergyConsumption (mJ since driver reload)"""
    if NVML_AVAILABLE and handle:
        try:
            energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            return energy_mj / 1000  # mJ → J
        except Exception:
            return None
    return None

def get_gpu_power(handle):
    """Get current GPU power in Watts (averaged over last second per nvidia-smi manual)"""
    if NVML_AVAILABLE and handle:
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            return power_mw / 1000  # mW → W
        except Exception:
            return None
    return None

# RAPL/CPU Functions
def read_rapl_energy_joules():
    """Read CPU energy via RAPL powercap with wrap-around handling and documentation"""
    rapl_path = '/sys/class/powercap/intel-rapl:0/energy_uj'
    rapl_max_path = '/sys/class/powercap/intel-rapl:0/max_energy_range_uj'

    if not os.path.isfile(rapl_path):
        # Check if file exists but is not readable
        if os.path.exists(rapl_path):
            return None, "rapl_permission_denied"
        else:
            return None, "rapl_not_available"

    try:
        with open(rapl_path, 'r') as f:
            uj = int(f.read().strip())

        # Try to read max_energy_range for wrap-around detection and documentation
        max_range = None
        if os.path.isfile(rapl_max_path):
            try:
                with open(rapl_max_path, 'r') as f:
                    max_range = int(f.read().strip())
            except:
                pass

        # Store max_range as function attribute for wrap-around detection
        if not hasattr(read_rapl_energy_joules, 'max_range'):
            read_rapl_energy_joules.max_range = max_range
            # Log the range for reproducibility documentation
            if max_range:
                print(f"RAPL max_energy_range: {max_range} µJ ({max_range/1_000_000:.1f} J)")
            else:
                print("RAPL max_energy_range: not available")

        return uj / 1_000_000, None  # µJ → J, no error

    except PermissionError:
        return None, "rapl_permission_denied"
    except Exception as e:
        return None, f"rapl_read_error_{str(e)}"

def calculate_rapl_energy_delta(e_before, e_after):
    """Calculate RAPL energy delta with wrap-around correction"""
    if e_before is None or e_after is None:
        return None

    # Convert to microjoules for wrap-around calculation
    uj_before = int(e_before * 1_000_000)
    uj_after = int(e_after * 1_000_000)

    delta_uj = uj_after - uj_before

    # Handle wrap-around if we have max_range info
    if hasattr(read_rapl_energy_joules, 'max_range') and read_rapl_energy_joules.max_range:
        if delta_uj < 0:
            # Potential wrap-around occurred
            delta_uj = (read_rapl_energy_joules.max_range + uj_after) - uj_before
            # Add note about wrap-around correction
            return delta_uj / 1_000_000 if delta_uj >= 0 else None, "rapl_wraparound_corrected"

    # Convert back to Joules
    if delta_uj >= 0:
        return delta_uj / 1_000_000, None
    else:
        return None, "rapl_negative_delta"

# Power Integration Functions
def integrate_power_to_energy(power_samples, timestamps):
    """Integrate power samples to energy using trapezoidal rule"""
    if len(power_samples) < 2 or len(timestamps) < 2:
        if power_samples and timestamps:
            avg_power = sum(power_samples) / len(power_samples)
            total_time = timestamps[-1] if timestamps else 0
            return avg_power * total_time
        return None

    energy = 0.0
    for i in range(1, len(power_samples)):
        dt = timestamps[i] - timestamps[i-1]
        avg_power = (power_samples[i] + power_samples[i-1]) / 2
        energy += avg_power * dt

    return energy

def enforce_target_runtime_gpu(workload_name, workload_data, gpu_handle, gpu_supports_total_energy, target_runtime=TARGET_RUNTIME_S):
    """
    Hard enforcement of target runtime through adaptive batch doubling.
    Includes Pascal power integration within measurement window.
    Returns: (batch_size, gpu_kernel_time, wall_time, energy_gpu, notes, qc_flag)
    """
    import cupy as cp

    notes = []
    batch_size = 1
    retry_attempted = False
    current_target = target_runtime
    energy_gpu = None

    while batch_size <= MAX_BATCH_SIZE:
        # Pre-measurement stability check
        util_before, pstate_before, power_limit, temp = get_gpu_utilization_and_state(gpu_handle)

        # Ensure GPU is idle
        cp.cuda.Stream.null.synchronize()

        # Measure energy before (for Ampere+)
        e_gpu_before = get_gpu_energy(gpu_handle) if gpu_supports_total_energy else None

        # Setup power sampling for Pascal (start sampling right before measurement)
        power_samples = []
        power_timestamps = []
        stop_event = threading.Event()
        sampler_thread = None

        if not gpu_supports_total_energy:
            # Pascal power integration - sample during actual measurement window
            t0 = time.perf_counter()

            def power_sampler_loop():
                while not stop_event.is_set():
                    p = get_gpu_power(gpu_handle)
                    if p is not None:
                        current_time = time.perf_counter()
                        power_samples.append(p)
                        power_timestamps.append(current_time - t0)
                    time.sleep(1.0 / POWER_SAMPLE_HZ)

            sampler_thread = threading.Thread(target=power_sampler_loop, daemon=True)
            sampler_thread.start()

        # Setup CUDA events
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Measurement window start
        wall_start = time.perf_counter()
        start_event.record()

        # Execute workload batch
        if workload_name == 'GEMM':
            A_gpu, B_gpu = workload_data
            for _ in range(batch_size):
                C_gpu = cp.dot(A_gpu, B_gpu)
        else:  # Reduction
            data_gpu = workload_data
            for _ in range(batch_size):
                result = cp.sum(data_gpu)

        # Measurement window end
        end_event.record()
        end_event.synchronize()
        wall_end = time.perf_counter()

        # Stop power sampling immediately after synchronize
        if not gpu_supports_total_energy and sampler_thread:
            stop_event.set()
            sampler_thread.join(timeout=1.0)

        # Measure energy after (for Ampere+)
        e_gpu_after = get_gpu_energy(gpu_handle) if gpu_supports_total_energy else None

        # Get timings
        gpu_kernel_time = cp.cuda.get_elapsed_time(start_event, end_event) / 1000  # ms to s
        wall_time = wall_end - wall_start

        # Calculate energy
        if gpu_supports_total_energy and e_gpu_before is not None and e_gpu_after is not None:
            energy_delta = e_gpu_after - e_gpu_before
            if energy_delta >= 0:
                energy_gpu = energy_delta
                notes.append('nvml_total_energy_delta')
            else:
                notes.append('negative_energy_delta')
        elif not gpu_supports_total_energy and power_samples:
            # Pascal power integration
            energy_gpu = integrate_power_to_energy(power_samples, power_timestamps)
            notes.append(f'power_integration_{len(power_samples)}_samples')

        # Post-measurement stability check
        util_after, pstate_after, power_limit_after, temp_after = get_gpu_utilization_and_state(gpu_handle)

        # Log GPU state before/after for load detection
        notes.append(f"pstate_before_{pstate_before}_after_{pstate_after}")
        notes.append(f"power_limit_before_{power_limit:.0f}W_after_{power_limit_after:.0f}W")
        notes.append(f"temp_before_{temp}C_after_{temp_after}C")

        # Check for stability
        if util_before is not None and util_after is not None:
            util_variance = abs(util_after - util_before)
            if util_variance > GPU_UTIL_VARIANCE_THRESHOLD:
                notes.append(f"gpu_util_variance_{util_variance:.1f}%")

        # Status line output for live monitoring - ONLY after measurement complete
        p_gpu = energy_gpu / wall_time if energy_gpu and wall_time > 0 else None
        method_gpu = 'NVML_ΔE' if gpu_supports_total_energy else 'PowerInt'

        # Safe formatting for None values - but don't print during timing measurements
        # This print happens AFTER all timing is complete

        # QC and retry logic
        runtime_ok = gpu_kernel_time >= current_target
        power_ok = p_gpu is None or (GPU_POWER_MIN <= p_gpu <= GPU_POWER_MAX)

        if runtime_ok and power_ok:
            # Success - target achieved with good QC
            qc_flag = "runtime_and_qc_passed"
            notes.append(f"target_runtime_achieved_batch_{batch_size}")
            return batch_size, gpu_kernel_time, wall_time, energy_gpu, '; '.join(notes), qc_flag

        # Target not reached or QC failed - try larger batch or retry
        if batch_size >= MAX_BATCH_SIZE:
            if not retry_attempted and (not runtime_ok or not power_ok):
                # One retry with relaxed target or different conditions
                current_target *= TARGET_RUNTIME_RETRY_FACTOR
                batch_size = 1
                retry_attempted = True
                retry_reason = "runtime" if not runtime_ok else "power_qc"
                notes.append(f"retry_due_to_{retry_reason}_target_{current_target:.1f}s")
                continue
            else:
                # Failed even with retry
                qc_flag = "runtime_or_qc_failed_max_batch"
                failure_reason = []
                if not runtime_ok:
                    failure_reason.append(f"runtime_{gpu_kernel_time:.3f}s_below_{current_target:.1f}s")
                if not power_ok:
                    failure_reason.append(f"power_{p_gpu:.1f}W_outside_range")
                notes.extend(failure_reason)
                return batch_size, gpu_kernel_time, wall_time, energy_gpu, '; '.join(notes), qc_flag

        # Double batch size for next iteration
        batch_size = min(batch_size * 2, MAX_BATCH_SIZE)

    # Should not reach here
    qc_flag = "runtime_enforcement_error"
    return batch_size, gpu_kernel_time, wall_time, energy_gpu, '; '.join(notes), qc_flag

# System Info Functions
def get_system_info():
    """Get system metadata"""
    return {
        'host': socket.gethostname(),
        'os': f"{platform.system()} {platform.release()}",
        'cpu_model': get_cpu_model()
    }

def get_cpu_model():
    """Get CPU model name"""
    try:
        out = subprocess.check_output(['lscpu'], universal_newlines=True)
        for line in out.splitlines():
            if 'Model name:' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return None

# Workload Execution Functions
def execute_gemm_batch(A, B, batch_size, use_gpu=False):
    """Execute GEMM workload batch"""
    if use_gpu:
        import cupy as cp
        for _ in range(batch_size):
            C = cp.dot(A, B)
        return C
    else:
        for _ in range(batch_size):
            C = A.dot(B)
        return C

def execute_reduction_batch(data, batch_size, use_gpu=False):
    """Execute Reduction workload batch"""
    if use_gpu:
        import cupy as cp
        for _ in range(batch_size):
            result = cp.sum(data)
        return result
    else:
        import numpy as np
        for _ in range(batch_size):
            result = np.sum(data)
        return result

# QC Validation Functions
def validate_power(power_w, device_type):
    """Validate power measurements and return QC flag"""
    if power_w is None:
        return "power_unavailable"

    if device_type == 'GPU':
        if power_w < GPU_POWER_MIN or power_w > GPU_POWER_MAX:
            return f"gpu_power_unrealistic_{power_w:.1f}W"
    elif device_type == 'CPU':
        if power_w < CPU_POWER_MIN or power_w > CPU_POWER_MAX:
            return f"cpu_power_unrealistic_{power_w:.1f}W"

    return "power_ok"

def main():
    # Initialize system
    system_info = get_system_info()
    driver_version = init_nvml()
    gpu_handle = get_gpu_handle(GPU_INDEX) if NVML_AVAILABLE else None
    gpu_name, gpu_uuid, power_limit_w, persistence_enabled, cuda_version = get_gpu_metadata(gpu_handle)

    # Determine GPU energy measurement method by testing actual capability
    gpu_supports_total_energy = supports_total_energy_consumption(gpu_handle)

    print(f"System: {system_info['host']} ({system_info['os']})")
    print(f"CPU: {system_info['cpu_model']}")
    print(f"GPU: {gpu_name} (UUID: {gpu_uuid})")
    print(f"Driver: {driver_version}, CUDA: {cuda_version}")
    print(f"GPU Energy Method: {'NVML TotalEnergyConsumption' if gpu_supports_total_energy else 'Power Integration (Pascal/older)'}")
    print(f"Target Runtime: {TARGET_RUNTIME_S}s per measurement")

    # Output file
    outfile = 'data/raw/energy_benchmark_production.csv'
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    # Timer function
    now = time.perf_counter

    # Check CuPy availability
    try:
        import cupy as cp
        CUPY_AVAILABLE = True
        print("CuPy available - using CUDA events for GPU timing")
    except ImportError:
        CUPY_AVAILABLE = False
        print("CuPy not available - GPU measurements will use CPU fallback")

    # Create data arrays outside the measurement loop
    import numpy as np

    # Workload definitions
    workloads = [
        ('GEMM', GEMM_SIZES),
        ('Reduction', REDUCTION_SIZES)
    ]

    with open(outfile, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'host', 'os',
            'device', 'device_model', 'device_uuid',
            'driver_version', 'cuda_version', 'power_limit_w', 'persistence_enabled',
            'workload', 'size', 'batch_size', 'repeats',
            'time_s', 'gpu_kernel_time_s',
            'energy_j_gpu', 'energy_j_cpu', 'p_w_gpu', 'p_w_cpu', 'edp_gpu_j_s', 'edp_cpu_j_s',
            'timing_method', 'energy_method_gpu', 'energy_method_cpu',
            'notes', 'qc_power_unreal_flag'
        ])
        f.flush()

        for workload_name, sizes in workloads:
            for size in sizes:
                print(f"\nStarting {workload_name} size {size}")

                # Prepare workload data
                if workload_name == 'GEMM':
                    A = np.random.rand(size, size).astype(np.float32)
                    B = np.random.rand(size, size).astype(np.float32)
                    workload_data = (A, B)
                    if CUPY_AVAILABLE:
                        A_gpu = cp.array(A)
                        B_gpu = cp.array(B)
                        workload_data_gpu = (A_gpu, B_gpu)
                elif workload_name == 'Reduction':
                    data = np.random.rand(size).astype(np.float32)
                    workload_data = data
                    if CUPY_AVAILABLE:
                        data_gpu = cp.array(data)
                        workload_data_gpu = data_gpu

                # Brief warm-up with timeout to trigger CuPy compilation
                warmup_notes = ''
                if CUPY_AVAILABLE and workload_name == 'GEMM':
                    try:
                        import signal
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Warmup timeout")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60s timeout

                        _ = cp.dot(A_gpu, B_gpu)
                        cp.cuda.Stream.null.synchronize()

                        signal.alarm(0)  # Cancel timeout
                        warmup_notes = 'warmup_gemm_completed; '
                    except (TimeoutError, Exception) as e:
                        warmup_notes = f'warmup_failed_{str(e)[:20]}; '
                elif CUPY_AVAILABLE and workload_name == 'Reduction':
                    try:
                        import signal
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Warmup timeout")

                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(60)  # 60s timeout

                        _ = cp.sum(data_gpu)
                        cp.cuda.Stream.null.synchronize()

                        signal.alarm(0)  # Cancel timeout
                        warmup_notes = 'warmup_reduction_completed; '
                    except (TimeoutError, Exception) as e:
                        warmup_notes = f'warmup_failed_{str(e)[:20]}; '

                for device in ['CPU', 'GPU']:
                    for rep in range(1, MACRO_REPEATS + 1):
                        timestamp = datetime.now().isoformat()
                        notes = warmup_notes  # Include warmup notes
                        timing_method = ''
                        energy_method_gpu = ''
                        energy_method_cpu = ''
                        gpu_kernel_time = None
                        qc_flag = ''

                        # Measure pre-energy
                        e_cpu_before, cpu_error_before = read_rapl_energy_joules()

                        # Execute workload with hard runtime enforcement
                        if device == 'GPU' and CUPY_AVAILABLE:
                            # Hard runtime enforcement for GPU with integrated energy measurement
                            batch_size, gpu_kernel_time, time_elapsed, energy_gpu, runtime_notes, runtime_qc = enforce_target_runtime_gpu(
                                workload_name, workload_data_gpu, gpu_handle, gpu_supports_total_energy, TARGET_RUNTIME_S
                            )

                            notes += runtime_notes + '; '
                            timing_method = 'CUDA_events_batched_hard_runtime'

                            # Energy method determination
                            if gpu_supports_total_energy:
                                energy_method_gpu = 'nvml_total_energy_delta'
                            else:
                                energy_method_gpu = 'power_integration'

                        elif device == 'GPU':
                            # GPU fallback to CPU
                            notes += 'GPU_requested_but_cupy_unavailable_using_cpu_fallback; '
                            device = 'CPU'  # Change device for this measurement

                        if device == 'CPU':
                            # CPU execution with hard runtime enforcement
                            timing_method = 'perf_counter_batched_hard_runtime'

                            # Hard enforcement for CPU too
                            batch_size = 1
                            target_achieved = False

                            while batch_size <= MAX_BATCH_SIZE and not target_achieved:
                                t_start = time.perf_counter()

                                if workload_name == 'GEMM':
                                    A, B = workload_data
                                    for _ in range(batch_size):
                                        C = A.dot(B)
                                else:  # Reduction
                                    data = workload_data
                                    for _ in range(batch_size):
                                        result = np.sum(data)

                                t_end = time.perf_counter()
                                time_elapsed = t_end - t_start

                                if time_elapsed >= TARGET_RUNTIME_S:
                                    target_achieved = True
                                    notes += f'cpu_target_runtime_achieved_batch_{batch_size}; '
                                else:
                                    # Check if we've already hit max batch size
                                    if batch_size >= MAX_BATCH_SIZE:
                                        break
                                    batch_size = min(batch_size * 2, MAX_BATCH_SIZE)

                            if not target_achieved:
                                notes += f'cpu_target_runtime_failed_max_batch_{MAX_BATCH_SIZE}; '
                                runtime_qc = "cpu_runtime_target_failed"
                            else:
                                runtime_qc = "cpu_runtime_target_achieved"

                            energy_gpu = None  # No GPU energy for CPU runs

                        # Measure post-energy
                        e_cpu_after, cpu_error_after = read_rapl_energy_joules()

                        # Calculate energy consumption with robust error handling
                        if device != 'GPU':
                            energy_gpu = None  # Only set for GPU measurements
                        # energy_gpu is already calculated in enforce_target_runtime_gpu for GPU cases

                        energy_cpu = None

                        # CPU energy calculation with robust RAPL handling (unchanged)
                        if cpu_error_before is None and cpu_error_after is None and e_cpu_before is not None and e_cpu_after is not None:
                            # Both readings successful
                            cpu_delta_result = calculate_rapl_energy_delta(e_cpu_before, e_cpu_after)
                            if isinstance(cpu_delta_result, tuple):
                                energy_cpu, rapl_note = cpu_delta_result
                                energy_method_cpu = 'rapl_powercap_energy_uj'
                                if rapl_note:
                                    notes += f'{rapl_note}; '
                                    if 'wraparound' in rapl_note:
                                        energy_method_cpu = 'rapl_powercap_energy_uj_wraparound_corrected'
                            else:
                                energy_cpu = cpu_delta_result
                                energy_method_cpu = 'rapl_powercap_energy_uj'

                            if energy_cpu is not None and energy_cpu < 0:
                                energy_cpu = None
                                energy_method_cpu = 'rapl_negative_delta_uncorrectable'
                                notes += 'rapl_negative_delta_after_correction; '

                            # Document RAPL range in notes for first measurement
                            if hasattr(read_rapl_energy_joules, 'max_range') and read_rapl_energy_joules.max_range:
                                if not hasattr(main, 'rapl_range_logged'):
                                    notes += f'rapl_max_range_{read_rapl_energy_joules.max_range}_uj; '
                                    main.rapl_range_logged = True
                        else:
                            # RAPL reading failed
                            error_reason = cpu_error_before or cpu_error_after or 'unknown_rapl_error'
                            energy_method_cpu = f'rapl_unavailable_{error_reason}'
                            notes += f'{error_reason}; '

                        # Calculate derived metrics and perform QC validation
                        p_gpu = energy_gpu / time_elapsed if energy_gpu is not None and time_elapsed > 0 else None
                        p_cpu = energy_cpu / time_elapsed if energy_cpu is not None and time_elapsed > 0 else None

                        edp_gpu = energy_gpu * time_elapsed if energy_gpu is not None else None
                        edp_cpu = energy_cpu * time_elapsed if energy_cpu is not None else None

                        # Comprehensive QC power validation
                        qc_flags = []

                        # Add runtime enforcement result - ONLY if failed
                        if 'runtime_qc' in locals():
                            if 'failed' in runtime_qc.lower() or 'below_target' in runtime_qc.lower():
                                qc_flags.append(runtime_qc)

                        # Validate GPU power if available
                        if device == 'GPU' and p_gpu is not None:
                            gpu_qc = validate_power(p_gpu, 'GPU')
                            if gpu_qc != 'power_ok':
                                qc_flags.append(gpu_qc)

                            # Additional GPU-specific QC checks
                            if gpu_kernel_time is not None and gpu_kernel_time < TARGET_RUNTIME_S:
                                qc_flags.append(f"gpu_runtime_below_target_{gpu_kernel_time:.3f}s")

                        # Validate CPU power if available
                        if p_cpu is not None:
                            cpu_qc = validate_power(p_cpu, 'CPU')
                            if cpu_qc != 'power_ok':
                                qc_flags.append(cpu_qc)

                        # Check for target runtime achievement (critical QC criterion)
                        if device == 'GPU' and gpu_kernel_time is not None:
                            if gpu_kernel_time < TARGET_RUNTIME_S:
                                qc_flags.append("CRITICAL_gpu_runtime_below_target")
                        elif device == 'CPU' and time_elapsed < TARGET_RUNTIME_S:
                            qc_flags.append("CRITICAL_cpu_runtime_below_target")

                        # Combine all QC flags
                        qc_flag_combined = '; '.join(qc_flags) if qc_flags else 'qc_passed'

                        writer.writerow([
                            timestamp, system_info['host'], system_info['os'],
                            device, gpu_name if device == 'GPU' else system_info['cpu_model'],
                            gpu_uuid if device == 'GPU' else '',
                            driver_version, cuda_version,
                            power_limit_w if device == 'GPU' else '', persistence_enabled if device == 'GPU' else '',
                            workload_name, size, batch_size, 1,  # repeats is always 1 due to adaptive batching
                            time_elapsed, gpu_kernel_time if device == 'GPU' else None,
                            energy_gpu, energy_cpu, p_gpu, p_cpu, edp_gpu, edp_cpu,
                            timing_method, energy_method_gpu, energy_method_cpu,
                            notes.rstrip('; '), qc_flag_combined
                        ])
                        f.flush()

                        # Enhanced progress reporting with QC status
                        runtime_status = f"✓{gpu_kernel_time:.3f}s" if (device == 'GPU' and gpu_kernel_time and gpu_kernel_time >= TARGET_RUNTIME_S) else f"⚠ {gpu_kernel_time:.3f}s" if (device == 'GPU' and gpu_kernel_time) else f"✓{time_elapsed:.3f}s" if time_elapsed >= TARGET_RUNTIME_S else f"⚠ {time_elapsed:.3f}s"

                        energy_status = f"E={energy_gpu:.2f}J P={p_gpu:.1f}W" if (device == 'GPU' and energy_gpu and p_gpu) else f"E={energy_cpu:.2f}J P={p_cpu:.1f}W" if (device == 'CPU' and energy_cpu and p_cpu) else "E=N/A"

                        qc_status = "QC_PASS" if qc_flag_combined == 'qc_passed' else "QC_ISSUE"

                        print(f"  {device} run {rep}/{MACRO_REPEATS}: {runtime_status} (batch={batch_size}) {energy_status} [{qc_status}]")

    shutdown_nvml()
    print(f"\nCompleted energy benchmark. Results saved to: {outfile}")
    print(f"Total measurements: {len(workloads) * sum(len(sizes) for _, sizes in workloads) * 2 * MACRO_REPEATS}")

if __name__ == "__main__":
    main()
