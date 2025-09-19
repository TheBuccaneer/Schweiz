# Reproduce — Methods & Runbook (updated 2025-09-19 09:39)

> Purpose: end‑to‑end, reproducible instructions to collect *valid* timing/energy data for **GEMM** (compute‑bound) and **Reduction** (memory‑bound), compare CPU↔GPU, and export CSVs ready for analysis. Written for our ICIMSD submission workflow.

---

## 0) Quick gist (what matters)
- **Kernel timing = CUDA Events** (device time).  
- **Energy/Power = ΔE/Δt** with an **identical host window** around the two energy reads.  
  - GPU: NVML `nvmlDeviceGetTotalEnergyConsumption` (returns **mJ since driver reload**) → Δ in **J**.  
  - CPU: RAPL `energy_uj` (µJ) → Δ in **J**.
- **QC tiers (based on the energy-window Δt):**  
  - `QC_PASS` if Δt ≥ **1.2 s**; `QC_ACCEPTABLE_SHORT` if 1.0–1.2 s; else `QC_CRITICAL_TOO_SHORT`.
- **Headless for final runs:** disable desktop/compositor (GUI adds board power to ΔE).  
- **GEMM**: sizes grid (64…1536); **Reduction**: single large arrays (`N ∈ {2^24,2^26}`, `float32`), report **GB/s** & **%peak_BW** (not TFLOPS).

---

## 1) Environment
```bash
python3 -m venv .venv && source .venv/bin/activate
python --version   # ≥3.9
pip install -U nvidia-ml-py cupy-cuda13x numpy pandas
# optional (plots): matplotlib
```
**BLAS threads fixed (before NumPy import):**
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

**TF32 (Ampere, e.g., RTX 3090)**
- Default **off** for FP32 comparability: set before CuPy import  
  `export CUPY_TF32=0` and `export NVIDIA_TF32_OVERRIDE=0`

**PCIe**
- System currently pinned to **PCIe Gen3** (info only; we exclude transfers from kernel timing).

---

## 2) Devices & energy/timing
- **GPU timing:** CUDA events around *only the kernel* (record → run → sync → elapsed).  
- **GPU energy:** read NVML energy **just before** and **just after** the kernel; wrap those two reads with a host `perf_counter()` window to define **Δt**. Compute **Power = ΔE/Δt**.  
- **CPU energy:** read RAPL `energy_uj` similarly; same host window defines **Δt**.

---

## 3) QC & Reliability
- `QC_PASS` if energy window **Δt ≥ 1.2 s**; `QC_ACCEPTABLE_SHORT` if 1.0–1.2 s; else `CRITICAL_TOO_SHORT`.
- `power_reliable = (energy_gpu is not None) or (Δt ≥ 1.2)`
- Always store: `kernel_time_s`, `energy_window_s`, `energy_J_(cpu/gpu)`, `power_W_(cpu/gpu)`.

---

## 4) Headless testing (used) — what we actually did
We **performed headless tests exactly as follows** (tested and confirmed):
```bash
# switch to text mode (headless), from a separate TTY (Ctrl+Alt+F3) or SSH:
sudo systemctl isolate multi-user.target

# optional but recommended: stabilize driver state
sudo nvidia-smi -pm 1               # persistence mode on
sudo nvidia-smi --lock-gpu-clocks=tdp,tdp   # lock clocks for stability

# ... run the benchmarks headless ...

# reset clocks and return to GUI
sudo nvidia-smi --reset-gpu-clocks
sudo systemctl isolate graphical.target
```
Rationale: NVML reports **total board energy**; disabling the desktop/compositor removes GUI baseline power from ΔE.

---

## 5) GEMM (compute‑bound)
**Sizes:** 64..1536.  
**Timing:** CUDA events → `gpu_kernel_time_s`; CPU uses wall time.  
**Energy/Power:** ΔE/Δt using the identical energy window; **do not** mix kernel time into Δt.

**CSV fields (minimum):**
```
timestamp, host, device, device_model,
matrix_size, batch_size,
runtime_s (energy_window), gpu_kernel_time_s,
energy_j_gpu, energy_j_cpu, power_w_gpu, power_w_cpu,
qc_status, power_reliable
```
**Expected (RTX 3090):** ~**200–350 W** headless under heavy GEMM; `power.draw` (1‑s average) close in magnitude.

**Status:** Implemented; GEMM is methodologically “green” (headless runs yield plausible board power; Events and energy-window agree).

---

## 6) Reduction (memory‑bound) — minimal benchmark
**Goal:** Bandwidth metrics (GB/s, %peak_BW) + energy/Power via ΔE/Δt.

**Problem sizes:** `N ∈ (26, 24)`, `dtype=float32`  
**Implementations:**
1) **Library baseline:** `cp.sum(x)` (or `torch.sum(x)`); one large array; copy H→D **once**; time only the kernel.
2) **CUB reference (optional):** `cub::DeviceReduce::Sum` (two‑call API for temp storage).
3) **Custom hierarchical (optional):** shared‑memory block reduction → warp‑shuffle (`__shfl_down_sync`) → **single `atomicAdd` per block**.

**Timing (GPU):** CUDA events **around the single reduction**.  
**Energy:** NVML ΔE (mJ→J) over identical host window.  
**Metrics:**
- `kernel_ms`
- `GBps = (N * sizeof(float)) / (kernel_ms / 1e3)`
- `%peak_BW = GBps / peak_BW_GBps`  (peak provided via CLI flag/config)
- `energy_J`, `power_W`, `qc_status` (Δt‑based)

**CSV fields:**
```
device, impl (library|cub|custom), dtype, N,
kernel_ms, GBps, pct_peak_bw,
energy_J, power_W, qc_status, energy_window_s
```

**Acceptance heuristics:**
- **CUB ≥ custom** (within ~5–10%).  
- **50–80%** of theoretical peak BW for well‑implemented reductions (varies by arch and tuning).  
- Power plausible and stable for Δt ≥ 1.2 s.

---

## 7) Definition of Done (per dataset)
- No negative energy deltas; `power_reliable=True` for most GPU rows.  
- For 3090: GPU power never » 350–400 W; median ≈ 200–350 W headless.  
- GEMM: kernel vs. energy-window times consistent; QC distribution sensible.  
- Reduction: GB/s and %peak_BW within expected bands; CUB validates custom/library.

---

## 8) Troubleshooting (fast)
- **Power too high:** ensure **Δt = energy window**, run **headless**, confirm **mJ→J** conversion once.  
- **Unstable readings:** extend Δt (≥1.2 s); verify `cudaEventSynchronize(stop)`; disable desktop/compositor.  
- **1080 Ti (no NVML total energy):** integrate `power.draw` over ≥1 s or omit energy/power and report GB/s only.

---

## 9) Artifacts & paths
- Raw CSVs: `data/raw/energy_benchmark_simple.csv` (GEMM), `data/raw/reduction.csv`
- Derived: harmonized CSVs, summaries, plots (`data/derived`, `figs/`, `reports/`)
