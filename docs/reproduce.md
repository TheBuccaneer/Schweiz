# reproduce.md — Pilot & Mess-Setup (Stand: 2025-09-18 09:19)

Dieses Dokument beschreibt **reproduzierbar**:
1) Umgebung & Pakete
2) Geräte/Methoden (Zeit/Energie)
3) Messplan (GEMM/Reduction)
4) Pipeline (QC → Harmonisierung → Summaries → Plots → Report)
5) Artefakte (Ablagepfade)
6) Vorher/Nachher‑Skripte (Pre/Post)

---

## 1) Umgebung & Pakete (Linux, CUDA 13, RTX 3090/1080 Ti)

```bash
# (empfohlen) venv aktivieren
python3 -m venv .venv && source .venv/bin/activate
python --version  # >= 3.9

# Kernpakete
pip install -U nvidia-ml-py cupy-cuda13x numpy pandas matplotlib

# Optional (R-Seite war schon genutzt, hier nur der Hinweis)
# R-Pakete: renv, tidyverse, data.table, janitor, ggplot2, etc.
```

**Hinweis:** Für GPU‑Zeitmessung werden **CUDA Events** genutzt (Gerätezeit). Für GPU‑Energie auf **Volta+** (z. B. RTX 3090) wird **NVML `nvmlDeviceGetTotalEnergyConsumption`** als **mJ seit Treiber‑Reload** verwendet (Differenz vor/nach Batch). Auf **Pascal** (1080 Ti) wird **Power‑Integration** über ein ≥1 s‑Fenster genutzt („Average power draw (last second)“ von nvidia‑smi). Für **CPU** wird **Linux powercap/RAPL** `energy_uj` (Δ→J) genutzt.  

Quellen: NVML‑Device‑Queries (TotalEnergy, mJ seit Driver‑Reload); nvidia‑smi Manual (Average Power last second); CUDA Runtime (Event Management); Linux powercap/RAPL (`energy_uj`, `max_energy_range_uj`).

---

## 2) Geräte/Methoden (Messprinzip)

- **GPU‑Zeit:** CUDA‑Events um **gesamten Batch‑Loop** (kein Python‑Overhead).
- **GPU‑Energie (RTX 3090 / Ampere):** **TotalEnergy‑Δ** (mJ→J).
- **GPU‑Energie (GTX 1080 Ti / Pascal):** **Power‑Integration** (≥1 s‑Fenster).
- **CPU‑Energie (Intel/AMD via powercap):** **`energy_uj`‑Δ** → Joule; `max_energy_range_uj` dokumentieren (Wrap‑Around‑Guard).
- **Stabilität:** **Mindestlaufzeit pro Messpunkt ≥ ~1.2 s** (adaptives Macro‑Batching); **Persistence Mode** an.

**Schnelltests**
```bash
# NVML: TotalEnergy‑Capability testen
python - <<'PY'
import pynvml as nv
nv.nvmlInit(); h = nv.nvmlDeviceGetHandleByIndex(0)
try:
    e = nv.nvmlDeviceGetTotalEnergyConsumption(h)
    print("TOTAL_ENERGY_SUPPORTED", e, "mJ_since_driver_reload")
except nv.NVMLError as err:
    print("TOTAL_ENERGY_NOT_SUPPORTED", type(err).__name__)
nv.nvmlShutdown()
PY

# Persistence Mode checken/setzen
nvidia-smi -q -d POWER | sed -n '1,120p'
sudo nvidia-smi -pm 1
```

---

## 3) Messplan (heute)

**Workloads**
- **GEMM** (compute‑bound, FP32, quadratisch N×N)
- **Reduction** (memory‑bound, Summe über N Elemente)

**Größenraster**  
- **GEMM N:** 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792  
- **Reduction N:** 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 4096, 8192, 16384, 32768, 65536

**Replikate**  
- je Größe/Device **3 Makro‑Runs**; innerhalb eines Makro‑Runs **batch_size adaptiv erhöhen**, bis **CUDA‑Events ≥ 1.2 s**.

**Geräte‑Matrix**
- **GPUs:** RTX 3090 (TotalEnergy‑Δ) und GTX 1080 Ti (Power‑Integration ≥1 s)
- **CPUs:** Intel und AMD (sofern powercap verfügbar)

---

## 4) Pipeline (gestern abgeschlossen, heute unverändert)

### 4.1 Daten → QC → Harmonisierung (R)
- Eingaben (standardisiert): `data/derived/pilot_standardized.csv`
- Harmonisierung: GPU‑Energie **nur** plausibler **NVML‑TotalEnergy‑Δ** (10–450 W), CPU via **RAPL**  
- Ergebnis: `data/derived/pilot_harmonized.csv`

### 4.2 Summaries & Vergleiche
- `data/derived/pilot_summary_by_dws.csv` (robuste Kennwerte je {workload,size,device})  
- `data/derived/pilot_comp.csv` (CPU↔GPU Gegenüberstellung, Speedup/Energie/EDP)

### 4.3 Plots (GEMM)
- `figs/pilot_gemm_speedup.png` (Speedup: CPU_time/GPU_time)  
- `figs/pilot_gemm_energy_ratio.png` (Energie‑Verhältnis: CPU/GPU)

### 4.4 QC‑Plot (Methodenvergleich)
- `figs/qc_gpu_power_by_method.png` (p_W vs. energy_method, used/discarded)

### 4.5 Mini‑Report
- `reports/pilot_mini_report.Rmd` (Tabelle + Plots + QC‑Abbildung)
- Entscheidungen: `reports/pilot_qc_decisions.txt`  
- Artefaktliste: `reports/pilot_artifacts.txt`  
- Nächste Schritte: `reports/next_run_todo.txt`

---

## 5) Artefakte (Pilot, bereits erstellt)

```
data/derived/pilot_harmonized.csv
data/derived/pilot_summary_by_dws.csv
data/derived/pilot_comp.csv
data/derived/plan_next_sizes_gemm.csv
data/derived/plan_next_sizes_reduction.csv
data/derived/plan_next_runs.csv
reports/appendix_pilot_summary.csv
reports/pilot_qc_decisions.txt
reports/pilot_mini_report.Rmd
reports/pilot_artifacts.txt
reports/next_run_todo.txt
figs/pilot_gemm_speedup.png
figs/pilot_gemm_energy_ratio.png
figs/qc_gpu_power_by_method.png
```

**Versionierung**
```bash
git add data/derived/*.csv reports/*.Rmd reports/*decisions*.txt reports/*artifacts*.txt figs/*.png
git commit -m "Pilot QC + harmonized metrics; plans and figures frozen"
```

---

## 6) Vorher-/Nachher‑Skripte (Pre/Post)

### 6.1 Vor der Messung (`scripts/pre_env.sh`)
- GPU‑Persistenz & Power‑Limit loggen: `nvidia-smi -q -d POWER`
- Treiber/CUDA/Device‑UUID: `nvidia-smi --query-gpu=name,uuid,driver_version --format=csv`
- CPU‑RAPL‑Domains & Rechte: `ls -R /sys/class/powercap` (ggf. `energy_uj` Leserechte prüfen)
- Python‑Pakete/Versionen: `pip freeze | sort`
- Zeitstempel/Host/OS in `logs/pre_post/pre_{ts}.txt`

### 6.2 Nach der Messung (`scripts/post_env.sh`)
- NVML‑Energiezähler (nur Info): `python -c '...nvmlDeviceGetTotalEnergyConsumption...'`
- Temperatur/Clocks snapshot: `nvidia-smi -q -d TEMPERATURE,CLOCK`
- RAPL‑Counter‑Snapshot: `for f in /sys/class/powercap/intel-rapl*/**/energy_uj; do echo $f: $(cat $f); done`
- Zeitstempel/Host/OS in `logs/pre_post/post_{ts}.txt`

---

## 7) Erwartete Plausibilitäten (Schnellabgleich)

- **GPU‑Leistung (3090, TotalEnergy‑Δ):** Median **~200–350 W**, keine Ausreißer > 450 W.
- **CPU‑Leistung (RAPL):** grob **5–200 W** je nach Last/Threads.
- **GEMM 2000–8000 (Pilot):** GPU dominiert (kein Break‑even dort); Break‑even wird **<2000** erwartet.
- **Reduction:** deutlich kleinere Speedups möglich; Energie‑Break‑even stärker größenabhängig.

---

## 8) Start (heute)

1. **RTX 3090:** `logger5.py` mit `plan_next_runs.csv` starten → GEMM (64–1792), dann Reduction (64–65536); je Größe 3 Makro‑Runs, **Events ≥ 1.2 s**.  
2. **GTX 1080 Ti:** gleiche Reihenfolge; Energie via Power‑Integration ≥ 1 s.  
3. **Intel/AMD‑CPU:** gleiche Größen; Energie via RAPL `energy_uj`‑Δ (falls verfügbar).

**Tip:** Für den ersten Punkt (GEMM N=256) sollte das Log enthalten: Events‑Zeit ≈ 1.0–1.3 s, ΔE>0, p_W ≈ 150–300 W (3090).

---

## 9) Quellen/Hinweise (Kurzbelege)

- **NVML TotalEnergy (mJ seit Driver‑Reload; Volta+ fully supported)** — NVIDIA NVML „Device Queries“.  
- **nvidia‑smi „Average Power Draw (last second)“** — NVIDIA nvidia‑smi Manual.  
- **CUDA Events (Event Management; Gerätezeit, ~0.5 µs)** — CUDA Runtime API.  
- **Linux powercap/RAPL (`energy_uj`, `max_energy_range_uj`; kein `power_uw` bei Intel)** — Kernel Doku.

```text
(Die exakten Links sind im Projekt‑Chat/Citations hinterlegt.)
```
