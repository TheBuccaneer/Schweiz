# Projektplan v3 — **Cost- & Carbon‑aware CPU‑vs‑GPU Selection**
*(Ergänzung/Überarbeitung des v2-Plans um erweiterte Stichprobe, Workload-Diversität und statistische Robustheit)*

> Bezug: **Projektplan v2** + **Kritik-basierte Erweiterungen** (Stichprobengröße, Workload-Diversität, statistische Power).

---

## 0) Projektziel (konstant)
Für typische Profile (compute, memory, transfer, irregulär, *mixed*) messen wir **Energie/Job**, **Zeit**, **EDP** und leiten **Break‑even‑Punkte** ab, um **3–5 Regeln** zur Gerätewahl (CPU vs. GPU) abzuleiten. Einordnung via **Roofline/Operational Intensity**. Messpfade: **RAPL/powercap** (CPU) und **NVML TotalEnergy** (GPU).

**Primäre Ergebnisgrößen**
- *Energy/job* (J), *Time/job* (s), *EDP = Energy×Time*
- *€/Job*, *gCO₂/Job* als **parametrisierte** Größen (Strompreis & Grid‑Faktor sind Eingabeparameter)

---

## 1) Arbeitspakete (AP) & Deliverables

### AP0 – Repo/Logistik & Templates (0.5 Tage) — *Pflicht*
- Repo‑Skeleton, **CSV‑Schema** (Rohdaten), **Notebooks/Plots** (Zeit, Energie, EDP, Pareto, Break‑even).
- Parametertabelle für **€/kWh** und **gCO₂/kWh** (Defaultwerte nur als Platzhalter; im Text explizit als *Input* gekennzeichnet).
**Deliverables:** Repo‑Skeleton + CSV‑Vorlage + `README_data.md`.

### AP1 – Mess‑Harness robust (2–3 Tage) — *Pflicht*
- **CPU/RAPL (powercap):** `energy_uj` + `max_energy_range_uj` (Overflow‑Guard). Instant‑Power **nicht verfügbar** ⇒ Energie via Differenzen.
- **GPU/NVML:** primär `nvmlDeviceGetTotalEnergyConsumption()` (mJ seit Driver‑Reload; Differenzen je Run). Sampling‑Fenster nur dokumentieren, *keine* 1 Hz‑Falle für Kurz‑Kernels.
- **Messumgebung stabilisieren:** Performance Governor auf 'performance', CPU-Frequenz fixieren, GPU im Persistence Mode, Thermal-Throttling monitoren, 5-min Aufwärmen vor Messungen.
- **Plausi‑Check:** 30–60 s steady‑state CPU/GPU Last ggf. gegen Steckdosen‑Messung vergleichen (nur Plausibilität).
**Deliverables:** Logging‑Skripte (RAPL/NVML) + `METHODS_measuring.md` (Wrap/Sampling/Counter‑Notes + Umgebungskontrolle).

### AP1b – **Statistik‑Pipeline** (1.5–2 Tage) — *NEU/Pflicht*
**Analyseplan (präregistrationsartig)**
- **Primärmodell (LMM):** `Energy_per_job ~ Device (CPU/GPU) + Size (small/large) + Device×Size + (1|Workload)`; optional `(Size|Workload)`.
- **SESOI explizit:** ±5% Energie-Delta als praktisch relevante Schwelle (a-priori festgelegt).
- **Effektstärken:** standardisierte **Hedges' g** + 95%-CIs.
- **Äquivalenz (für „praktisch gleich"):** **TOST** mit SESOI-Grenzen.
- **Multiple Tests:** **Holm-Bonferroni** (FWER-Kontrolle) für Sekundärvergleiche.
- **Power‑Begründung:** **simulationsbasierte Power‑Analyse** (R: *lme4* + *simr*) mit Ziel ≥80% Power für SESOI-Effekte.
- **Stichprobenplanung:** Pilot-basierte Varianzschätzung → finale n-Berechnung für **≥800 Beobachtungen**.
- **Diagnostik/QC:** Residuen‑QQ, Residuen vs. Fits, Influenz; Heteroskedastizität prüfen (robuste SEs falls nötig).
**Deliverables:** `STAT_plan.md`, R‑Skripte (`01_pilot_variance.R`, `02_power_analysis.R`, `03_fit_lmm.R`, `04_effectsizes_tost.R`, `05_plots_diagnostics.R`), `STAT_report.html`.

### AP2 – **Baseline‑Block** (2 Tage) — *NEU/Pflicht*
- **Naive Heuristiken:** z. B. „immer GPU ab N", „CPU bei *irregular*"; Parameter N über Pilotdaten kalibrieren.
- **Roofline‑abgeleitete Vorhersage:** Break‑even‑Schätzung aus OI & Bandbreiten/Peaks; **Vergleich Messung vs. Vorhersage** (Δ).
- **Literatur‑Regeln** (falls vorhanden) kurz replizieren.
**Deliverables:** 2–3 Baseline‑Plots (BW/GEMM/mini‑Roofline), Tabelle „Prediction vs. Measured Break‑even", kurze Takeaways.

### AP3 – **Erweiterte Workload‑Suite & Messmatrix** (6–8 Tage) — *Pflicht, erweitert*
- **Workloads (7 statt 5):** 
  - W1: GEMM (compute-intensiv)
  - W2: STREAM‑Triad (CPU) & GPU‑Copy/Triad (memory-bound)
  - W3: SAXPY+Memcpy (transfer-bound)
  - W4: SpMV×2 (SuiteSparse, irregular/sparse)
  - W5: CNN‑Inference (ResNet‑18, mixed)
  - W6: FFT (compute + memory, regular pattern)
  - W7: Random Memory Access (irregular memory-bound)
- **Design:** Device∈{CPU,GPU} × Size∈{small,medium,large} × **25 Repeats** ⇒ **7×2×3×25 = 1050 Runs**.
- **Qualitätskontrolle:** Varianz pro Konfiguration ≤15% CI-Halbbreite; Ausreißer-Detektion (>3σ); Wiederholung instabiler Messungen.
**Deliverables:** Vollständige CSV‑Rohdaten + **QC‑Bericht** (Varianz, Ausreißerregeln, Stabilität).

### AP3b – **Pilot-zu-Haupt Übergang** (1 Tag) — *NEU/Pflicht*
- **Pilot-Analyse:** Erste 2-3 Workloads mit je 10 Repeats → Varianzschätzung.
- **Power-Simulation:** Basierend auf Pilot-Varianz → finale Repeat-Anzahl bestimmen.
- **Anpassung:** Falls Power <80%, Repeats auf 30-35 erhöhen.
**Deliverables:** Power-Analyse-Report, finale Messmatrix-Spezifikation.

### AP4 – **Analyse & Regeln** (4 Tage) — *Pflicht, erweitert*
- **Plots:** Energie‑vs‑Größe (CPU/GPU), **Pareto** (Zeit↔Energie), **Break‑even** pro Profil; Roofline‑Inset.
- **LMM-Ergebnisse:** Effektstärken mit CIs, TOST-Äquivalenzbefunde, Device×Size Interaktionen.
- **Regeln (3–5):** aus LMM‑Schätzungen + TOST‑Befunden ableiten („praktisch gleich bis N*", "GPU vorteilhaft ab Größe X für Workload-Typ Y").
- **Sensitivitätsanalyse:** Robustheit der Regeln bei verschiedenen SESOI-Schwellen (±3%, ±7%, ±10%).
- **Business‑Layer:** €/Job & gCO₂/Job in alle Tabellen/Plots (Parameter sichtbar).
**Deliverables:** finalisierte Abbildungen + **Regel‑Boxen** + Sensitivitäts-Tabellen.

### AP5 – **Schreiben** (5–6 Tage) — *Pflicht, erweitert*
- **Struktur (≈12 S.):** Einleitung (Green/€), Related (kurz), Methodik (RAPL/NVML + Stichprobenplanung), Ergebnisse (LMM + Effektstärken), **Regeln/Checkliste**, Limitations, Threats (Counter‑Bias/Sampling/HW‑Spezifik), Fazit.
- **Methodentransparenz:** Vollständige Beschreibung der Power-Analyse, SESOI-Begründung, Workload-Auswahl.
- **Limitations-Abschnitt:** Generalisierbarkeit auf andere Hardware, kleine Effektgrößen, Messunsicherheit.
- **Supplement:** Befehle, Versionen, Seeds; Statistik‑Plan; Power‑Simulationsergebnisse; alle QC-Reports.
**Deliverables:** Manuskript + Supplement + Response-to-Reviewers Template.

### AP6 – **Erweiterungen** (2–4 Tage) — *Optional, parallelisierbar*
- **E1 DVFS/Power‑Cap:** je Profil **ein** zusätzlicher Punkt (z. B. `cpupower …` / `nvidia-smi -pl …`).
- **E2 Zweite Plattform/Generation:** kurze Replikation (je Workload 10 Repeats, nur „large").
- **E3 Längere Workloads:** 1-2 Stunden-Jobs für Thermal-Charakterisierung.

### AP7 – **Threat‑Modell & Validierung** (1.5 Tage) — *NEU/Pflicht, erweitert*
- **Thermal‑Throttling / Clocks:** Kontinuierliches Monitoring, Logs + Gegenprobe unter Power‑Cap/DVFS.
- **Counter‑Bias:** RAPL/NVML Limitierungen offen dokumentieren, inkl. aktueller Validierungsarbeiten.
- **Messumgebung-Validierung:** Reproduzierbarkeit über mehrere Sessions, Tageszeit-Effekte.
- **Hardware-Spezifität:** Diskussion der Generalisierbarkeit auf andere CPU/GPU-Generationen.
- **Plausibilisierung:** Steckdosen‑Mitschnitt für 3-5 repräsentative Jobs.
**Deliverables:** Threat-Model-Report, Validierungs-Plots, Hardware-Spezifitäts-Diskussion.

### AP8 – **Business‑Case quantifizieren** (1 Tag) — *NEU/erweitert*
- **€/Job & gCO₂/Job:** Sensitivität ggü. Strompreis (0.10-0.40 €/kWh)/CO₂‑Intensität (200-600 gCO₂/kWh).
- **ROI‑Analyse:** GPU‑Investment-Rechnung über realistische Workload-Mixes.
- **Szenario-Analyse:** Verschiedene Nutzungsprofile (HPC, Cloud, Edge).
**Deliverables:** Business-Case-Spreadsheet + Grafiken + Szenario-Report.

---

## 2) Zeitplan (Wochen, mit Quality‑Gates)

**Woche 1 (AP0–AP1, Start AP1b)**  
Gate **G1**: Harness stabil, Messumgebung konfiguriert; erste Pilotdaten (2-3 Workloads × 10 Repeats) mit plausiblen ΔE und Varianz <20%.

**Woche 2 (AP1b–AP2–AP3b)**  
Gate **G2**: Power-Analyse abgeschlossen, finale Messmatrix definiert; Baselines etabliert; erste 200 Messungen gesammelt.

**Woche 3-4 (AP3)**  
Gate **G3**: ≥70% aller geplanten Messungen (≥700 von 1050) abgeschlossen; QC-Bericht zeigt Varianz ≤15% für Energie-Messungen.

**Woche 5 (AP3–AP4–AP7)**  
Gate **G4**: Vollständige Messungen; erste LMM-Fits mit Effektstärken; Threat-Modell dokumentiert.

**Woche 6 (AP4–AP5)**  
Gate **G5**: Kern-Plots und Regeln finalisiert; Draft-Manuskript mit Methods und Results.

**Woche 7 (AP5, ggf. AP6/8)**  
Gate **G6**: Final-Draft + Supplement + alle Reports; optionale Erweiterungen integriert.

---

## 3) Erweiterte Quality‑Hygiene & Messregeln

### Mess-Stabilität
- **Performance Governor:** 'performance' mode für CPU, GPU im Persistence Mode
- **Frequency Locking:** CPU auf max non-turbo frequency, GPU memory/core clocks dokumentiert
- **Thermal Control:** 5-min Aufwärmen, Temperatur-Monitoring, Throttling-Detektion
- **Background Control:** Minimaler System-Load, dedicated Messsystem

### Energie-Messungen
- **RAPL:** `energy_uj` + `max_energy_range_uj` mit Overflow-Detektion; **keine** `power_uw`‑Attribute
- **NVML:** **TotalEnergy** (mJ seit Driver‑Reload) als Primärquelle; Sampling-Rate dokumentiert
- **Validierung:** Negative Δ-Erkennung, Plausibilitäts-Checks gegen theoretische Bounds

### Workload-Spezifikationen
- **STREAM‑Größe:** Arrays ≥ **4× Summe der LLC‑Größen** (oder ≥ 2M Elemente für 'large')
- **GEMM:** Optimierte BLAS-Bibliotheken (OpenBLAS/MKL vs cuBLAS)
- **SuiteSparse:** Matrizen mit SSMC‑IDs; verschiedene Sparsity-Pattern
- **CNN:** Standardisierte Modelle (ResNet-18), definierte Batch-Sizes

### Statistische Transparenz
- **SESOI:** ±5% Energie-Delta a-priori festgelegt und begründet
- **Power-Ziel:** ≥80% für SESOI-Effekte, simulationsbasiert validiert
- **Multiple Testing:** Holm-Korrektur für alle Sekundärvergleiche
- **Repro-Artefakte:** Seeds, Versionen, Hardware-Konfiguration vollständig dokumentiert

---

## 4) Erweiterte Daten‑ und Ordnerstruktur

```
repo/
  data/
    raw/
      pilot_*.csv           # Pilot-Messungen
      main_*.csv            # Haupt-Messungen  
      qc_reports/           # Quality Control Reports
    processed/
      cleaned_data.csv      # Nach QC
      power_analysis/       # Power-Simulationen
    meta/
      hardware_config.json  # Detaillierte HW-Specs
      environment_log.csv   # Temperatur, Frequenzen, etc.
  code/
    measure/
      pilot_logger.py       # Pilot-Messungen
      main_logger.py        # Erweiterte Messungen
      environment_monitor.py # System-Monitoring
    analysis/
      01_pilot_analysis.R   # Pilot-Varianz-Analyse
      02_power_simulation.R # Power-Analyse mit simr
      03_data_cleaning.R    # QC und Bereinigung
      04_fit_lmm.R         # Haupt-LMM-Analyse
      05_effectsizes_tost.R # Effektstärken und TOST
      06_plots_results.R    # Finale Visualisierungen
      07_sensitivity.R      # Sensitivitäts-Analysen
  docs/
    METHODS_measuring.md
    STAT_plan.md
    POWER_analysis.md        # Power-Analyse-Report
    LIMITATIONS.md           # Bekannte Limitationen
    THREAT_model.md          # Erweiterte Threat-Analyse
    README_data.md
  reports/
    QC_report.html          # Quality Control
    STAT_report.html        # Statistische Analyse
    BUSINESS_case.html      # ROI-Analyse
  figures/
    exploratory/            # Pilot-Plots
    main_results/           # Publikations-Grafiken
    supplementary/          # Zusätzliche Analysen
  notebooks/
    pilot_exploration.ipynb
    main_analysis.ipynb
    business_calculations.ipynb
```

**Erweitertes CSV‑Schema:**
`timestamp, session_id, workload, workload_params, size, device, repeat, energy_j, time_s, edp, cpu_temp, gpu_temp, cpu_freq, gpu_freq_core, gpu_freq_mem, throttled, notes`

---

## 5) Detaillierte Stichproben-Planung

### Pilot-Phase (AP1b/AP3b)
- **3 Workloads** × 2 Devices × 3 Größen × **10 Repeats** = **180 Pilot-Messungen**
- Ziel: Varianz-Schätzung für Power-Analyse
- Dauer: 1-2 Tage intensive Messung

### Haupt-Phase (AP3)
- **7 Workloads** × 2 Devices × 3 Größen × **25+ Repeats** = **≥1050 Messungen**
- Basierend auf Pilot-Power-Analyse ggf. auf 30-35 Repeats erhöht
- Ziel: ≥80% Power für ±5% Energie-Effekte
- Dauer: 4-5 Tage intensive Messung

### Quality Gates
- **Varianz-Kriterium:** CV ≤15% für Energie-Messungen pro Konfiguration
- **Stabilität:** Keine systematischen Trends über Sessions
- **Vollständigkeit:** ≥95% erfolgreiche Messungen (ohne Hardware-Fehler)

---

## 6) Kernquellen (für Methods/Appendix)

### Grundlagen (wie v2)
- **RAPL/powercap:** https://docs.kernel.org/power/powercap/powercap.html
- **NVML TotalEnergy:** https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
- **Roofline:** https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf

### Statistische Methoden
- **LMM Power-Analyse:** Green & MacLeod (2016), *simr* package
- **SESOI & TOST:** Lakens (2017, 2018) Equivalence Testing Tutorials
- **Sample Size Mixed Models:** Brysbaert & Stevens (2018), "Power Analysis and Effect Size in Mixed Effects Models"
- **Multiple Testing:** Holm (1979), Benjamini-Hochberg (1995)

### Neue Validierungs-Quellen
- **Energy Measurement Validation:** "Can we trust our energy measurements?" (arXiv:2206.10377)
- **RAPL Accuracy:** SPEC ICPE 2024 Proceedings
- **Mixed Model Sample Size:** Kumle et al. (2021), "Estimating power in (generalized) linear mixed models"

---

### Implementation Notes
- **R-Pipeline:** `lme4` + `simr` + `effectsize` + `TOSTER` für vollständige Analyse-Chain
- **Reproducibility:** Alle Seeds, Package-Versionen, Hardware-Configs dokumentiert
- **Automation:** Batch-Scripts für wiederholbare Messdurchläufe
- **Backup:** Kontinuierliche Datensicherung während langer Messreihen

---

**Zusammenfassung v2→v3:**
- **+470 zusätzliche Messungen** (von 580 auf 1050+)
- **+2 neue Workloads** für bessere Random-Effect-Abdeckung  
- **Explizite Power-Analyse** mit Pilot-basierter Stichprobenplanung
- **Erweiterte QC** und Mess-Stabilisierung
- **Vollständige Limitations-Diskussion** für methodische Transparenz

Dieser Plan adressiert alle identifizierten statistischen und methodischen Kritikpunkte und stellt eine publikationsfähige, robuste Studie sicher.