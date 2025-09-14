# STAT_plan.md — Statistischer Analyseplan (SAP)

## Zweck und Bedeutung
Der **Statistische Analyseplan (SAP)** legt *vor* der eigentlichen Auswertung fest, wie Daten analysiert und interpretiert werden. Ziel: Transparenz, Reproduzierbarkeit, weniger Bias (kein „p-hacking“) und klare Entscheidungsregeln.

---

## 1) Fragestellungen & Hypothesen
**Primärfrage:** Ist die **Energie pro Job** auf der **GPU** im Vergleich zur **CPU** bei *großen* Probleminstanzen *signifikant und praktisch relevant* niedriger?
- H0 (Überlegenheit): Δ_Energie = 0
- H1 (GPU < CPU): Δ_Energie < 0
- **SESOI** (praktische Relevanz): ±5 % relativ zu CPU-Mittel (anpassbar; Begründung in Methods).

**Sekundärfragen:** (Beispiele)
- Kleine Instanzen: GPU-Overheads → wann **praktische Gleichheit** (Äquivalenz) vorliegt?
- Transfer-/irreguläre Workloads (z. B. SpMV): ist CPU effizienter?
- Zeit pro Job, **EDP = Energie × Zeit**, €/Job, gCO₂/Job als zusätzliche Outcomes.

---

## 2) Ergebnisgrößen (Outcomes)
- **Energy/job [J]** (Primär-Outcome)
- **Time/job [s]**
- **EDP [J·s]**
- **€/Job**, **gCO₂/Job** (parametrisiert: Strompreis, Grid-CO₂-Faktor)
- *Berechnung Energie:* Differenz aus Energiezählern (CPU: RAPL `energy_uj`; GPU: NVML TotalEnergy in mJ).

---

## 3) Design, Stichprobe & Power
- Faktoren: **Device ∈ {CPU, GPU} × Size ∈ {small, large}**; Workload als **Gruppe**.
- Wiederholungen pro Zelle: **n = 20** (anpassbar nach Pilot-Varianz).
- **Power-Analyse (simulationsbasiert)** für das Primärmodell (LMM; s. Abschnitt 4) mit realistischen Effektgrößen (z. B. 5–10 % Energie-Δ) und Varianzschätzungen aus Pilotdaten. Ziel-Power **0.8**.

Artefakte:
- `code/analysis/03_power_simr.R` (R + {lme4, simr}).
- Dokumentiere Input-Parameter (Effekt, Varianz, Korrelation innerhalb Workloads).

---

## 4) Statistische Methoden
### 4.1 Primärmodell (LMM)
Lineares Mixed-Effects Modell:
```
Energy_per_job ~ Device * Size + (1 | Workload)
```
Optional: zufällige Steigung für Size je Workload:
```
Energy_per_job ~ Device * Size + (1 + Size | Workload)
```

- **Schätzung:** R: `lme4::lmer()`
- **Bericht:** Punktschätzer, 95 %-Konfidenzintervalle, **Effektstärken** (z. B. Hedges’ g, mit Bias-Korrektur).
- **Inference:** Likelihood-Ratio-Tests / Kenward-Roger/Satterthwaite-df (transparent angeben).
- **Modellwahl:** Primärformel *vorab* fixiert; Abweichungen begründen.

### 4.2 Äquivalenztests (TOST)
Für Aussagen **„praktisch gleich“** nutze **TOST** mit **SESOI**-Grenzen (z. B. ±5 % relativ). Implementierung in R (z. B. `TOSTER`-Paket oder manuell).

### 4.3 Multiple Tests
Wenn zusätzlich Einzelvergleiche (pro Workload/Größe) nötig sind:
- **Holm-Korrektur** (FWER) *oder* **Benjamini–Hochberg** (FDR). Vorgehen *vorab* festlegen.

### 4.4 Sekundär-Outcomes
Gleiche Modellstruktur separat für **Time/job**, **EDP**; für **€/Job** und **gCO₂/Job** identische Tests (Parameter klar nennen).

---

## 5) Diagnostik & Qualitätskontrolle
- **Residuen-Diagnostik:** QQ-Plot (Normalität), Residuen vs. Fits (Homoskedastizität), Einflusspunkte (Cook’s D / dfbetas).
- **Robuste SEs / Varianzstruktur**, falls Heteroskedastizität.
- **Ausreißerregeln:** vorab definieren (z. B. > Q3+3·IQR oder technische Artefakte). Immer protokollieren, Ausschlüsse begründen.
- **Reproduzierbarkeit:** Seeds, Software-/Treiber-Versionen, HW-Details ins Supplement.

Artefakte:
- `code/analysis/04_plots_diagnostics.R` (Residuendiagnostik).

---

## 6) Baselines & Vorhersagen
- **Naive Heuristiken:** „GPU ab N“, „CPU bei irregulär“ – N aus Pilotdaten.
- **Roofline-abgeleitete Vorhersage** (Operational Intensity + BW/Peak) → **Break-even** schätzen und mit Messung vergleichen (Δ).
- **Literatur-Regeln** (sofern vorhanden) replizieren/prüfen.

Artefakte:
- `docs/BASELINES.md`, Plots in `figures/`.

---

## 7) Ergebnisdarstellung
- **Plots:** Energie vs. Größe (CPU/GPU), **Pareto** (Zeit↔Energie), **Break-even** je Profil; Roofline-Inset.
- **Tabellen:** Effektschätzer, 95 %-CIs, p-Werte (adj.), **TOST-Ergebnisse**; €/Job, gCO₂/Job.
- **Regeln (3–5)** für die Gerätewahl, textlich präzise (mit Gültigkeitsbereich).

---

## 8) Sonderfälle
- **Fehlende/fehlerhafte Messungen:** definierte Re-Runs; Dokumentation im QC-Log.
- **Zähler-Spezifika:** RAPL-Wrap-Around handhaben (`max_energy_range_uj`), NVML-TotalEnergy (mJ seit Treiberstart).
- **Kurzläufer:** ggf. Batching/Mehrfachläufe, um Zählerauflösung zu überwinden.

---

## 9) Software & Versionen
- **R**: lme4, lmerTest/emmeans (optional), effectsize, TOSTER, simr
- **Python** (optional): statsmodels mixedlm
- **System**: CPU/GPU-Modelle, Kernel/Treiber, BLAS/cuBLAS/NVML Versionen

Alle Versionen im Supplement listen (`docs/SUPPLEMENT_versions.md`).

---

## 10) Entscheidungen, die *vor* der Analyse fixiert werden
- SESOI-Grenzen (Standard: ±5 % Energie-Δ; projektspezifisch anpassbar).
- Primärmodellformel, Korrekturverfahren (Holm *oder* BH), Diagnosekriterien.
- Kriterien für Ausschlüsse und Re-Runs.

---

## 11) Quellen (Auswahl)
- **Mixed-Effects Modelle (R; lme4):** Bates et al., *Journal of Statistical Software* 2015.
- **Power per Simulation (simr):** Green & MacLeod, *Behavior Research Methods* 2016.
- **Äquivalenztests (TOST):** Lakens, *Social Psychological and Personality Science* 2017; Tutorials 2018.
- **Multiple Testing:** Holm (1979), Benjamini–Hochberg (1995).
- **Roofline-Modell:** Williams, Waterman, Patterson (2009).
- **RAPL/NVML Grundlagen:** Linux powercap / NVIDIA NVML Doku.

*Hinweis:* Vollzitate/Links im Manuskript/Supplement; dieser SAP referenziert etablierte Primärquellen.
