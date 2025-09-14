# Cost- & Carbon-aware CPU vs GPU Selection

Kurze Projektbeschreibung:  
Messungen von CPU vs. GPU bzgl. Energie/Job, Zeit, EDP, Kosten und CO₂. Ziel: Regeln für Gerätewahl ableiten, basierend auf Workloads & Roofline-Einordnung.

---

## Inhaltsverzeichnis / Projektstruktur

- `data/raw/`     — Rohdaten (unbearbeitet)  
- `data/meta/`     — Hardware / Treiber / Umgebungsinformationen  
- `code/measure/`   — Messen via RAPL / NVML Skripte  
- `code/analysis/`   — Analyse-Skripte (LMM, Effektstärken, TOST, Power-Simulation, Diagnostik)  
- `docs/`        — Methoden, Statistik-Plan, Messregeln, Beschreibung des Experiments  
- `figures/`      — Abbildungen, Plots  
- `notebooks/`     — Explorative Notebooks, ggf. Reproduktions-Workflows

---

## Voraussetzungen und Umgebung

- Betriebssystem / Hardwarebasis: z. B. CPU Modell, GPU Modell, Kernel / Treiber Version …  
- Software: R (Version), Python (Version), Bibliotheken & Versionen  
- Optional: Container / virtuelle Umgebung / Docker / Conda falls verwendet  
- Seed / Reproduzierbarkeit / Logging …

---

## Ablauf zur Reproduktion der Ergebnisse

1. Rohdaten sammeln (via `code/measure/`)  
2. QC & Messpfade verifizieren (siehe `docs/METHODS_measuring.md`)  
3. Statistik-Plan anwenden (`stat/ …`)  
4. Analyse via LMM + Baseline vergleichen + TOST + Power-Simulation  
5. Plots & Break-even Regeln erstellen  
6. Kosten- und CO₂-Rechnung parametrieren  
7. Manuskript & Supplement generieren

---

## Lizenz

(Lizenznamen hier einfügen, z. B. MIT / Apache 2.0 / GPLv3 …)

---

## Kontakt

Autor / Team / Rückfragen  
