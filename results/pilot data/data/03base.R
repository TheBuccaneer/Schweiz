setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(readr)
library(janitor)

df <- readr::read_csv("data/derived/pilot_standardized.csv", show_col_types = FALSE) |>
  janitor::clean_names()

required <- c("device","workload","size","time_s","energy_j")
missing  <- setdiff(required, names(df))
if (length(missing) > 0) stop(paste("Fehlende Spalten:", paste(missing, collapse = ", ")))

message("3.1 OK")

# 3.2: Numerische Typen + Mini-Summary
num_cols <- intersect(c("time_s","energy_j","power_w","size","repeats","batch_size"), names(df))
for (nm in num_cols) {
  df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
}

summary_32 <- data.frame(
  n          = nrow(df),
  min_time   = min(df$time_s, na.rm = TRUE),
  med_time   = median(df$time_s, na.rm = TRUE),
  max_time   = max(df$time_s, na.rm = TRUE),
  min_energy = min(df$energy_j, na.rm = TRUE),
  med_energy = median(df$energy_j, na.rm = TRUE),
  max_energy = max(df$energy_j, na.rm = TRUE)
)
print(summary_32)
message("3.2 OK")


p <- df$energy_j / df$time_s
summary(p[is.finite(p)])
message("3.3 OK")

# 3.2b: Ausreißer nach Leistung (W) inspizieren
library(dplyr)

# Zeilennummer einmalig anheften
df <- df |> dplyr::mutate(.row = dplyr::row_number())

# höchste p_w
sus <- df |>
  dplyr::mutate(p_w = energy_j / time_s) |>
  dplyr::arrange(dplyr::desc(p_w)) |>
  dplyr::select(.row, device, workload, size, time_s, energy_j, p_w,
                dplyr::any_of(c("energy_method","timing_method","batch_size","repeat_id","notes"))) |>
  head(8)

# niedrigste p_w
low <- df |>
  dplyr::mutate(p_w = energy_j / time_s) |>
  dplyr::arrange(p_w) |>
  dplyr::select(.row, device, workload, size, time_s, energy_j, p_w,
                dplyr::any_of(c("energy_method","timing_method","batch_size","repeat_id","notes"))) |>
  head(8)

print(sus); cat("\n---- LOW ----\n"); print(low)
message("3.2b OK")

# 3.3: Leistung (W) berechnen und unplausible Werte flaggen
library(dplyr)

df <- df |>
  mutate(
    p_w = energy_j / time_s,
    # einfache device-basierte Plausibilitätsgrenzen
    p_lo = ifelse(device == "GPU", 10, 5),    # sehr konservativ
    p_hi = ifelse(device == "GPU", 450, 200), # 3090 ~350W Board-Power
    qc_power_unreal = is.finite(p_w) & (p_w < p_lo | p_w > p_hi),
    qc_reason = ifelse(qc_power_unreal,
                       paste0("p_w=", round(p_w,1), "W outside [", p_lo, ",", p_hi, "] via ", energy_method),
                       NA_character_)
  )

# Kurzüberblick
by_device <- df |>
  summarise(
    n = n(),
    flagged = sum(qc_power_unreal, na.rm=TRUE),
    .by = device
  )

head_flagged <- df |>
  filter(qc_power_unreal) |>
  arrange(desc(p_w)) |>
  select(.row, device, workload, size, time_s, energy_j, p_w,
         any_of(c("energy_method","timing_method","batch_size","repeat_id","notes"))) |>
  head(6)

print(by_device); cat("\n--- Beispiele (flagged) ---\n"); print(head_flagged)
message("3.3 OK")


# 3.3a: GPU—Verteilung der Flags je energy_method + Medianleistung
library(dplyr)

diag_em <- df |>
  filter(device == "GPU") |>
  count(energy_method, qc_power_unreal)

med_pw <- df |>
  filter(device == "GPU") |>
  group_by(energy_method) |>
  summarise(n = dplyr::n(),
            med_p_w = median(p_w, na.rm = TRUE),
            .groups = "drop")

print(diag_em); cat("\n--- Median p_w (W) je Methode ---\n"); print(med_pw)
message("3.3a OK")


# 3.3b: Harmonisierung der Energiequelle
library(dplyr)

df <- df |>
  mutate(
    p_w = energy_j / time_s,  # falls noch nicht vorhanden
    use_energy = case_when(
      device == "GPU" & energy_method == "nvml_total_energy" &
        is.finite(p_w) & p_w >= 10 & p_w <= 450 ~ TRUE,
      device == "CPU" & energy_method %in% c("rapl", "rapl_pkg") ~ TRUE,
      TRUE ~ FALSE
    ),
    energy_harmonized_j = ifelse(use_energy, energy_j, NA_real_),
    p_w_harmonized      = ifelse(use_energy, energy_harmonized_j / time_s, NA_real_)
  )

table_use <- df |> count(device, use_energy)
summary_h <- df |> filter(use_energy) |>
  group_by(device) |>
  summarise(n = dplyr::n(),
            med_p_w = median(p_w_harmonized, na.rm = TRUE),
            .groups = "drop")

print(table_use); cat("\n--- med_p_w (nur verwendete) ---\n"); print(summary_h)
message("3.3b OK")


# 3.4: Analyse-Frame (harmonisiert) + EDP berechnen + exportieren
library(dplyr)
library(readr)

df_use <- df |>
  filter(use_energy, is.finite(time_s), is.finite(energy_harmonized_j)) |>
  mutate(edp_j_s = energy_harmonized_j * time_s) |>
  select(.row, device, workload, size, time_s,
         energy_j_raw = energy_j,
         energy_method,
         energy_j = energy_harmonized_j,
         p_w = p_w_harmonized,
         any_of(c("batch_size","repeat_id","timing_method","notes")),
         edp_j_s)

# Kurzüberblick
tab_use <- df_use |> count(device)

# Export
dir.create("data/derived", showWarnings = FALSE)
readr::write_csv(df_use, "data/derived/pilot_harmonized.csv")

print(tab_use)
message("3.4 OK")


# 3.5: robuste Kennwerte + Export
library(dplyr)
library(readr)

summary_dws <- df_use |>
  group_by(workload, size, device) |>
  summarise(
    n              = dplyr::n(),
    med_time_s     = median(time_s, na.rm = TRUE),
    mad_time_s     = mad(time_s, constant = 1.4826, na.rm = TRUE),
    med_energy_j   = median(energy_j, na.rm = TRUE),
    mad_energy_j   = mad(energy_j, constant = 1.4826, na.rm = TRUE),
    med_edp        = median(edp_j_s, na.rm = TRUE),
    mad_edp        = mad(edp_j_s, constant = 1.4826, na.rm = TRUE),
    med_power_w    = median(p_w, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(workload, size, device)

readr::write_csv(summary_dws, "data/derived/pilot_summary_by_dws.csv")

print(head(summary_dws, 8))
message("3.5 OK")


# 3.6: Vergleich CPU vs. GPU je {workload, size}
library(dplyr); library(tidyr)

comp <- summary_dws |>
  select(workload, size, device, med_time_s, med_energy_j, med_edp) |>
  tidyr::pivot_wider(
    names_from = device,
    values_from = c(med_time_s, med_energy_j, med_edp),
    names_sep = "_"
  ) |>
  mutate(
    speedup_time      = med_time_s_CPU / med_time_s_GPU,
    ratio_energy      = med_energy_j_CPU / med_energy_j_GPU,
    ratio_edp         = med_edp_CPU     / med_edp_GPU,
    gpu_better_time   = speedup_time  > 1,
    gpu_better_energy = ratio_energy  > 1,
    gpu_better_edp    = ratio_edp     > 1
  ) |>
  arrange(workload, size)

print(comp)
message("3.6 OK")

# 3.7: Break-even Quick-Check aus den Verhältnissen
library(dplyr)

be <- comp |>
  mutate(
    speedup_time  = round(speedup_time, 2),
    ratio_energy  = round(ratio_energy, 2),
    ratio_edp     = round(ratio_edp, 2),
    verdict_time   = ifelse(gpu_better_time,   "GPU schneller",     "CPU schneller"),
    verdict_energy = ifelse(gpu_better_energy, "GPU weniger Energie","CPU weniger Energie"),
    verdict_edp    = ifelse(gpu_better_edp,    "GPU weniger EDP",    "CPU weniger EDP")
  )

print(be)

agg <- comp |>
  summarise(
    min_speedup_time = min(speedup_time, na.rm = TRUE),
    min_ratio_energy = min(ratio_energy, na.rm = TRUE),
    min_ratio_edp    = min(ratio_edp, na.rm = TRUE)
  )

print(agg)

# Kurzfazit in Klartext
if (agg$min_speedup_time > 1) cat("=> Kein Zeit-Break-even in [2000,8000]; GPU dominiert. Kandidat Break-even < 2000.\n")
if (agg$min_ratio_energy > 1) cat("=> Kein Energie-Break-even in [2000,8000]; GPU dominiert. Kandidat Break-even < 2000.\n")
if (agg$min_ratio_edp    > 1) cat("=> Kein EDP-Break-even in [2000,8000]; GPU dominiert. Kandidat Break-even < 2000.\n")

message("3.7 OK")

# 3.8: Kandidaten-Größen <2000 für GEMM planen
library(readr)

sizes <- c(64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
           640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792)

plan <- data.frame(
  workload = "GEMM",
  size = sizes,
  repeats = 10
)

readr::write_csv(plan, "data/derived/plan_next_sizes_gemm.csv")
print(head(plan, 8))
message("3.8 OK")


# 3.9: weitere Workloads (nur Profil, noch keine Größen)
library(readr)

workloads <- data.frame(
  workload  = c("GEMM", "SpMV", "Reduction", "FFT_1D"),
  intensity = c("compute_bound", "memory_bound", "memory_bound", "mixed")
)

readr::write_csv(workloads, "data/derived/plan_workloads.csv")
print(workloads)
message("3.9 OK")


# 3.10: Größenplan für Reduction
library(readr)

sizes <- c(64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512,
           640, 768, 896, 1024, 1280, 1536, 1792, 2048,
           4096, 8192, 16384, 32768, 65536)

plan_reduc <- data.frame(
  workload = "Reduction",
  size = sizes,
  repeats = 10
)

readr::write_csv(plan_reduc, "data/derived/plan_next_sizes_reduction.csv")
print(head(plan_reduc, 8))
message("3.10 OK")

# 3.11: combine plans into one run sheet
library(readr); library(dplyr)

plan_all <- bind_rows(
  read_csv("data/derived/plan_next_sizes_gemm.csv", show_col_types = FALSE),
  read_csv("data/derived/plan_next_sizes_reduction.csv", show_col_types = FALSE)
) |>
  arrange(workload, size) |>
  mutate(order = dplyr::row_number())

readr::write_csv(plan_all, "data/derived/plan_next_runs.csv")
print(head(plan_all, 10))
message("3.11 OK")

# 3.12: Ergebnisse sichern (klein & reproduzierbar)
library(dplyr); library(readr)

dir.create("reports", showWarnings = FALSE)

# CPU↔GPU Gegenüberstellung sichern
readr::write_csv(comp, "data/derived/pilot_comp.csv")

# QC-/Harmonisierungskurzbericht
counts <- df |> count(device, use_energy)
med_pw  <- df |> filter(use_energy) |>
  group_by(device) |>
  summarise(median_power_W = median(p_w, na.rm = TRUE), .groups="drop")

lines <- c(
  "Pilot QC/Harmonisierung",
  paste0("Datum: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
  "",
  "Energie-Policy:",
  "- GPU: nur nvml_total_energy mit plausibler Leistung [10, 450] W",
  "- CPU: rapl/rapl_pkg",
  "",
  "Nutzung (Zeilen):",
  paste(capture.output(print(counts)), collapse = "\n"),
  "",
  "Medianleistung (W, nur verwendete):",
  paste(capture.output(print(med_pw)), collapse = "\n"),
  "",
  "Hinweis: power_integration_fallback_zero_counter wurde für GPU verworfen (verdächtig niedrige W)."
)

writeLines(lines, "reports/pilot_qc_decisions.txt")
message("3.12 OK")

# 3.13: Appendix-Tabelle für Methoden/Ergebnisse
library(dplyr); library(readr)

appendix_tbl <- summary_dws |>
  select(workload, size, device, n,
         med_time_s, med_energy_j, med_edp) |>
  tidyr::pivot_wider(
    names_from = device,
    values_from = c(n, med_time_s, med_energy_j, med_edp),
    names_sep = "_"
  ) |>
  mutate(
    speedup_time = med_time_s_CPU / med_time_s_GPU,
    ratio_energy = med_energy_j_CPU / med_energy_j_GPU,
    ratio_edp    = med_edp_CPU     / med_edp_GPU
  ) |>
  arrange(workload, size)

dir.create("reports", showWarnings = FALSE)
readr::write_csv(appendix_tbl, "reports/appendix_pilot_summary.csv")

print(head(appendix_tbl, 6))
message("3.13 OK")

# 3.14: zwei kleine Plots als PNG speichern
library(dplyr); 
library(ggplot2)

dir.create("figs", showWarnings = FALSE)

gemm <- appendix_tbl |> dplyr::filter(workload == "GEMM")

p_speed <- ggplot(gemm, aes(x = size, y = speedup_time)) +
  geom_line() + geom_point() +
  labs(title = "GEMM: GPU Speedup (Zeit)", x = "Größe", y = "CPU_time / GPU_time")

p_energy <- ggplot(gemm, aes(x = size, y = ratio_energy)) +
  geom_line() + geom_point() +
  labs(title = "GEMM: Energie-Verhältnis", x = "Größe", y = "CPU_energy / GPU_energy")

ggsave("figs/pilot_gemm_speedup.png", p_speed, width = 5, height = 3, dpi = 150)
ggsave("figs/pilot_gemm_energy_ratio.png", p_energy, width = 5, height = 3, dpi = 150)

message("3.14 OK")


# 3.15: QC-Plot für GPU: p_w nach energy_method, markiert ob verwendet
library(dplyr); library(ggplot2)

dir.create("figs", showWarnings = FALSE)

gpu <- df |>
  filter(device == "GPU") |>
  mutate(use = ifelse(use_energy, "used", "discarded"))

p <- ggplot(gpu, aes(x = energy_method, y = p_w)) +
  geom_boxplot(outlier.shape = NA) +
  geom_jitter(aes(shape = use, alpha = use), width = 0.15, height = 0, size = 2) +
  scale_alpha_manual(values = c(discarded = 0.6, used = 1)) +
  coord_cartesian(ylim = c(0, 500)) +
  labs(title = "GPU: Durchschnittsleistung nach Messmethode",
       x = "energy_method", y = "Watt (J/s)")

ggsave("figs/qc_gpu_power_by_method.png", p, width = 6, height = 3.5, dpi = 150)
message("3.15 OK")


# 3.16: Mini-Report Rmd erzeugen (noch nicht rendern)
dir.create("reports", showWarnings = FALSE)

rmd <- c(
  "---",
  'title: "Pilot: CPU↔GPU (GEMM) — Mini-Report"',
  "output: html_document",
  "---",
  "",
  "## QC-Entscheidungen (kurz)",
  " - GPU-Energie: nur `nvml_total_energy` mit Leistung 10–450 W",
  " - CPU-Energie: `rapl`/`rapl_pkg`",
  "",
  "## Ergebnisse GEMM",
  "```{r, echo=FALSE}",
  "library(readr); library(dplyr); library(knitr)",
  'appendix_tbl <- readr::read_csv("appendix_pilot_summary.csv", show_col_types = FALSE)',
  'gemm <- appendix_tbl %>% dplyr::filter(workload=="GEMM") %>%',
  '  dplyr::select(size, speedup_time, ratio_energy, ratio_edp)',
  'knitr::kable(gemm, digits=2, caption="GEMM: Verhältnisse CPU/GPU")',
  "```",
  "",
  "### Abbildungen",
  "![](../figs/pilot_gemm_speedup.png)",
  "",
  "![](../figs/pilot_gemm_energy_ratio.png)",
  "",
  "## QC-Plot (Methodenvergleich)",
  "![](../figs/qc_gpu_power_by_method.png)",
  "",
  "## Reproduzierbarkeit",
  "```{r, echo=FALSE}",
  "print(sessionInfo())",
  "```"
)

writeLines(rmd, "reports/pilot_mini_report.Rmd")
message("3.16 OK")


# 3.17: Artefakt-Manifest + To-do für nächste Runde
dirs <- c("reports","figs","data/derived"); invisible(lapply(dirs, dir.create, showWarnings=FALSE))

files <- c(
  "data/derived/pilot_harmonized.csv",
  "data/derived/pilot_summary_by_dws.csv",
  "data/derived/pilot_comp.csv",
  "data/derived/plan_next_sizes_gemm.csv",
  "data/derived/plan_next_sizes_reduction.csv",
  "data/derived/plan_next_runs.csv",
  "reports/appendix_pilot_summary.csv",
  "reports/pilot_qc_decisions.txt",
  "reports/pilot_mini_report.Rmd",
  "figs/pilot_gemm_speedup.png",
  "figs/pilot_gemm_energy_ratio.png",
  "figs/qc_gpu_power_by_method.png"
)

fi <- file.info(files)
manifest <- sprintf("%-45s  %10s  %s",
                    rownames(fi), format(fi$size, big.mark=","), fi$mtime)
manifest <- c("Pilot-Artfakte (Pfad | Größe | mtime)", manifest)
writeLines(manifest, "reports/pilot_artifacts.txt")

todo <- c(
  "Nächste Messrunde — To-do (kurz)",
  "- OS: Linux; Logger: logger4.py (GPU-Energie NUR nvml_total_energy, kein Fallback).",
  "- Input: data/derived/plan_next_runs.csv (GEMM + Reduction, <2000 inkl.).",
  "- Für jeden Run: device/GPU-Modell, Treiber, Takt/Power-Limits, seed dokumentieren.",
  "- Output-Dateien: data/raw/next_full.csv und data/derived/next_standardized.csv.",
  "- Danach gleich 3.1–3.7 Pipeline wiederholen und Break-even <2000 prüfen."
)
writeLines(todo, "reports/next_run_todo.txt")

message("3.17 OK")
