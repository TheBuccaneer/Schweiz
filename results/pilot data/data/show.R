setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# Pakete
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(janitor)
})

# === 1) Datei finden & einlesen (robust: erst , dann ;) ===
paths <- c("data/raw/pilot_full_production.csv",
           "data/raw/pilot_full_improved.csv",
           "pilot_full_production.csv",
           "pilot_full_improved.csv")
path <- paths[file.exists(paths)][1]
if (is.na(path)) stop("Keine der erwarteten CSV-Dateien gefunden.")

pilot_raw <- tryCatch(
  readr::read_csv(path, show_col_types = FALSE),
  error = function(e) readr::read_csv2(path, show_col_types = FALSE) # europäisch ; und ,
)

# === 2) Spaltennamen säubern (snake_case, ASCII etc.) ===
pilot0 <- janitor::clean_names(pilot_raw)  # macht z.B. "GPU Kernel (s)" -> "gpu_kernel_s"

# === 3) Hilfsfunktion: erste passende Spalte anhand von Musterlisten finden ===
pick_col <- function(nms, patterns, prefer = character()) {
  # 1) exakte Präferenz
  for (p in prefer) if (p %in% nms) return(p)
  # 2) regex-Suche
  hits <- vapply(patterns, function(pt) any(stringr::str_detect(nms, regex(pt, ignore_case = TRUE))), TRUE)
  if (!any(hits)) return(NA_character_)
  # wähle das erste Muster mit einem Treffer und nimm den ersten passenden Namen
  first_pt <- patterns[which(hits)[1]]
  return(nms[which(stringr::str_detect(nms, regex(first_pt, ignore_case = TRUE)))[1]])
}

nms <- names(pilot0)

# Kandidaten/Regex (an dein Logger angelehnt)
time_col   <- pick_col(nms,
                       patterns = c("^time(_s)?$", "elapsed", "runtime", "duration", "gpu.*kernel.*(s|sec)"),
                       prefer   = c("time_s"))
energy_col <- pick_col(nms,
                       patterns = c("^energy(_j(oules)?)?$", "joule", "nvml.*energy", "power.*time", "energy_used"),
                       prefer   = c("energy_j"))
device_col <- pick_col(nms,
                       patterns = c("^device$", "accelerator", "^cpu$|^gpu$|processor|hardware|platform"),
                       prefer   = c("device"))
size_col   <- pick_col(nms,
                       patterns = c("^size$", "problem.*size", "matrix.*(n|size)", "^n$", "dim"),
                       prefer   = c("size"))
workload_col <- pick_col(nms,
                         patterns = c("^workload$", "benchmark", "kernel", "task", "application|app"),
                         prefer   = c("workload"))
notes_col  <- pick_col(nms,
                       patterns = c("^notes?$", "comment|remark|flag"),
                       prefer   = c("notes"))
timing_col <- pick_col(nms,
                       patterns = c("timing.*method|method.*time", "^timing_method$"),
                       prefer   = c("timing_method"))

# Fallbacks (z.B. wenn time_s fehlt, aber gpu_kernel_time_s existiert)
if (is.na(time_col) && "gpu_kernel_time_s" %in% nms) time_col <- "gpu_kernel_time_s"

# === 4) Mapping anzeigen (zur Kontrolle) ===
mapping <- tibble::tibble(
  target = c("time_s","energy_j","device","size","workload","notes","timing_method"),
  source = c(time_col, energy_col, device_col, size_col, workload_col, notes_col, timing_col)
)
print(mapping)

# Mindestens device/size/time müssen existieren:
need <- c(time_col, device_col, size_col)
if (any(is.na(need))) {
  stop("Automapping unvollständig. Bitte prüfe die obige 'mapping'-Tabelle und passe patterns/prefer an.")
}

# === 5) Standardisieren: einheitliche Spalten erzeugen ===
pilot <- pilot0 %>%
  mutate(
    time_s = suppressWarnings(as.numeric(.data[[time_col]])),
    energy_j = if (!is.na(energy_col)) suppressWarnings(as.numeric(.data[[energy_col]])) else NA_real_,
    device = factor(.data[[device_col]]),
    size   = factor(.data[[size_col]], levels = sort(unique(.data[[size_col]]))),
    workload = if (!is.na(workload_col)) as.character(.data[[workload_col]]) else NA_character_,
    notes  = if (!is.na(notes_col)) as.character(.data[[notes_col]]) else NA_character_,
    timing_method = if (!is.na(timing_col)) as.character(.data[[timing_col]]) else NA_character_
  )

# Kurzer Check:
stopifnot(is.finite(sum(pilot$time_s, na.rm = TRUE)))
message("Standardisierung fertig. Zeilen: ", nrow(pilot))

# (Optional) Wegschreiben für Folgeschritte
outdir <- "data/derived"; if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
outpath <- file.path(outdir, "pilot_standardized.csv")
readr::write_csv(pilot, outpath)
message("Gespeichert als: ", outpath)
