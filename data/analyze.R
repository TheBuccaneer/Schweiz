# analysis/analyse.R  (neu anlegen)

# Pakete
suppressPackageStartupMessages({
  library(readr)   # CSV-Import
  library(dplyr)   # Summaries
})

# 1) Standardisierte Daten einlesen (das ist die Datei, die dein Standardisierungs-Skript erzeugt hat)
path <- "data/derived/pilot_standardized.csv"
stopifnot(file.exists(path))
pilot <- readr::read_csv(path, show_col_types = FALSE)

# 2) Deine drei QC-Blöcke

# Vollständigkeit & 0-J-Anteil
c(n = nrow(pilot),
  na_time   = sum(!is.finite(pilot$time_s)),
  na_energy = sum(!is.finite(pilot$energy_j)),
  zero_energy = sum(pilot$energy_j == 0, na.rm = TRUE))

# Median-Zeiten & Energie-Verfügbarkeit je device×size
pilot %>%
  group_by(device, size) %>%
  summarise(n = n(),
            med_time = median(time_s, na.rm = TRUE),
            energy_avail = mean(is.finite(energy_j)),
            zero_energy  = mean(energy_j == 0, na.rm = TRUE),
            .groups = "drop")

# Grobe Ausreißer-/Skalenprüfung
quantile(pilot$time_s, c(.01,.5,.99), na.rm = TRUE)


# === Abschnitt: per-Run-Kennzahlen & Zusammenfassung ===
library(dplyr)
library(ggplot2)

# Batchgröße verwenden (wenn vorhanden) oder aus Größe ableiten
pilot <- pilot %>%
  mutate(
    batch_size = if ("batch_size" %in% names(pilot)) batch_size else dplyr::case_when(
      as.numeric(as.character(size)) <= 2000 ~ 50L,
      as.numeric(as.character(size)) <= 4000 ~ 20L,
      TRUE ~ 5L
    ),
    time_per_run   = time_s   / batch_size,
    energy_per_run = energy_j / batch_size,
    edp            = time_per_run * energy_per_run
  )

# Übersicht je device×size (Median + IQR)
summary_runs <- pilot %>%
  group_by(device, size) %>%
  summarise(
    n = n(),
    med_time_per_run   = median(time_per_run, na.rm = TRUE),
    iqr_time_per_run   = IQR(time_per_run, na.rm = TRUE),
    med_energy_per_run = median(energy_per_run, na.rm = TRUE),
    iqr_energy_per_run = IQR(energy_per_run, na.rm = TRUE),
    med_edp            = median(edp, na.rm = TRUE),
    iqr_edp            = IQR(edp, na.rm = TRUE),
    .groups = "drop"
  )
print(summary_runs)

# Optional: prüfe ~n^3-Skalierung (log-log-Steigung ~3)
sizes_num <- as.numeric(as.character(pilot$size))
fit_loglog <- lm(log(time_per_run) ~ log(sizes_num), data = pilot)
print(coef(summary(fit_loglog)))  # Steigung sollte ~3 sein

# Optional: Plot per-Run-Zeit
ggplot(pilot, aes(x = size, y = time_per_run, color = device)) +
  geom_boxplot(outlier.alpha = 0.5) +
  labs(x = "Größe (n)", y = "Zeit pro Run [s]", title = "per-Run Laufzeit nach Größe und Device") +
  theme_minimal()
