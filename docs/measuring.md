Messung & Logging (RAPL / NVML) – Regeln

Für jede Messung wird der Energie-Zähler vor dem Start des Jobs und nach dem Ende des Jobs gelesen; der Energie-verbrauch ergibt sich aus der Differenz.

RAPL / powercap (CPU):
  ・ Verwendet energy_uj und max_energy_range_uj.
  ・ Overflow / Wrap-around prüfen (wenn Zähler überläuft) und korrekt behandeln.
  ・ Keine Nutzung von Instant-Power, da nicht zuverlässig bzw. nicht in allen Domains verfügbar.

NVML (GPU):
  ・ Verwendet den TotalEnergyCounter seit dem Treiberstart (z. B. in mJ); Differenzen über Runs.
  ・ Samplingfenster dokumentieren (z. . z. B. wie oft Abfrage, wie groß das Intervall).

Synchronisation der Start- und Stoppzeitpunkte: Zeitstempel (z. B. UNIX Timestamp mit hoher Auflösung) möglichst nahe an den Calls zu RAPL/NVML messen, minimaler Overhead.

Hintergrundlast kontrollieren: während der Messung sollten möglichst keine anderen signifikanten Jobs laufen, die CPU oder GPU auslasten, um Störeinflüsse zu minimieren.

Wiederholungen: pro Konfiguration mindestens N Runs (z. B. 20) zur Abschätzung Varianz.

Metadaten loggen: Hardware (CPU Modell, Anzahl Threads, Frequenzen), Driver Versionen, BLAS/cuBLAS Libraries, NVML Version, Stromversorgungssituation, evtl. Umgebungstemperatur falls möglich.

Validierung mit externen Quellen (wenn möglich, Stichprobe mit Steck­dosenmessung oder externem Energiekalkulator), um die Glaubwürdigkeit der internen Messungen zu prüfen.

Dokumentation von Ausreißern / Anomalien: Runs mit ungewöhnlich hohen Laufzeiten, auffälligem Energieverbrauch etc. protokollieren und ggf. ausschließen oder gesondert berichten.

Alle Messungen werden in rohform gespeichert (Zeit, Energie, Device, Größe, Workload etc.) + QC-Skripte zur Überprüfung.