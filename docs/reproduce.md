## Vorher-/Nachher-Skripte

Für jede Messung werden spezielle Skripte **jeweils vor und nach** der Durchführung ausgeführt, um Systemzustände, Metadaten und Messumgebungsvariablen zuverlässig zu dokumentieren und zu sichern.

### Ablauf

1. **Vor der Messung**  
   Direkt vor Beginn einer Messung werden die folgenden Skripte ausgeführt:  
   - `script_pre_1`  
   - `script_pre_2`  
   - … (weitere Skripte, z. B. `pre_env_check.sh` etc.)  

2. **Nach der Messung**  
   Unmittelbar nach Abschluss einer Messung werden diese Skripte ausgeführt:  
   - `script_post_1`  
   - `script_post_2`  
   - … (weitere Skripte, z. B. `post_env_check.sh` etc.)

### Zweck

- Sicherstellen, dass die Ausgangsbedingungen jeder Messung dokumentiert sind  
- Erfassen aller relevanten Systemparameter (Hardware, Betriebssystem, Bibliotheksversionen, Umgebungsvariablen) vor und nach der Messung  
- Aufzeichnung aller Logs und Metadaten, sodass spätere Vergleiche möglich sind  

### Log-Aufbewahrung und Versionierung

- Alle Skript-Ausgaben werden versioniert und abgelegt in `logs/pre_post/`  
- Jede Datei bekommt einen Zeitstempel und Angabe zu Datum, Uhrzeit und Benutzer  
- Notiert wird auch die Version der verwendeten Skripte und der relevanten Softwarekomponenten  

### Beispiel

