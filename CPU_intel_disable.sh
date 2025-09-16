#!/bin/bash

# CPU Reset Script - Setzt CPU auf Standardeinstellungen zurück

echo "=== CPU Reset to Default Settings ==="

# Frequenz-Limits aus cpuinfo lesen
MIN_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq 2>/dev/null)
MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)

# Governor zurück auf ondemand oder powersave
AVAILABLE_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
if [[ "$AVAILABLE_GOV" == *"ondemand"* ]]; then
    TARGET_GOV="ondemand"
elif [[ "$AVAILABLE_GOV" == *"powersave"* ]]; then
    TARGET_GOV="powersave"
else
    TARGET_GOV=$(echo $AVAILABLE_GOV | awk '{print $1}')
fi

echo "Resetting CPU governor to $TARGET_GOV..."
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -d "$cpu/cpufreq" ]; then
        echo $TARGET_GOV | sudo tee $cpu/cpufreq/scaling_governor >/dev/null 2>&1
    fi
done

# Frequenz-Limits zurücksetzen
if [ -n "$MIN_FREQ" ] && [ -n "$MAX_FREQ" ]; then
    echo "Resetting CPU frequency limits..."
    echo "Min: $(($MIN_FREQ/1000)) MHz, Max: $(($MAX_FREQ/1000)) MHz"
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -d "$cpu/cpufreq" ]; then
            echo $MIN_FREQ | sudo tee $cpu/cpufreq/scaling_min_freq >/dev/null 2>&1
            echo $MAX_FREQ | sudo tee $cpu/cpufreq/scaling_max_freq >/dev/null 2>&1
        fi
    done
fi

# Turbo Boost wieder aktivieren
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
    echo "Turbo Boost re-enabled"
fi

echo ""
echo "CPU reset to default settings!"
echo "Current governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "Frequency scaling: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) - $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"
