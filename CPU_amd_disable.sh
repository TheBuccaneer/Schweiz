#!/bin/bash

# AMD CPU Reset Script - Setzt Ryzen Threadripper 3970X auf Standardeinstellungen zurück

echo "=== AMD CPU Reset to Default Settings ==="

# Frequenz-Limits aus cpuinfo lesen
MIN_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq 2>/dev/null)
MAX_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)

echo "AMD Ryzen Threadripper 3970X reset..."
echo "Frequency range: $(($MIN_FREQ/1000)) - $(($MAX_FREQ/1000)) MHz"

# Governor zurück auf ondemand oder schedutil (AMD bevorzugt)
AVAILABLE_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
if [[ "$AVAILABLE_GOV" == *"schedutil"* ]]; then
    TARGET_GOV="schedutil"  # AMD-optimierter Governor
elif [[ "$AVAILABLE_GOV" == *"ondemand"* ]]; then
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

# AMD Boost wieder aktivieren
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
    echo "AMD Boost re-enabled"
elif [ -f /sys/devices/system/cpu/amd_pstate/boost ]; then
    echo 1 | sudo tee /sys/devices/system/cpu/amd_pstate/boost >/dev/null
    echo "AMD Boost re-enabled (amd_pstate)"
fi

echo ""
echo "AMD CPU reset to default settings!"
echo "Current governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
echo "Frequency scaling: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq) - $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)"

# Zeige Boost-Status
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo "Boost status: $(cat /sys/devices/system/cpu/cpufreq/boost)"
elif [ -f /sys/devices/system/cpu/amd_pstate/boost ]; then
    echo "Boost status: $(cat /sys/devices/system/cpu/amd_pstate/boost)"
fi

echo ""
echo "AMD Ryzen Threadripper 3970X ready for normal operation!"
