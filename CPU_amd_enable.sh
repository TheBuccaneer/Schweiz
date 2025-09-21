#!/bin/bash

# AMD CPU Stabilisierungs-Script für Ryzen Threadripper 3970X
# Fixiert CPU-Frequenz und wärmt auf für konsistente Energiemessungen

echo "=== AMD CPU Stabilization ==="

# Dynamische Frequenz-Erkennung für AMD
BASE_FREQ=$(lscpu | grep "Model name" | grep -oP '@\s*\K[0-9.]+' | awk '{print int($1*1000000)}')
if [ -z "$BASE_FREQ" ] || [ "$BASE_FREQ" -eq 0 ]; then
    # Fallback: Nutze base_frequency oder min_freq
    BASE_FREQ=$(cat /sys/devices/system/cpu/cpu0/cpufreq/base_frequency 2>/dev/null || \
                cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq)
fi

echo "Detected base frequency: $(($BASE_FREQ/1000)) MHz"
echo "AMD Ryzen Threadripper 3970X (32C/64T) detected"

# Verfügbare Governors prüfen
AVAILABLE_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)
if [[ "$AVAILABLE_GOV" == *"performance"* ]]; then
    echo "Setting CPU governor to performance mode..."
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        echo performance | sudo tee $cpu/cpufreq/scaling_governor >/dev/null 2>&1
    done
else
    echo "Performance governor not available, using current: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
fi

# Fixiere CPU-Frequenz auf erkannte Basis-Frequenz
echo "Locking CPU frequency to $(($BASE_FREQ/1000)) MHz..."
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    if [ -d "$cpu/cpufreq" ]; then
        echo $BASE_FREQ | sudo tee $cpu/cpufreq/scaling_min_freq >/dev/null 2>&1
        echo $BASE_FREQ | sudo tee $cpu/cpufreq/scaling_max_freq >/dev/null 2>&1
    fi
done

# AMD Boost deaktivieren für konsistente Frequenz
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
    echo "AMD Boost disabled"
elif [ -f /sys/devices/system/cpu/amd_pstate/boost ]; then
    echo 0 | sudo tee /sys/devices/system/cpu/amd_pstate/boost >/dev/null
    echo "AMD Boost disabled (amd_pstate)"
fi

# Status zeigen
echo ""
echo "Current AMD CPU settings:"
echo -n "Governor: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo -n "Current freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
echo -n "Min freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
echo -n "Max freq: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq

# Boost-Status prüfen
if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo -n "Boost status: "
    cat /sys/devices/system/cpu/cpufreq/boost
fi

echo ""
echo "Warming up AMD Threadripper (64 threads) for 60 seconds..."

# CPU Warmup - 64 Threads für Threadripper 3970X
NUM_CPUS=64
echo "Starting intensive warmup on all $NUM_CPUS threads..."

stress-ng --cpu $NUM_CPUS --timeout 60s --metrics-brief 2>/dev/null || {
    # Fallback wenn stress-ng nicht installiert
    echo "Using Python fallback for warmup..."
    python3 -c "
import time
import numpy as np
from multiprocessing import Pool

def cpu_work(x):
    # Matrix-Operationen für intensive CPU-Last
    size = 1000
    a = np.random.rand(size, size)
    end = time.time() + 60
    while time.time() < end:
        b = np.dot(a, a)
    return x

# Nutze alle 64 Threads des Threadrippers
with Pool($NUM_CPUS) as p:
    p.map(cpu_work, range($NUM_CPUS))
print('AMD CPU warmup complete!')
" 2>/dev/null || echo "Install stress-ng for optimal warmup performance"
}

echo ""
echo "AMD Ryzen Threadripper 3970X stabilized and ready for measurements!"
echo ""
echo "Current CPU frequency per core (first 8 cores):"
grep MHz /proc/cpuinfo | head -n 8
