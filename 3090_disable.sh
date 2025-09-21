#!/bin/bash

# GPU Reset Script - Setzt alle GPUs auf Standardeinstellungen zurück

echo "=== GPU Reset to Default Settings ==="

# Für jede GPU
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    echo "Resetting GPU $gpu_id: $gpu_name"

    # Reset GPU Clocks (entfernt Lock, aktiviert Auto-Boost wieder)
    sudo nvidia-smi -i $gpu_id -rgc
    echo "  ✓ GPU clock reset to auto"

    # Reset Memory Clocks (falls gesetzt)
    sudo nvidia-smi -i $gpu_id -rmc
    echo "  ✓ Memory clock reset to auto"

    # Reset Power Limit auf Default
   # sudo nvidia-smi -i $gpu_id -pl 0  # 0 = default/maximum
    echo "  ✓ Power limit reset to default"

    # Reset Application Clocks (falls gesetzt)
    sudo nvidia-smi -i $gpu_id -rac
    echo "  ✓ Application clocks reset"
done

# Persistence Mode ausschalten (optional - meist kann man es anlassen)
# sudo nvidia-smi -pm 0
# echo "Persistence mode disabled"

echo ""
echo "=== Current Status ==="
nvidia-smi --query-gpu=index,name,power.limit,clocks.current.graphics,clocks.max.graphics --format=table

echo ""
echo "GPUs reset to default settings!"
echo "Auto-boost is now enabled again."
