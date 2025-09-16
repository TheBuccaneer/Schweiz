#!/bin/bash

# Minimales GPU Stabilisierungs-Script für Energiemessungen

echo "=== GPU Stabilization (Minimal) ==="

# Persistence Mode (Treiber bleibt geladen)
sudo nvidia-smi -pm 1

# Für jede GPU
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    echo "Configuring GPU $gpu_id: $gpu_name"

    # Power Limit und Clock fixieren
    if [[ "$gpu_name" == *"3090"* ]]; then
        sudo nvidia-smi -i $gpu_id -pl 350    # Power limit
        sudo nvidia-smi -i $gpu_id -lgc 1695   # Lock GPU clock
        sudo nvidia-smi -i $gpu_id -lmc 9750   # Lock memory clock
    elif [[ "$gpu_name" == *"1080"* ]]; then
        sudo nvidia-smi -i $gpu_id -pl 250    # Power limit
        sudo nvidia-smi -i $gpu_id -lgc 1480   # Lock GPU clock
        sudo nvidia-smi -i $gpu_id -lmc 5505   # Lock memory clock
    fi
done

echo ""
echo "Warming up GPU for 60 seconds..."

# Einfacher Warmup mit Python (falls cupy installiert)
python3 -c "
try:
    import cupy as cp
    import time

    # Die eine GPU aufwärmen
    a = cp.random.random((6144, 6144), dtype=cp.float32)
    b = cp.random.random((6144, 6144), dtype=cp.float32)

    start = time.time()
    while time.time() - start < 60:  # 60 sec für die eine GPU
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
    print('Warmup complete!')
except:
    print('Skipping warmup (cupy not installed)')
" 2>/dev/null || echo "Skipping warmup"

# Status zeigen
echo ""
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.current.graphics --format=table

echo ""
echo "GPUs ready for measurements!"
