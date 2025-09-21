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

# Warmup aus venv-Python
VENV_PYTHON="$(pwd)/.venv/bin/python3"
"$VENV_PYTHON" << 'EOF'
import cupy as cp
import time

# Erstelle große Matrizen einmal
a = cp.ones((6144, 6144), dtype=cp.float32)
b = cp.ones((6144, 6144), dtype=cp.float32)

start = time.time()
while time.time() - start < 60:
    c = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    time.sleep(0.1)
print("Warmup complete!")
EOF

echo ""
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.current.graphics --format=table
echo ""
echo "GPUs ready for measurements!"
