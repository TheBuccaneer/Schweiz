#!/bin/bash
# RTX 5050 Setup - identisch zur RTX 3090 Methodik

echo "=== RTX 5050 Setup (3090-kompatibel) ==="

# GPU konfigurieren
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 130        # Max für RTX 5050
sudo nvidia-smi -lgc 2572      # Boost Clock
sudo nvidia-smi -lmc 10000     # Memory Clock

echo "RTX 5050: 130W, 2572MHz, 10000MHz"

# 60s Warmup
echo "Warmup 60s..."
python3 << 'EOF'
import cupy as cp
import time

# Gleiche Matrix-Größe wie RTX 3090
a = cp.ones((6144, 6144), dtype=cp.float32)
b = cp.ones((6144, 6144), dtype=cp.float32)

start = time.time()
while time.time() - start < 60:
    c = cp.dot(a, b)
    cp.cuda.Stream.null.synchronize()
    time.sleep(0.1)

print("Fertig!")
EOF

nvidia-smi
