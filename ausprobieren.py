#!/usr/bin/env python3

import pynvml

try:
    print("=== NVML GPU Energy Debug (Fixed) ===")

    # NVML initialisieren
    pynvml.nvmlInit()
    print("✅ NVML Init erfolgreich")

    # GPU Handle holen
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    print("✅ GPU Handle erhalten")

    # GPU Name (ohne .decode())
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    print(f"GPU: {gpu_name}")

    # Treiber Version (ohne .decode())
    driver_version = pynvml.nvmlSystemGetDriverVersion()
    print(f"Treiber: {driver_version}")

    # TotalEnergy testen
    try:
        energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        print(f"✅ GPU Energy Counter: {energy} mJ")
    except Exception as e:
        print(f"❌ TotalEnergy Error: {e}")

        # Fallback: Power Usage testen
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            print(f"✅ GPU Power Usage: {power} mW (Fallback möglich)")
        except Exception as e2:
            print(f"❌ Power Usage Error: {e2}")

    pynvml.nvmlShutdown()

except Exception as main_error:
    print(f"❌ Hauptfehler: {main_error}")
