
import sys
try:
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    try:
        e = nv.nvmlDeviceGetTotalEnergyConsumption(h)
        print("TOTAL_ENERGY_SUPPORTED", e, "mJ_since_driver_reload")
    except nv.NVMLError as err:
        print("TOTAL_ENERGY_NOT_SUPPORTED", type(err).__name__)
    nv.nvmlShutdown()
except Exception as ex:
    print("PYNVML_ERROR", type(ex).__name__, ex)
PY
