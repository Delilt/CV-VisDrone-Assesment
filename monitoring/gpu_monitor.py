# GPU YUKU VE BELLEK YUKU

import pynvml
import time

class GPUMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception:
            self.available = False
            print("[GPUMonitor] NVIDIA GPU veya sürücü bulunamadı. Mock veri dönülecek.")

    def get_stats(self):
        if not self.available:
            return {"gpu_util": 0, "memory_used": 0, "memory_total": 0}
        
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        
        return {
            "gpu_util": util.gpu,
            "memory_used": mem.used / 1024**2, # MB
            "memory_total": mem.total / 1024**2 # MB
        }