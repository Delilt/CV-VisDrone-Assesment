# SANIYEDE KAC KARE ISLENDIGINI OLCEN KOD

import time
import collections

class FPSMeter:
    def __init__(self, buffer_len=100):
        """
        Son N karenin ortalamasını alan FPS sayacı.
        """
        self.frametimes = collections.deque(maxlen=buffer_len)
        self.last_time = time.time()

    def tick(self):
        """Her karede çağırılır"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.frametimes.append(dt)

    def get_fps(self):
        """Ortalama FPS döndürür"""
        if not self.frametimes:
            return 0.0
        avg_dt = sum(self.frametimes) / len(self.frametimes)
        return 1.0 / (avg_dt + 1e-6)