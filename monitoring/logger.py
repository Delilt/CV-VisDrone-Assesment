# JSON LOGU

import logging
import json
import os
from datetime import datetime

# Log Klasörü
LOG_DIR = "logs_system"
os.makedirs(LOG_DIR, exist_ok=True)

# Logger Ayarları
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"system_{datetime.now().strftime('%Y%m%d')}.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SystemLogger:
    @staticmethod
    def log_inference(inference_time, obj_count):
        """Her tahminin sonucunu JSON formatında kaydeder"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "metric": "inference",
            "latency_ms": round(inference_time, 2),
            "objects_detected": obj_count
        }
        logging.info(json.dumps(data))
        
    @staticmethod
    def log_error(error_msg):
        data = {
            "timestamp": datetime.now().isoformat(),
            "metric": "error",
            "message": str(error_msg)
        }
        logging.error(json.dumps(data))