# HIZ OLCUMLERI

import time
import json
import numpy as np
from ultralytics import YOLO

def run_benchmark():
    print("--- Benchmarking Başlıyor ---")
    model = YOLO("../models/latest.pt") # Varsa .onnx kullan
    
    # Sahte veri (640x640)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    latencies = []
    
    print("Isınma turları (Warmup)...")
    for _ in range(15):
        model.predict(img, verbose=False)
        
    print("Ölçüm yapılıyor...")
    start_time = time.time()
    for _ in range(100):
        t0 = time.time()
        model.predict(img, verbose=False)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
    
    total_time = time.time() - start_time
    
    # 3. İstatistikler (p50, p95)
    latencies = np.array(latencies)
    results = {
        "model": "YOLOv8-VisDrone",
        "avg_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p90_latency_ms": float(np.percentile(latencies, 90)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "throughput_fps": 100 / total_time
    }
    
    print(json.dumps(results, indent=2))
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Sonuçlar 'benchmark_results.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_benchmark()