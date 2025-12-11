# API KODU

import sys
import os
import cv2
import numpy as np
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from monitoring.logger import SystemLogger
from monitoring.fps_meter import FPSMeter
from monitoring.gpu_monitor import GPUMonitor

fps_meter = FPSMeter()
gpu_monitor = GPUMonitor()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector

app = FastAPI(
    title="Dataguess AI Assessment API",
    description="YOLOv8 Object Detection & Tracking API",
    version="1.0.0"
)

detector = None

@app.on_event("startup")
def load_model():
    """Sunucu başlarken modeli belleğe yükle"""
    global detector
    model_path = "models/latest.pt"
    
    if os.path.exists("models/latest.onnx"):
        model_path = "models/latest.onnx"
        print("ONNX model bulundu, aktif ediliyor.")

    try:
        detector = Detector(model_path=model_path, conf_thres=0.25)
        print(f"Model başarıyla yüklendi: {model_path}")
    except Exception as e:
        print(f" HATA: Model yüklenemedi! {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Web Arayüzünü Yükle"""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()
# --------------------------------------------

@app.get("/health")
def health_check():
    """Sistem ayakta mı kontrolü"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")
    return {"status": "healthy", "service": "Dataguess AI API"}

@app.get("/metrics")
def get_metrics():
    """Sistem performans metriklerini döndürür"""
    gpu_stats = gpu_monitor.get_stats()
    return {
        "fps": fps_meter.get_fps(),
        "gpu_usage_percent": gpu_stats["gpu_util"],
        "gpu_memory_mb": gpu_stats["memory_used"],
        "system_status": "active"
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Resim yükle -> Sonuçları (JSON) al"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model hazır değil")

    # 1. Resmi Oku
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        raise HTTPException(status_code=400, detail="Geçersiz resim dosyası")

    # 2. Tahmin Yap
    # detector.detect() bize (detections, inference_time) dönüyor
    detections, latency = detector.detect(frame)

    # 3. Sonucu Formatla
    results = []
    for det in detections:
        # det: [x1, y1, x2, y2, score, class_id]
        results.append({
            "bbox": [int(det[0]), int(det[1]), int(det[2]), int(det[3])],
            "confidence": float(det[4]),
            "class_id": int(det[5])
        })
    # Tahminleri Loglama
    SystemLogger.log_inference(inference_time=latency, obj_count=len(results))
    
    return {
        "inference_time_ms": latency,
        "object_count": len(results),
        "predictions": results
    }