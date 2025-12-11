# DETECTOR DUZGUN CALISIYOR MU TESTLERI

import sys
import os
import numpy as np
import pytest

# Ana dizini path'e ekle ki modülleri bulabilsin
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector

# Testte kullanılacak model
MODEL_PATH = "models/latest.onnx" 

def test_model_loading():
    """Modelin başarıyla yüklenip yüklenmediğini test et"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model dosyası bulunamadı: {MODEL_PATH}")
    
    detector = Detector(model_path=MODEL_PATH)
    assert detector.model is not None

def test_prediction_shape():
    """Modelin çıktı formatının doğru olduğunu test et"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model yok")

    detector = Detector(model_path=MODEL_PATH)
    
    # Sahte bir resim oluştur (640x640, siyah)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    detections, latency = detector.detect(dummy_frame)
    
    # Çıktı bir liste olmalı (boş olabilir ama liste olmalı)
    assert isinstance(detections, list)
    assert isinstance(latency, float)