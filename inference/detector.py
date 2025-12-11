# ONNX/TENSOSRRT ILE TEK KARE INFERENCE

import time
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, device=0):
        """
        Multi-Backend Detector Class
        Desteklenen Formatlar: .pt (PyTorch), .onnx (ONNX Runtime), .engine (TensorRT)
        """
        # self.device = device if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'  # GPU hatası aldığımız için CPU'ya zorluyoruz.
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        print(f"[Detector] Yükleniyor: {model_path} | Cihaz: {self.device}")
        
        # Ultralytics Wrapper
        # Bu yapı .pt, .onnx ve .engine dosyalarını otomatik tanır.
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[Detector] HATA: Model yüklenemedi! {e}")
            raise e

        # Isınma Turu (Warm-up)
        # Modelin hafızaya yerleşmesi için boş bir veriyle çalıştırıyoruz.
        self._warmup()

    def _warmup(self):
        print("[Detector] Isınma turu (Warm-up)...")
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_input, verbose=False, device=self.device)
        print("[Detector] Hazır!")

    def detect(self, frame):
        """
        Tek bir kare üzerinde tespit yapar.
        Döndürür: detections listesi [[x1, y1, x2, y2, score, class_id], ...]
        """
        t0 = time.time()
        
        # Inference
        results = self.model.predict(
            source=frame, 
            conf=self.conf_thres, 
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
            imgsz=640
        )
        
        t1 = time.time()
        inference_time = (t1 - t0) * 1000 # ms cinsinden

        detections = []
        
        # Sonuçları ayrıştır
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                r = box.xyxy[0].astype(int) # x1, y1, x2, y2
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                
                # Format: [x1, y1, x2, y2, score, class_id]
                detections.append([r[0], r[1], r[2], r[3], conf, cls_id])

        return detections, inference_time

if __name__ == "__main__":
    # BASİT TEST
    # Bu dosya doğrudan çalıştırılırsa test yapar.
    
    # 1. Modeli Seç (PyTorch veya ONNX ile test et)
    model_path = "../models/latest.onnx" # ONNX modelini test edelim
    
    detector = Detector(model_path=model_path)
    
    # 2. Rastgele bir resim oluştur (Simülasyon)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 3. Tespit Yap
    dets, dt = detector.detect(img)
    
    print(f"Test Sonucu: {len(dets)} nesne bulundu.")
    print(f"Süre: {dt:.2f} ms")