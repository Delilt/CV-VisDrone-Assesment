# ONNX ---> TENSORRT EXPORTER

from ultralytics import YOLO
import os

MODEL_PATH = "../models/latest.pt" 

def build_engine():
    if not os.path.exists(MODEL_PATH):
        print("Eğitilmiş model bulunamadı, standart yolov8n kullanılıyor.")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(MODEL_PATH)

    # 1. FP16 (Half Precision) Engine Üretimi

    print("\n--- FP16 TensorRT Engine Oluşturuluyor ---")
    try:
        model.export(
            format="engine",  # TensorRT formatı
            device=0,
            half=True,        # FP16 modu (Hızlandırır)
            dynamic=True,
            workspace=4,
            simplify=True
        )
        print("FP16 Engine tamamlandı.")
    except Exception as e:
        print(f"FP16 Export Hatası (TensorRT kurulu mu?): {e}")

    # 2. INT8 Engine Üretimi
   
    print("\n--- INT8 TensorRT Engine Oluşturuluyor ---")
    try:
        model.export(
            format="engine",
            device=0,
            int8=True,        # INT8 modu (En hızlısı ama kalibrasyon gerekir)
            data="../datasets/hard_hat_workers/data.yaml", # 
            dynamic=True,
            simplify=True
        )
        print("INT8 Engine tamamlandı.")
    except Exception as e:
        print(f"INT8 Export Hatası: {e}")

if __name__ == "__main__":
    # Uyarı: Bu kodun çalışması için sistemde TensorRT kütüphanelerinin kurulu olması gerekir.
    # Docker ortamında çalıştırılması önerilir.
    build_engine()