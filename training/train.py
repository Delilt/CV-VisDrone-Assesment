import os
# 1. OMP HATASI ÇÖZÜMÜ: Bu satır en üstte olmalı!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# --- AYARLAR ---
DATA_CONF = "VisDrone.yaml" 
MODEL_NAME = "yolov8n.pt"   
EPOCHS = 10                 
IMG_SZ = 640                
BATCH_SIZE = 4              
PROJECT_NAME = "logs"

def train_model():
    print(f"Model yükleniyor: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print("VisDrone eğitimi başlıyor...")
    print("NOT: Windows hatasını önlemek için 'workers=0' yapıldı.")

    results = model.train(
        data=DATA_CONF,
        epochs=EPOCHS,
        imgsz=IMG_SZ,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name="visdrone_run",
        
        # --- Donanım Optimizasyonu ---
        device=0,           
        workers=0,          # <--- KRİTİK DEĞİŞİKLİK: Windows'ta hata almamak için 0 yaptık.
        cache=False,        
        amp=True,           
        
        # --- Optimizasyon ---
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=3.0,
        
        # --- Augmentation ---
        mosaic=1.0,               
        mixup=0.1,                
        degrees=5.0,
        
        exist_ok=True,
        plots=True
    )

    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    train_model()