# PYTHORCH ---> ONNX EXPORTER


from ultralytics import YOLO
import os

# Yollar
MODEL_PATH = "../models/latest.pt"  # Eğitilen model (veya yolov8n.pt)
ONNX_PATH = "../models/model.onnx"

def export_onnx():
    # Model var mı kontrol et (Yoksa indirir veya uyarı verir)
    if not os.path.exists(MODEL_PATH):
        print(f"Uyarı: {MODEL_PATH} bulunamadı. 'yolov8n.pt' indiriliyor...")
        model = YOLO("yolov8n.pt")
    else:
        print(f"Model yükleniyor: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

    # ONNX'e Çevir
    # Assessment isteği: "Dynamic axes (batch size, height/width), opset >= 12"
    print("ONNX export işlemi başlıyor...")
    success = model.export(
        format="onnx",
        opset=12,           # İstenen Opset
        dynamic=True,       # Dynamic axes (Batch size ve boyut esnekliği için)
        simplify=True       # Modeli sadeleştir (Gereksiz katmanları atar)
    )
    
    if success:
        print(f"Başarılı! Model kaydedildi: {success}")
        # Çıktıyı bizim models klasörüne taşıyalım (Ultralytics olduğu yere kaydedebilir)
        # Not: Ultralytics genelde aynı klasöre kaydeder, manuel taşımaya gerek kalmayabilir.
    else:
        print("Export başarısız oldu.")

if __name__ == "__main__":
    export_onnx()