# PYTHORCH ---> ONNX EXPORTER


from ultralytics import YOLO
import os

# Yollar
MODEL_PATH = "../models/latest.pt"
ONNX_PATH = "../models/model.onnx"

def export_onnx():
    if not os.path.exists(MODEL_PATH):
        print(f"Uyarı: {MODEL_PATH} bulunamadı. 'yolov8n.pt' indiriliyor...")
        model = YOLO("yolov8n.pt")
    else:
        print(f"Model yükleniyor: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)


    print("ONNX export işlemi başlıyor...")
    success = model.export(
        format="onnx",
        opset=12,           # İstenen Opset
        dynamic=True,       # Dynamic axes (Batch size ve boyut esnekliği için)
        simplify=True      
    )
    
    if success:
        print(f"Başarılı! Model kaydedildi: {success}")
    else:
        print("Export başarısız oldu.")

if __name__ == "__main__":
    export_onnx()