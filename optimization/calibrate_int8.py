# INT8 ICIN KALIBRASYON

from ultralytics import YOLO

def calibrate():
    """
    Bu script, YOLOv8'in kendi içindeki kalibrasyon mekanizmasını tetikler.
    Gerçek kalibrasyon 'build_trt_engine.py' içinde export(int8=True)
    komutuyla otomatik yapılır.
    """
    print("Kalibrasyon işlemi build_trt_engine.py üzerinden yürütülmektedir.")
    print("Veri seti: datasets/VisDrone/val")
    # Temsili kod:
    # model.export(format='engine', int8=True, data='VisDrone.yaml')

if __name__ == "__main__":
    calibrate()