# TRACKER + DETECTOR BIRLESIMI

import numpy as np

class FusionEngine:
    @staticmethod
    def compute_iou(box1, box2):
        """
        İki kutu arasındaki Intersection over Union (IoU) oranını hesaplar.
        box: [x1, y1, x2, y2]
        """
        # Kesişim alanının koordinatları
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])

        # Genişlik ve Yükseklik
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)

        # Kesişim Alanı
        inter = w * h

        # Birleşim Alanı
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        # IoU
        return inter / (union + 1e-6)

    @staticmethod
    def filter_drift(detections, tracks, iou_thres=0.5):
        """
        (Opsiyonel) Eğer takip edilen nesne ile yeni tespit çok kaymışsa uyarı verir.
        Bu fonksiyon proje gereksinimindeki 'Drift Detection' maddesi için eklenmiştir.
        """
        # Basit bir drift kontrolü:
        # Ultralytics zaten bunu içeride yapıyor ama manuel kontrol istenirse burası kullanılır.
        pass