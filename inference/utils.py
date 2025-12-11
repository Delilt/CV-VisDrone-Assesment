# BOYUTLANDIRMA \ NMS VB.

import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_detections(frame, tracks):
        """
        Tespitleri ve takip bilgilerini kare üzerine çizer.
        tracks: [[x1, y1, x2, y2, track_id, score, class_id], ...]
        """
        for trk in tracks:
            x1, y1, x2, y2 = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3])
            track_id = int(trk[4])
            score = trk[5]
            class_id = int(trk[6])

            # Renk belirle (ID'ye göre rastgele ama sabit renk)
            np.random.seed(track_id)
            color = np.random.randint(0, 255, size=3).tolist()

            # Kutu Çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Etiket Hazırla
            label = f"ID:{track_id} Conf:{score:.2f}"
            
            # Yazı Arka Planı
            (w, h), _ = cv2.getTextSize(label, 0, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            # Yazı Yaz
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    @staticmethod
    def draw_fps(frame, fps):
        """FPS bilgisini ekrana yazar"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame