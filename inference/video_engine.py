import cv2
import time
import threading
from queue import Queue, Empty
import numpy as np

from tracker import Tracker
from utils import Visualizer
from fusion import FusionEngine 

class VideoEngine:
    def __init__(self, source, model_path):
        self.source = source
        # Webcam (0) ise int'e çevir, dosya yoluysa string kalsın
        if str(source).isdigit():
            self.source = int(source)
            
        self.cap = cv2.VideoCapture(self.source)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.stop_event = threading.Event()
        
        self.tracker = Tracker(model_path=model_path)
        self.processing_fps = 0

    def capture_thread(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break
            
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)
        print("[VideoEngine] Okuma thread'i durdu.")

    def processing_thread(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            t0 = time.time()
            
            # Takip İşlemi
            tracks = self.tracker.update(frame)
            
            if len(tracks) > 0:
                
                dummy_iou = FusionEngine.compute_iou(tracks[0][:4], tracks[0][:4])
            
            t1 = time.time()
            self.processing_fps = 1 / (t1 - t0 + 1e-6)

            self.result_queue.put((frame, tracks, self.processing_fps))

    def start(self):
        print(f"[VideoEngine] Başlatılıyor... Kaynak: {self.source}")
        
        t_capture = threading.Thread(target=self.capture_thread)
        t_process = threading.Thread(target=self.processing_thread)
        
        t_capture.start()
        t_process.start()
        
        print("[VideoEngine] 'Q' tuşuna basarak çıkabilirsiniz.")
        
        while not self.stop_event.is_set():
            try:
                frame, tracks, fps = self.result_queue.get(timeout=1)
                
                frame = Visualizer.draw_detections(frame, tracks)
                frame = Visualizer.draw_fps(frame, fps)
                
                cv2.imshow("Dataguess AI Assessment - Real Time", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
                    
            except Empty:
                pass
                
        self.cap.release()
        cv2.destroyAllWindows()
        t_capture.join()
        t_process.join()
        print("[VideoEngine] Sistem kapandı.")

if __name__ == "__main__":
    # Webcam için 0, video dosyası için dosya yolu
    engine = VideoEngine(source=0, model_path="../models/latest.pt")
    engine.start()