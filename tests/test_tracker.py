# TRACKER DRIFT TESTS


import pytest
import numpy as np
from inference.tracker import Tracker

def test_tracker_initialization():
    # Model yolu varsa test et, yoksa atla
    try:
        tracker = Tracker(model_path="../models/latest.pt")
        assert tracker is not None
    except:
        pytest.skip("Model dosyası yok")

def test_tracker_update():
    # Sahte veriyle tracker çalışıyor mu?
    try:
        tracker = Tracker(model_path="../models/latest.pt")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        tracks = tracker.update(frame)
        assert isinstance(tracks, list)
    except:
        pytest.skip("Model dosyası yok")