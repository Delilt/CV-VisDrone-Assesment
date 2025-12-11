# GIRIS CIKIS VERI MODELLERI

from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    bbox: List[int]
    confidence: float
    class_id: int

class DetectionResponse(BaseModel):
    inference_time_ms: float
    object_count: int
    predictions: List[BoundingBox]

class MetricsResponse(BaseModel):
    avg_latency: float
    fps: float
    gpu_util: float