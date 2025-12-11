# AI Field Application Engineer Assessment - Dataguess

This repository contains a complete **Edge AI Video Analytics System** developed for the Dataguess AI FAE Technical Assessment. The project demonstrates an end-to-end pipeline including model training, optimization (ONNX/TensorRT), and deployment via FastAPI.

## 🚀 Key Features

- **Model Training:** YOLOv8 trained on **VisDrone** dataset with advanced augmentations (Mosaic, MixUp).
- **Optimization:** Automated pipeline for PyTorch (`.pt`) $\to$ ONNX $\to$ TensorRT (`.engine`) conversion.
- **Inference Engine:** Multi-backend support (ONNX Runtime / TensorRT) with **ByteTrack** for real-time object tracking.
- **Deployment:** REST API served via **FastAPI** with Docker support.
- **Monitoring:** Real-time FPS monitoring and JSON-based inference logging.

## 📂 Project Structure

```text
cv-advanced-assessment/
├── training/       # Training scripts & configs
├── optimization/   # ONNX/TensorRT export scripts
├── inference/      # Detector, Tracker & Video Engine
├── api/            # FastAPI server & Dockerfile
├── monitoring/     # Logging & FPS utilities
├── tests/          # Unit tests (Pytest)
└── models/         # Model storage
```
