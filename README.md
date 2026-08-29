# Simple OCR & Document Contour Detection

A lightweight Optical Character Recognition (OCR) and document analysis pipeline powered by **EasyOCR** and **OpenCV**, supporting both printed text and handwritten notes.

## Features
1. **Pretrained OCR Engine (`EasyOCR`):** High-accuracy text recognition supporting both printed documents and handwritten notes out-of-the-box.
2. **Document Contour Detection (`OpenCV`):** Automatically detects document borders and text bounding boxes, highlighting them on annotated output images.
3. **Web UI (`src/app.py`):** Gradio web interface supporting:
   - File Upload (Images & PDFs)
   - Webcam Capture / Photo Upload
   - Visual annotations (Document contours + text bounding boxes)
4. **Live Camera Feed (`src/live_camera.py`):** Real-time webcam OCR with contour detection and automatic fallback demo mode.
5. **Synthetic Data & Training (`src/generate_data.py`, `src/train.py`):** Utilities for synthetic dataset generation and custom CRNN training.

---

## Getting Started

### 1. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Web UI (Images, PDFs & Webcam)
```bash
python src/app.py
```
Open `http://127.0.0.1:7860` in your browser (or use the generated public link) to upload documents or use your webcam.

### 3. Run Live Camera OCR CLI
```bash
python src/live_camera.py
```

### 4. Run CLI Inference
```bash
python src/infer.py
```
