# Simple OCR & OpenCV Document Scanner

A robust, Optical Character Recognition (OCR) and document scanning pipeline powered by **EasyOCR** and **OpenCV**, supporting both printed text and handwritten notes.

## Features
1. **100% Local & Offline OCR (`EasyOCR`):** Pretrained PyTorch models running entirely offline without external API keys.
2. **Document Contour Detection (`OpenCV`):** Uses Otsu thresholding and area filtering to isolate the white paper sheet from dark backgrounds, avoiding stray contours.
3. **Perspective Warping & Adaptive Thresholding (`four_point_transform`):** Straightens angled/skewed document photos into a clean top-down view and binarizes text using adaptive thresholding.
4. **Auto-Orientation Correction:** Automatically detects horizontal/landscape images and rotates them to vertical portrait orientation.
5. **Web UI (`src/app.py`):** Gradio interface supporting:
   - File Upload (Images & PDFs)
   - Visual annotations (Paper sheet contours + text bounding boxes)

---

## Getting Started

### 1. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Web UI (Images & PDFs)
```bash
python src/app.py
```
Open `http://127.0.0.1:7860` in your browser (or use the generated public link) to upload images or PDF documents.

### 3. Run CLI Inference
```bash
python src/infer.py
```
