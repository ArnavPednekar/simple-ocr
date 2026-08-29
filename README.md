# Simple OCR from Scratch

A lightweight Optical Character Recognition (OCR) pipeline built from scratch using **PyTorch**, **CRNN (Convolutional Recurrent Neural Network)**, and **CTC Loss**.

## Project Architecture
1. **Synthetic Data Generator (`src/generate_data.py`):** Renders random alphanumeric words onto synthetic images with configurable fonts.
2. **Dataset & DataLoader (`src/dataset.py`):** Handles image loading, preprocessing, normalization, aspect-ratio-preserving resizing, and CTC target encoding.
3. **CRNN Model (`src/model.py`):** 
   - **CNN Feature Extractor:** Extracts spatial feature maps from text line images.
   - **BiLSTM Sequence Encoder:** Captures character context left-to-right and right-to-left.
   - **Linear Classifier:** Maps hidden states to vocabulary classes.
4. **Training Loop (`src/train.py`):** Trains the network using Connectionist Temporal Classification (CTC) loss.
5. **Inference (`src/infer.py`):** Decodes model predictions using greedy CTC decoding.
6. **Cleanup (`src/clean.py`):** Removes temporary synthetic datasets.

---

## Getting Started

### 1. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python src/generate_data.py
```

### 3. Train the OCR Model
```bash
python src/train.py
```

### 4. Run Inference
```bash
python src/infer.py
```

### 5. Clean Generated Data
```bash
python src/clean.py
```
