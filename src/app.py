import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import gradio as gr
import fitz  # PyMuPDF for PDF support
from model import CRNN
from dataset import VOCAB, IDX2CHAR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_height = 32
num_classes = len(VOCAB) + 1

model = CRNN(img_height=img_height, num_channels=1, num_classes=num_classes).to(device)
model_path = "models/ocr_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def decode_predictions(preds):
    pred_indices = torch.argmax(preds, dim=1).cpu().numpy()
    decoded_chars = []
    prev_idx = -1
    for idx in pred_indices:
        if idx != 0 and idx != prev_idx:
            if idx in IDX2CHAR:
                decoded_chars.append(IDX2CHAR[idx])
        prev_idx = idx
    return "".join(decoded_chars)

def run_ocr_on_image(image):
    image = image.convert("L")
    w, h = image.size
    if h == 0 or w == 0:
        return ""
    new_w = max(int(w * (img_height / h)), 16)
    image = image.resize((new_w, img_height), Image.Resampling.BILINEAR)
    
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze(1)
        
    return decode_predictions(output)

def ocr_inference(file_obj):
    if file_obj is None:
        return "Please upload an image or a PDF file."
    
    # Check if input is a PDF or an Image
    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    
    results = []
    
    if file_path and file_path.lower().endswith(".pdf"):
        # Process PDF using PyMuPDF
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = run_ocr_on_image(img)
                results.append(f"--- Page {page_num + 1} ---\n{text}")
            doc.close()
            return "\n\n".join(results)
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    else:
        # Process single image
        if isinstance(file_obj, str):
            img = Image.open(file_obj)
        elif isinstance(file_obj, np.ndarray):
            img = Image.fromarray(file_obj)
        else:
            img = Image.open(file_obj.name) if hasattr(file_obj, "name") else Image.fromarray(file_obj)
            
        return run_ocr_on_image(img)

# Gradio UI supporting both images and PDFs
demo = gr.Interface(
    fn=ocr_inference,
    inputs=gr.File(label="Upload Image or PDF File", file_types=[".png", ".jpg", ".jpeg", ".pdf"]),
    outputs=gr.Textbox(label="Extracted OCR Text", lines=10),
    title="OCR from Scratch (Images & PDFs)",
    description="Upload a text image or a PDF document to run OCR inference using our custom CRNN model."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
