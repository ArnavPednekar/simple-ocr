import os
from PIL import Image
import numpy as np
import gradio as gr
import fitz  # PyMuPDF for PDF support
import easyocr
import cv2

# Initialize EasyOCR Reader for English
reader = easyocr.Reader(['en'], gpu=False)

def find_document_contour(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 500:
            doc_cnt = approx
            break
    return doc_cnt

def auto_rotate_image(img_np):
    h, w = img_np.shape[:2]
    # If image is horizontal (landscape: width > height), rotate 90 degrees clockwise to become vertical (portrait)
    if w > h:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    return img_np

def run_ocr_on_image(image):
    if image is None:
        return None, ""
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = image
        
    # Auto-rotate horizontal/landscape images to vertical/portrait
    img_np = auto_rotate_image(img_np)
    annotated_img = img_np.copy()
    
    # 1. Find document contour
    contour = find_document_contour(img_np)
    if contour is not None:
        cv2.drawContours(annotated_img, [contour], -1, (255, 0, 0), 3) # Blue contour for document
    else:
        h, w, _ = img_np.shape
        cv2.rectangle(annotated_img, (10, 10), (w-10, h-10), (255, 0, 0), 3)
        
    # 2. Run EasyOCR
    results = reader.readtext(img_np)
    text_lines = []
    for (bbox, text, prob) in results:
        text_lines.append(text)
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(annotated_img, [pts], -1, (0, 255, 0), 2) # Green boxes for text
        top_left = (int(bbox[0][0]), max(int(bbox[0][1]) - 10, 15))
        cv2.putText(annotated_img, f"{text}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return Image.fromarray(annotated_img), "\n".join(text_lines)

def ocr_inference(file_obj):
    if file_obj is None:
        return None, "Please upload an image or a PDF file."
    
    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    
    if file_path and file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            results = []
            first_img = None
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if page_num == 0:
                    first_img, text = run_ocr_on_image(img)
                else:
                    _, text = run_ocr_on_image(img)
                results.append(f"--- Page {page_num + 1} ---\n{text}")
            doc.close()
            return first_img, "\n\n".join(results)
        except Exception as e:
            return None, f"Error processing PDF: {str(e)}"
    else:
        if isinstance(file_obj, str):
            img = Image.open(file_obj)
        elif isinstance(file_obj, np.ndarray):
            img = Image.fromarray(file_obj)
        else:
            img = Image.open(file_obj.name) if hasattr(file_obj, "name") else Image.fromarray(file_obj)
            
        return run_ocr_on_image(img)

def webcam_inference(cam_image):
    if cam_image is None:
        return None, "No webcam image captured."
    return run_ocr_on_image(cam_image)

with gr.Blocks(title="OCR - Document Contour & Text Extraction") as demo:
    gr.Markdown("# OCR & Document Contour Detection (Print & Handwritten Notes)")
    gr.Markdown("Upload printed text, handwritten notes, PDFs, or use your webcam to detect document contours, text boxes, and extract text.")
    
    with gr.Tabs():
        with gr.TabItem("Upload File (Image / PDF)"):
            with gr.Row():
                file_input = gr.File(label="Upload Image or PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"])
            file_btn = gr.Button("Run OCR & Detect Contours")
            with gr.Row():
                file_img_output = gr.Image(label="Annotated Image (Blue: Document Contour, Green: Text Boxes)")
                file_text_output = gr.Textbox(label="Extracted OCR Text", lines=10)
            file_btn.click(fn=ocr_inference, inputs=file_input, outputs=[file_img_output, file_text_output])
            
        with gr.TabItem("Webcam Capture"):
            with gr.Row():
                webcam_input = gr.Image(label="Webcam / Upload Photo", sources=["upload", "webcam"], type="pil")
            webcam_btn = gr.Button("Capture & Extract Text")
            with gr.Row():
                webcam_img_output = gr.Image(label="Annotated Image")
                webcam_text_output = gr.Textbox(label="Extracted OCR Text", lines=10)
            webcam_btn.click(fn=webcam_inference, inputs=webcam_input, outputs=[webcam_img_output, webcam_text_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
