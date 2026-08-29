import os
from PIL import Image
import numpy as np
import gradio as gr
import fitz  # PyMuPDF for PDF support
import easyocr
import cv2
import base64
from io import BytesIO

# Initialize EasyOCR Reader for fallback
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
    # If image is horizontal (landscape: width > height), rotate 90 degrees counter-clockwise to become vertical (portrait)
    if w > h:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def apply_adaptive_threshold(img_np):
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_text_via_llm(warped_pil_img, provider, api_key):
    if provider == "EasyOCR (Local)" or not api_key:
        # Fallback to local EasyOCR with adaptive thresholding & line sorting
        processed = apply_adaptive_threshold(np.array(warped_pil_img))
        results = reader.readtext(processed, paragraph=True)
        results = sorted(results, key=lambda x: x[0][0][1])
        return "\n".join([text for (_, text) in results])
        
    base64_image = image_to_base64(warped_pil_img)
    prompt = "Transcribe all the handwritten and printed text in this document accurately in natural reading order, preserving line breaks and formatting."
    
    if provider == "OpenAI GPT-4o":
        try:
            import openai
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"
            
    elif provider == "Google Gemini":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt, warped_pil_img])
            return response.text
        except Exception as e:
            return f"Gemini Error: {str(e)}"
            
    return "Invalid provider selected."

def process_document(file_obj, provider, api_key):
    if file_obj is None:
        return None, "Please upload an image or a PDF file."
        
    file_path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
    
    if file_path and file_path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(file_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            doc.close()
        except Exception as e:
            return None, f"Error processing PDF: {str(e)}"
    else:
        if isinstance(file_obj, str):
            img = Image.open(file_obj)
        elif isinstance(file_obj, np.ndarray):
            img = Image.fromarray(file_obj)
        else:
            img = Image.open(file_obj.name) if hasattr(file_obj, "name") else Image.fromarray(file_obj)
        img_np = np.array(img.convert("RGB"))
        
    # 1. Auto-rotate horizontal/landscape images to vertical/portrait
    img_np = auto_rotate_image(img_np)
    annotated_img = img_np.copy()
    
    # 2. OpenCV Contour Detection & Perspective Warp (Warp Perspective)
    contour = find_document_contour(img_np)
    if contour is not None:
        cv2.drawContours(annotated_img, [contour], -1, (255, 0, 0), 3) # Blue contour for document
        warped_np = four_point_transform(img_np, contour)
        warped_pil = Image.fromarray(warped_np)
    else:
        h, w, _ = img_np.shape
        cv2.rectangle(annotated_img, (10, 10), (w-10, h-10), (255, 0, 0), 3)
        warped_pil = Image.fromarray(img_np)
        
    # 3. Transcribe via LLM (or EasyOCR fallback)
    extracted_text = extract_text_via_llm(warped_pil, provider, api_key)
    
    return Image.fromarray(annotated_img), extracted_text

with gr.Blocks(title="OCR - OpenCV Contours & LLM Transcription") as demo:
    gr.Markdown("# OpenCV Document Scanner + LLM Transcription (Print & Handwriting)")
    gr.Markdown("OpenCV detects document contours & warps perspective, and an LLM (GPT-4o or Gemini) accurately reads handwritten notes and printed text!")
    
    with gr.Row():
        provider_dropdown = gr.Dropdown(
            choices=["EasyOCR (Local)", "OpenAI GPT-4o", "Google Gemini"],
            value="EasyOCR (Local)",
            label="OCR / Transcription Engine"
        )
        api_key_input = gr.Textbox(
            label="API Key (Required for OpenAI / Gemini)",
            type="password",
            placeholder="sk-... or AIza..."
        )
        
    with gr.Tabs():
        with gr.TabItem("Upload File (Image / PDF)"):
            with gr.Row():
                file_input = gr.File(label="Upload Image or PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"])
            file_btn = gr.Button("Scan Document & Extract Text")
            with gr.Row():
                file_img_output = gr.Image(label="Annotated Document (OpenCV Contour)")
                file_text_output = gr.Textbox(label="Transcribed Text", lines=12)
            file_btn.click(fn=process_document, inputs=[file_input, provider_dropdown, api_key_input], outputs=[file_img_output, file_text_output])
            
        with gr.TabItem("Webcam / Photo Capture"):
            with gr.Row():
                webcam_input = gr.Image(label="Webcam / Upload Photo", sources=["upload", "webcam"], type="pil")
            webcam_btn = gr.Button("Capture & Extract Text")
            with gr.Row():
                webcam_img_output = gr.Image(label="Annotated Document")
                webcam_text_output = gr.Textbox(label="Transcribed Text", lines=12)
            webcam_btn.click(fn=process_document, inputs=[webcam_input, provider_dropdown, api_key_input], outputs=[webcam_img_output, webcam_text_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
