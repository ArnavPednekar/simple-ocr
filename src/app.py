import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from PIL import Image
import numpy as np
import gradio as gr
import pymupdf as fitz  # PyMuPDF for PDF support
import easyocr
import cv2

# Initialize local EasyOCR Reader (100% offline, free, local execution via PyTorch)
reader = easyocr.Reader(['en'], gpu=False)

def find_document_contour(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Otsu's thresholding to cleanly separate white paper from dark background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    doc_cnt = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > (image_np.shape[0] * image_np.shape[1] * 0.2): # at least 20% of image area
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_cnt = approx
                break
                
    # Fallback to Canny edge detection if Otsu didn't find 4 corners
    if doc_cnt is None:
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 1000:
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

def process_document(file_obj):
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
    
    # 2. OpenCV Contour Detection & Perspective Warp targeting the white paper sheet
    contour = find_document_contour(img_np)
    if contour is not None:
        cv2.drawContours(annotated_img, [contour], -1, (255, 0, 0), 3) # Blue contour for paper
        warped_np = four_point_transform(img_np, contour)
    else:
        h, w, _ = img_np.shape
        cv2.rectangle(annotated_img, (10, 10), (w-10, h-10), (255, 0, 0), 3)
        warped_np = img_np
        
    # 3. Grayscale Conversion & Adaptive Thresholding for clean text recognition
    processed_img = apply_adaptive_threshold(warped_np)
    
    # 4. Run local EasyOCR with paragraph grouping and top-to-bottom line sorting
    results = reader.readtext(processed_img, paragraph=True)
    results = sorted(results, key=lambda x: x[0][0][1])
    
    text_lines = []
    for (bbox, text) in results:
        text_lines.append(text)
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        cv2.drawContours(annotated_img, [pts], -1, (0, 255, 0), 2) # Green boxes for text
        top_left = (int(bbox[0][0]), max(int(bbox[0][1]) - 10, 15))
        cv2.putText(annotated_img, f"{text}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    return Image.fromarray(annotated_img), "\n".join(text_lines)

with gr.Blocks(title="100% Local OCR & Document Scanner") as demo:
    gr.Markdown("# 100% Local OCR & OpenCV Document Scanner")
    gr.Markdown("Runs entirely offline using local PyTorch models (EasyOCR) and OpenCV paper contour detection with perspective warping.")
    
    with gr.Row():
        file_input = gr.File(label="Upload Image or PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"])
    file_btn = gr.Button("Scan Document & Extract Text")
    with gr.Row():
        file_img_output = gr.Image(label="Annotated Document (Paper Sheet Contour)")
        file_text_output = gr.Textbox(label="Extracted OCR Text", lines=12)
    file_btn.click(fn=process_document, inputs=file_input, outputs=[file_img_output, file_text_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
