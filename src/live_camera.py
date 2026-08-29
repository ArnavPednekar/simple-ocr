import cv2
import easyocr
import numpy as np
import os

def find_document_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
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

def main():
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\n[Notice] No physical webcam detected or accessible in this environment.")
        print("Switching to Demo Mode: processing sample images from data/synthetic/ with document contour detection...\n")
        
        synthetic_dir = "data/synthetic"
        if not os.path.exists(synthetic_dir) or not os.listdir(synthetic_dir):
            print("No synthetic images found. Please run generate_data.py first.")
            return
            
        files = [f for f in os.listdir(synthetic_dir) if f.endswith(".png")]
        for sample_file in files[:5]:
            sample_path = os.path.join(synthetic_dir, sample_file)
            img = cv2.imread(sample_path)
            if img is None:
                continue
                
            # Find document contour
            contour = find_document_contour(img)
            display_img = img.copy()
            if contour is not None:
                cv2.drawContours(display_img, [contour], -1, (255, 0, 0), 2)
            else:
                # Fallback outer border if no 4-corner contour found
                h, w, _ = img.shape
                cv2.rectangle(display_img, (5, 5), (w-5, h-5), (255, 0, 0), 2)
                
            results = reader.readtext(img)
            print(f"File: {sample_file}")
            for (bbox, text, prob) in results:
                print(f"  -> Detected Text: '{text}' (Confidence: {prob:.2f})")
                pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                cv2.drawContours(display_img, [pts], -1, (0, 255, 0), 2)
                
            cv2.imshow("Document OCR & Contour Detection", display_img)
            print("Press any key in the image window to view next image (or 'q' to quit)...")
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return
        
    print("Webcam opened successfully! Press 'q' to quit live OCR window.")
    
    frame_count = 0
    cached_results = []
    cached_contour = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        frame_count += 1
        if frame_count % 5 == 0:
            try:
                cached_contour = find_document_contour(frame)
                cached_results = reader.readtext(frame)
            except Exception as e:
                cached_results = []
                cached_contour = None
                
        # Draw document contour
        if cached_contour is not None:
            cv2.drawContours(frame, [cached_contour], -1, (255, 0, 0), 2)
            
        # Draw text bounding boxes and recognized text
        for (bbox, text, prob) in cached_results:
            pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(frame, [pts], -1, (0, 255, 0), 2)
            top_left = (int(bbox[0][0]), int(bbox[0][1]) - 10)
            cv2.putText(frame, f"{text} ({prob:.2f})", top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.imshow("Live OCR & Document Contour (Press 'q' to quit)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
