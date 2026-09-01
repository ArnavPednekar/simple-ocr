import os
import sys

lib_dir = os.path.abspath("tesseract_local/usr/lib")
tessdata_dir = os.path.abspath("tesseract_local/usr/share/tessdata")
tesseract_bin = os.path.abspath("tesseract_local/usr/bin/tesseract")

if os.path.exists(tesseract_bin) and lib_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["TESSDATA_PREFIX"] = tessdata_dir
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = tesseract_bin

def predict(image_path):
    text = pytesseract.image_to_string(Image.open(image_path))
    return text.strip()

if __name__ == "__main__":
    synthetic_dir = "data/synthetic"
    if os.path.exists(synthetic_dir):
        files = [f for f in os.listdir(synthetic_dir) if f.endswith(".png")]
        print("Testing Tesseract OCR on sample images:")
        for sample_file in files[:5]:
            sample_path = os.path.join(synthetic_dir, sample_file)
            true_text = sample_file.split("_")[1].split(".")[0] if "_" in sample_file else ""
            pred_text = predict(sample_path)
            print(f"File: {sample_file} | True: {true_text} | Predicted: {pred_text}")
    else:
        print("Synthetic dir not found.")
