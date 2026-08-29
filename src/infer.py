import os
import easyocr
from PIL import Image

def predict(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image_path)
    return " ".join([text for (_, text, _) in results])

if __name__ == "__main__":
    synthetic_dir = "data/synthetic"
    if os.path.exists(synthetic_dir):
        files = [f for f in os.listdir(synthetic_dir) if f.endswith(".png")]
        print("Testing EasyOCR on sample images:")
        for sample_file in files[:5]:
            sample_path = os.path.join(synthetic_dir, sample_file)
            true_text = sample_file.split("_")[1].split(".")[0] if "_" in sample_file else ""
            pred_text = predict(sample_path)
            print(f"File: {sample_file} | True: {true_text} | Predicted: {pred_text}")
    else:
        print("Synthetic dir not found.")
