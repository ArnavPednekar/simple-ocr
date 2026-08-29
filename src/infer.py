import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from model import CRNN
from dataset import VOCAB, IDX2CHAR

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

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_height = 32
    num_classes = len(VOCAB) + 1
    
    model = CRNN(img_height=img_height, num_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/ocr_model.pth", map_location=device))
    model.eval()
    
    image = Image.open(image_path).convert("L")
    w, h = image.size
    new_w = max(int(w * (img_height / h)), 16)
    image = image.resize((new_w, img_height), Image.Resampling.BILINEAR)
    
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze(1)
        
    return decode_predictions(output)

if __name__ == "__main__":
    synthetic_dir = "data/synthetic"
    files = [f for f in os.listdir(synthetic_dir) if f.endswith(".png")]
    
    print("Testing on 5 sample images:")
    for sample_file in files[:5]:
        sample_path = os.path.join(synthetic_dir, sample_file)
        true_text = sample_file.split("_")[1].split(".")[0]
        pred_text = predict(sample_path)
        print(f"File: {sample_file} | True: {true_text} | Predicted: {pred_text}")
