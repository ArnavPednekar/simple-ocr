import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configuration
OUTPUT_DIR = "data/synthetic"
NUM_IMAGES = 100
IMG_HEIGHT = 32
VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

os.makedirs(OUTPUT_DIR, exist_ok=True)
labels_file = open(os.path.join(OUTPUT_DIR, "labels.txt"), "w")

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except IOError:
    font = ImageFont.load_default()

print(f"Generating {NUM_IMAGES} synthetic word images...")

for i in range(NUM_IMAGES):
    word_len = random.randint(3, 8)
    word = "".join(random.choices(VOCAB, k=word_len))
    
    bbox = font.getbbox(word)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    img_width = max(text_width + 20, 64)
    img = Image.new("L", (img_width, IMG_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    
    y_offset = (IMG_HEIGHT - text_height) // 2
    draw.text((10, y_offset), word, fill=0, font=font)
    
    img_name = f"{i:05d}_{word}.png"
    img_path = os.path.join(OUTPUT_DIR, img_name)
    img.save(img_path)
    
    labels_file.write(f"{img_name}\t{word}\n")

labels_file.close()
print("Synthetic data generation complete!")
