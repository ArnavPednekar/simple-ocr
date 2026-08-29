import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(VOCAB)}
CHAR2IDX['<blank>'] = 0
IDX2CHAR = {idx: char for char, idx in CHAR2IDX.items()}

class OCRDataset(Dataset):
    def __init__(self, data_dir, labels_file_name="labels.txt", img_height=32):
        self.data_dir = data_dir
        self.img_height = img_height
        self.samples = []
        
        labels_path = os.path.join(data_dir, labels_file_name)
        with open(labels_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_name, text = self.samples[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # Load image in grayscale
        image = Image.open(img_path).convert("L")
        
        # Resize maintaining aspect ratio to fixed height
        w, h = image.size
        new_w = max(int(w * (self.img_height / h)), 16)
        image = image.resize((new_w, self.img_height), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        img_np = np.array(image, dtype=np.float32) / 255.0
        # Add channel dimension: [1, H, W]
        img_tensor = torch.tensor(img_np).unsqueeze(0)
        
        # Encode text to integer list
        target = [CHAR2IDX[c] for c in text if c in CHAR2IDX]
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return img_tensor, target_tensor

def collate_fn(batch):
    images, targets = zip(*batch)
    
    # Find max width in batch
    max_w = max(img.shape[2] for img in images)
    c, h = images[0].shape[0], images[0].shape[1]
    
    # Pad all images to max width with white (1.0)
    padded_images = torch.ones(len(images), c, h, max_w, dtype=torch.float32)
    for i, img in enumerate(images):
        w = img.shape[2]
        padded_images[i, :, :, :w] = img
        
    # Concatenate targets for CTC Loss
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flattened_targets = torch.cat(targets)
    
    return padded_images, flattened_targets, target_lengths
