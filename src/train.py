import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OCRDataset, collate_fn, VOCAB
from model import CRNN

def train():
    data_dir = "data/synthetic"
    batch_size = 64
    epochs = 3
    lr = 0.001
    img_height = 32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = OCRDataset(data_dir=data_dir, img_height=img_height)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    num_classes = len(VOCAB) + 1
    model = CRNN(img_height=img_height, num_channels=1, num_classes=num_classes).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long, device=device)
            
            log_probs = torch.log_softmax(outputs, dim=2)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ocr_model.pth")
    print("Training complete! Model saved to models/ocr_model.pth")

if __name__ == "__main__":
    train()
