import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 32 -> 16
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 16 -> 8
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # Height 8 -> 4, Width unchanged
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # Height 4 -> 2, Width unchanged
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), # Height 2 -> 1
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        
        # 2. RNN Sequence Encoder (Bidirectional LSTM)
        self.lstm = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.2)
        
        # 3. Linear Classifier
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x) # [B, 512, 1, W_prime]
        features = features.squeeze(2) # [B, 512, W_prime]
        features = features.permute(0, 2, 1) # [B, W_prime, 512]
        
        rnn_out, _ = self.lstm(features) # [B, W_prime, hidden_size * 2]
        output = self.classifier(rnn_out) # [B, W_prime, num_classes]
        
        # CTC loss expects [W_prime, B, num_classes]
        output = output.permute(1, 0, 2)
        
        return output
