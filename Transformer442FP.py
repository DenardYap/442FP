import numpy as np 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os 
from CNN442FP import ImageCNNTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(x.shape, self.pe[:x.size(1), :].shape)
        x = x + self.pe[:x.size(1), :]
        
        return x
num_classes = 5 

# Transformer Model
class LipReadingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ImageCNNTransformer()
        self.pos_encoder = PositionalEncoding(d_model=128)
        encoder_layers = TransformerEncoderLayer(d_model=128, nhead=2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        self.decoder = nn.Linear(128, num_classes)  

    def forward(self, src):
        # src : B x 29 x 96 x 96 
        batch_size, timesteps, H, W = src.size()
        src = src.view(batch_size * timesteps, 1, H, W)
        src = self.cnn(src)  
        # EXPECTED OUTPUT: 64 x 29 x 128 
        # 64 X 29 x 128 
        src = src.view(batch_size, timesteps, -1)
        src = self.pos_encoder(src)
        src = src.view(timesteps, batch_size, -1)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=0))
        return torch.softmax(output, dim=1) 