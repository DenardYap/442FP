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
from torchvision import models
import torch.nn.functional as F

def load_vgg_params(is_transformer):
    vgg16 = models.vgg16(pretrained=True)
    
    if is_transformer:
        vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        num_features = vgg16.classifier[6].in_features
        vgg16.classifier[6] = nn.Linear(num_features, 512)
    else:
        vgg16.classifier = nn.Identity()
    return vgg16

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
        # Adjust to index based on sequence length and broadcast across batch size
        x = x + self.pe[:x.size(0), :].unsqueeze(1).expand(-1, x.size(1), -1)
        return x
num_classes = 5 

# Transformer Model
class LipReadingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ImageCNNTransformer()
        # self.cnn = load_vgg_params()
        self.pos_encoder = PositionalEncoding(d_model=256)
        encoder_layers = TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6, norm=nn.LayerNorm(256))
        self.decoder = nn.Linear(256, num_classes)  

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
        return output


class RNNModel(nn.Module):
    def __init__(self, num_classes=5, hidden_size=1024, num_layers=1):
        super().__init__()
        # CNN for feature extraction (using VGG16)
        # self.cnn = load_vgg_params()
        self.cnn = ImageCNNTransformer()

        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, src):
        # src : B x 29 x 96 x 96 (batch size, timesteps, height, width)
        batch_size, timesteps, H, W = src.size()

        src = src.view(batch_size * timesteps, 1, H, W)
        src = self.cnn(src) 
        src = src.view(batch_size, timesteps, -1)
        output, _ = self.rnn(src)
        output = self.fc(output[:, -1, :])

        return output

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ImageCNNTransformer()
        self.fc = nn.Linear(256*29, 5)
    def forward(self, x):
        batch_size, timesteps, H, W = x.size()
        x = x.view(batch_size * timesteps, 1, H, W)
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        return self.fc(x)
        
        
# class LinearClassifier(nn.Module):
#     def __init__(self):
#         super(LinearClassifier, self).__init__()
#         self.cnn = ImageCNNTransformer()
#         self.fc1 = nn.Linear(256 * 29, 512)
#         self.fc2 = nn.Linear(512, 5)

#     def forward(self, x):
#         batch_size, timesteps, C, H, W = x.size()
#         c_in = x.view(batch_size * timesteps, C, H, W)
#         c_out = self.cnn(c_in)
#         c_out = c_out.view(batch_size, -1)
#         c_out = F.relu(self.fc1(c_out))
#         c_out = self.fc2(c_out)
#         return c_out