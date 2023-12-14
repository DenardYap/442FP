import numpy as np 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataloader import Dataset442FP 
from CNN442FP import ImageCNNRaw, VideoCNN3D
from Transformer442FP import LipReadingTransformer, load_vgg_params, RNNModel, VideoClassifier, LinearClassifier
import torch
import os
import matplotlib as plt
import numpy as np
import pandas as pd
import pdb
from torch.utils.data import Dataset, DataLoader
import gc
import warnings
import sys
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm.auto import tqdm
from greg_script import *
import sys
import random
import time 
from torchvision import models
from custom_tcn import TemporalConvNet
torch.manual_seed(69)
np.random.seed(69)
random.seed(69)

classes = ["ABOUT", "BECAUSE", "CALLED", "DAVID", "EASTERN"]

def load_pretrained_weights(trained_model_path, new_model):
    trained_state_dict = torch.load(trained_model_path)['state_dict']
    
    for name, param in new_model.named_parameters():
        if name in trained_state_dict and 'cnn' in name:
            param.data = trained_state_dict[name]
            param.requires_grad = False

    return new_model


def formatData(image_name, partition): 
    """
    Given an image_name, find the respective .npz file in the directory 
    and format it in the format that the dataLoader expected

    
    image_name : str - the path to the image 
    partition : str - either train, test, or val 
    returns : a numpy array in shape N x 29 x 96 x 96 
    """
    assert partition == "train" or partition == "test" or partition == "val"
    image_name = image_name.upper()
    dirPath = f"./datasets/visual_data/{image_name}/{partition}"
    numOfFiles = os.listdir(dirPath)
    print(f"Num of files found for {dirPath} is {str(len(numOfFiles))}")
    res = []

    for i in range(1, len(numOfFiles) + 1):
        index = str(i).zfill(5)
        npz_data = np.load(f'{dirPath}/{image_name}_{index}.npz')
        res.append(npz_data["data"])

    return np.array(res, dtype=np.float32)

train_ = {}
val_ = {}
test_ = {}
for class_ in classes:
    train_[class_] = formatData(class_, "train")
for class_ in classes:
    val_[class_] = formatData(class_, "val")
for class_ in classes:
    test_[class_] = formatData(class_, "test")
print(train_["ABOUT"].shape) # 1000 X 29 X 96 X 96 
tr_loader = DataLoader(Dataset442FP("train"), batch_size=64, shuffle=True)
va_loader = DataLoader(Dataset442FP("val"), batch_size=8, shuffle=False)
te_loader = DataLoader(Dataset442FP("test"), batch_size=8, shuffle=False)


def save_checkpoint(model, epoch, checkpoint_dir, stats, info):
    """Save a checkpoint file to checkpoint_dir."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats}

    name = "b{b}_lr{lr}_p{p}_wd{wd}".format(b = info["batch"],
                                                 lr = info["lr"],
                                                 p = info["p"],
                                                 wd = info["wd"])

    checkpoint_dir = os.path.join(checkpoint_dir, name)
    checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)


    filename = os.path.join(checkpoint_dir, f"epoch={epoch}.checkpoint.pth.tar")
    torch.save(state, filename)
    
def trial(batch_size_in, learning_rate_in, momentum_in, weight_decay_in, save_folder, reg):
    print(f'save_folder:{save_folder}')
    
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    
    # tr_loader,va_loader,te_loader  = get_train_val_test_loaders(batch_size_in)
    
    # model = ImageCNNRaw()
    model = LipReadingTransformer()
    # model = load_vgg_params()
    # model = RNNModel()
    # model = VideoClassifier()
    # model = LinearClassifier()
    # model = VideoCNN3D()
    # model = TemporalConvNet()
    print(f'device being used: {device}')
    model.to(device)
    
    # print("loading pretrained weights")
    
    # trained_model_path = '/home/miyen/442FP/cnn_transformer_embeddings:01:56:46 12/10/23 CST/b64_lr1e-05_p0.9_wd0.0001/checkpoints/epoch=20.checkpoint.pth.tar'
    # load_pretrained_weights(trained_model_path, model)
    
    start_epoch = 0
    stats = []
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate_in, weight_decay=weight_decay_in)

    saved_path = os.path.join(save_folder,
                              f"b{batch_size_in}_lr{learning_rate_in}_p{momentum_in}_wd{weight_decay_in}")
    info = {"batch":batch_size_in, "lr":learning_rate_in,"p": momentum_in, "wd": weight_decay_in}
    
    if not os.path.exists(saved_path):
        os.makedirs(saved_path, exist_ok=True)
    
    
    print("inital eval")
    evaluate_epoch(tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats,
                   device,info, save_folder,reg)


    global_min_loss = stats[-1][-2]
    
    patience = 5
    curr_count_to_patience = 0
    
    # Loop over the entire dataset multiple times
    epoch = start_epoch
    print(f"Entering train loop for lr:{learning_rate_in} p:{momentum_in} wd:{weight_decay_in}")
    while curr_count_to_patience < patience:
        print(f"starting epoch {epoch}")
        
        # Train model
        start_time = time.time()
        train_epoch(tr_loader, model, criterion, optimizer, device)
        # Evaluate model
        evaluate_epoch(tr_loader, va_loader, te_loader,model, criterion, epoch + 1, stats,
                       device, info, save_folder,reg)
        print(f"Time for epoch {epoch}: {time.time() - start_time}")

        # Save model parameters
        save_checkpoint(model, epoch + 1, save_folder, stats, info)

        if epoch > 1:
            curr_count_to_patience, global_min_loss = early_stopping(stats, curr_count_to_patience, global_min_loss)
        epoch += 1
    print(f"Finished Training after {epoch} epochs")
    
trial(64, 1e-5, 0.9, 1e-4,f"transformer-full:{time.strftime('%X %x %Z')}",reg=True)
