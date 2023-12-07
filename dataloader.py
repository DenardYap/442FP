import numpy as np 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# npz_data = np.load('./datasets/visual_data/CALLED/train/CALLED_00001.npz')
# print(type(npz_data["data"]))

# image = Image.fromarray(npz_data["data"][0])
# image.save('CALLED_00001_01.png')


classes = ["ABOUT", "BECAUSE", "CALLED", "DAVID", "EASTERN"]

# class Dataset442FP(Dataset):
#     def __init__(self, data):  
#         self.data = data            

#         # Go through each class in the classes the data in the following format
#         """
#         N x 29 x 96 x 96 x 1 matrix
#         """
#         # for class_ in classes:
#         self.idToClassNameMap = {id: className for id, className in enumerate(classes)}
    
#     def __len__(self): 
#         return len(self.data)
    
#     def __getitem__(self, id): 
#         # className = self.idToClassNameMap[classId]
#         classId = self.data[id, :, :, :, -1]
#         label = torch.zeros(len(classes))
#         label[classId] = 1
#         sample = self.data[classId]
#         print("SAMPLE!!", sample.shape)
#         # sample is a dictionary of length N, where N is the length of the data 
#         # (1000 for train and 50 for test/val)
#         return sample[i, :, :, :], label
    
class Dataset442FP(Dataset):

    def __init__(self, partition = "train"):
        self.filepaths = [] 
        self.classMap = []

        self.classNameToId = {id: val for id, val in enumerate(classes)}

        assert partition == "train" or partition == "test" or partition == "val"

        for i, class_ in enumerate(classes):
            
            dirPath = f"./datasets/visual_data/{class_}/{partition}"
            filepaths = os.listdir(dirPath)

            self.filepaths.extend([dirPath + "/" + filepath for filepath in filepaths])
            self.classMap.extend([i for _ in range(len(filepaths))])

         
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, id):
        self.filepaths[id]
        res = np.load(self.filepaths[id])
        classId = self.classMap[id]
        label = torch.zeros(len(classes))
        label[classId] = 1
        # print(res["data"])
        return np.array(res["data"]).astype(np.float32), label