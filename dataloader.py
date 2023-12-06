import numpy as np 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import os 
npz_data = np.load('./datasets/visual_data/CALLED/train/CALLED_00001.npz')
print(type(npz_data["data"]))

# image = Image.fromarray(npz_data["data"][0])
# image.save('CALLED_00001_01.png')


classes = ["ABOUT", "BECAUSE", "CALLED", "DAVID", "EASTERN"]

INPUT_DIM = 96 
FRAME_DIM = 96 

class Dataset442FP(Dataset):
    def __init__(self, data):  
        self.data = data            
        print(len(self.data))

        # Go through each class in the classes the data in the following format
        """
        {
          "class1" :  N x 29 x 96 x 96 matrix
          "class2" :  N x 29 x 96 x 96 matrix
          ...
          "class5" :  N x 29 x 96 x 96 matrix
        }
        """
        # for class_ in classes:
        self.idToClassNameMap = {id: className for id, className in enumerate(classes)}
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, classId): 
        className = self.idToClassNameMap[classId]
        sample = self.data[className]

        # sample is a dictionary of length N, where N is the length of the data 
        # (1000 for train and 50 for test/val)
        return sample
    
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

    return np.array(res)

train_ = {}
val_ = {}
test_ = {}
for class_ in classes:
    train_[class_] = formatData(class_, "train")
for class_ in classes:
    val_[class_] = formatData(class_, "val")
for class_ in classes:
    test_[class_] = formatData(class_, "test")


trainLoader = DataLoader(Dataset442FP(train_), batch_size=64, shuffle=True)
valLoader = DataLoader(Dataset442FP(val_), batch_size=8, shuffle=True)
testLoader = DataLoader(Dataset442FP(test_), batch_size=8, shuffle=True)

