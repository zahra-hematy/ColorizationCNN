import os
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray


X = []
SIZE = 256

class Data(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transforms.Compose([
                       transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                       transforms.RandomHorizontalFlip(), # A little data augmentation!
                       ])
        self.image_list = os.listdir(image_path)  # List all files in the directory


    def __len__(self):
        return len(self.image_list)  # Return the number of images in the dataset
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_path, self.image_list[idx]))  # Load image using the file path

        img = self.transform(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] 
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        return {'L': L, 'ab': ab}


full_dataset = Data('dataset')
Xtrain = int(0.95 * len(full_dataset))
Xtrain =int(Xtrain)
test_size = len(full_dataset) - Xtrain
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [Xtrain, test_size])
type(Xtrain), Xtrain
loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                     batch_size = 5,
                                     shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=5,
                                          shuffle=False)