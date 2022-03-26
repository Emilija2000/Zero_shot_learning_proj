import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    # load the dataset
    def __init__(self, path_data, path_labels, transf_wh, transf_m, ch = 3, img_size=32, patch_size=6, stride=1, padding=0):

        with open(path_data, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        data = data.reshape((-1,ch,img_size,img_size))
        self.data = torch.from_numpy(data)

        with open(path_labels, 'rb') as f:
            self.labels = pickle.load(f, encoding='bytes')

        self.patch_size=patch_size
        self.img_size=img_size
        self.ch = ch
        self.stride=stride
        self.padding = padding

        self.transf = torch.from_numpy(transf_wh).float()
        self.transf_m = torch.Tensor(transf_m).float()
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.labels)
 
    # get a row at an index
    def __getitem__(self, idx):
        # divide into patches
        img = self.data[idx,:,:,:]
        
        if (self.padding>0):
            pad_dim = [self.padding, self.padding, self.padding, self.padding]
            img = nn.functional.pad(img, pad_dim, 'constant',0)

        patches = torch.zeros(((self.img_size+2*self.padding-self.patch_size+1)**2, self.ch*self.patch_size**2))
        i=0
        for x in range(0,img.shape[1]-self.patch_size+1, self.stride):
            for y in range(0,img.shape[2]-self.patch_size+1, self.stride):
                patches[i,:] = img[:,x:x+self.patch_size,y:y+self.patch_size].flatten()  
                i = i+1
        
        # normalize individual patches
        num_patches = patches.shape[0]
        std = torch.std(patches,1).reshape(num_patches,1)
        std[std==0] = 1
        patches = torch.divide((patches - torch.mean(patches,1).reshape(num_patches,1)),std)

        # zca whitening of image patches
        patches = torch.matmul(patches-self.transf_m,self.transf)

        return [patches, self.labels[idx]]
        