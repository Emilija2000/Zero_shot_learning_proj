import pickle
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    # load the dataset
    def __init__(self, path_data, path_labels,img_size=32,ch=3):
        
        with open(path_data, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        data = data.reshape((-1,img_size,img_size,ch))
        data = torch.from_numpy(data)
        self.data = data.permute(0,3,1,2)

        with open(path_labels, 'rb') as f:
            self.labels = pickle.load(f, encoding='bytes')
        
    # number of rows in the dataset
    def __len__(self):
        return len(self.labels)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.data[idx,:,:,:], self.labels[idx]]