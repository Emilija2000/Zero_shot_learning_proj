import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import sys
sys.path.append('./dataPrep')
import data_utils

class SemanticSpaceDataset(Dataset):
    # load the dataset
    def __init__(self, data_path, labels_path, word_emb_path, num_classes=8):

        self.num_classes = num_classes
        
        # load all files
        self.data = data_utils.load_data(data_path)

        self.labels = data_utils.load_data(labels_path)
        self.labels = self.labels.astype(int)

        # match image label and word representation
        self.word_data = data_utils.load_data(word_emb_path).astype(np.float32)
        self.word_emb_labels = self.word_data[self.labels,:]

        #self.labels = torch.from_numpy(self.labels).to(torch.long)

        # map labels to 0:num_classes-1
        self.unique_labels = np.unique(self.labels)
        label_mapping = np.zeros(np.max(self.unique_labels)+1)
        for i in range(len(self.unique_labels)):
            label_mapping[self.unique_labels[i]] = i
        self.labels_mapped = label_mapping[self.labels]

        self.labels_mapped = torch.from_numpy(self.labels_mapped).to(torch.long)

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx,:])
        label = self.labels_mapped[idx]
        return data,label
    
    def get_label(self, label_idx):
        return self.unique_labels[label_idx]


class AllFtrsDataset(Dataset):
    # load the dataset
    def __init__(self, image_ftrs_path, semantic_ftrs_path, labels_path, word_emb_path, unseen_classes):
        
        # load all files
        self.imgs = data_utils.load_data(image_ftrs_path)

        self.semantic = data_utils.load_data(semantic_ftrs_path)

        self.labels = data_utils.load_data(labels_path)
        self.labels = self.labels.astype(int)

        # match image label and word representation
        self.word_data = data_utils.load_data(word_emb_path).astype(np.float32)
        self.word_emb_labels = self.word_data[self.labels,:]

        #self.labels = torch.from_numpy(self.labels).to(torch.long)

        max_label = np.max(self.labels)
        self.sup_labels = [i for i in range(max_label+1) if not(i in unseen_classes)]
        self.sup_labels = np.array(self.sup_labels)

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx,:])
        sem = torch.from_numpy(self.semantic[idx,:])
        label = self.labels[idx]
        return img,sem,label
    
    def get_label(self, label_idx):
        return self.sup_labels[label_idx]

if __name__ == '__main__':

    config = data_utils.load_config()

    train_data_path = config['DATASET']['SEMANTIC']['train_ftrs_path']
    train_labels_path = config['DATASET']['SEMANTIC']['train_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    train_dataset = SemanticSpaceDataset(train_data_path, train_labels_path, word_embs_path)

    print(train_dataset[0])
