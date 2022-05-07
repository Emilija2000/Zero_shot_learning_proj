import numpy as np
import torch
from torch.utils.data import Dataset

import os
import sys
sys.path.append('./dataPrep')
import data_utils

class ImgWordEmbDatasetSeen(Dataset):
    # load the dataset
    def __init__(self, img_emb_path, img_emb_filename, img_batch_num, img_emb_labels, word_emb_path, classes, unseen):

        # load all files
        x = data_utils.load_image_embeddings(img_emb_path, img_emb_filename, img_batch_num)
        x = x.astype(np.float32)

        self.img_labels = data_utils.load_data(img_emb_labels)
        self.img_labels = self.img_labels.astype(int)

        # eliminate unseen classes
        uns = np.where(np.in1d(classes,unseen))
        ind = 1-np.in1d(self.img_labels, uns)
        self.img_labels = self.img_labels[ind]
        self.img_data = x[ind,:]

        # match image label and word representation
        self.word_data = data_utils.load_data(word_emb_path).astype(np.float32)
            
        self.labels = self.word_data[self.img_labels,:]

    def __len__(self):
        return len(self.img_labels)
  
    def __getitem__(self, idx):
        data = torch.from_numpy(self.img_data[idx,:])
        label = torch.from_numpy(self.labels[idx])
        return data,label

    def ftrs_size(self):
        return len(self.img_data[0,:])



if __name__ == '__main__':

    config = data_utils.load_config()

    img_ftrs_path = config['DATASET']['IMAGES']["imgs_ftrs_path"]
    img_ftrs_filename = config['DATASET']['IMAGES']["imgs_ftrs_filename"]
    img_batch_num = config['DATASET']['IMAGES']['imgs_batch_num']
    img_label_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"], "train_labels.pkl")
    word_ftrs_path = config['DATASET']['WORDS']['word_ftrs_path']
    classes = config['DATASET']['classes']
    unseen = config['DATASET']['unseen']

    dt = ImgWordEmbDatasetSeen(img_ftrs_path, img_ftrs_filename, img_batch_num, img_label_path, word_ftrs_path, classes, unseen)

    print(dt[0])