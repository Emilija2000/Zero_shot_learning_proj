import torch
from torch.utils.data import DataLoader

from img2word_dataset import ImgWordEmbDataset
from img2word_model import Img2WordModel

import os
import sys
sys.path.append('./dataPrep')
import data_utils

if __name__ == '__main__':

    config = data_utils.load_config()

    img_ftrs_path = config['DATASET']['IMAGES']["imgs_ftrs_path"]
    img_ftrs_filename = config['DATASET']['IMAGES']["imgs_ftrs_filename"]
    img_batch_num = config['DATASET']['IMAGES']['imgs_batch_num']
    img_label_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"], "train_labels.pkl")
    word_ftrs_path = config['DATASET']['WORDS']['word_ftrs_path']

    # load image embedding datasets
    batch_size = config['MAPPING']['batch_size']

    classes = config['DATASET']['classes']
    unseen = config['DATASET']['unseen'] # train images from unseen classes are not used
    train_dataset = ImgWordEmbDataset(img_ftrs_path, img_ftrs_filename, img_batch_num, img_label_path, word_ftrs_path, classes, unseen)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    unseen = [] # extract for all test classes (including novelty classes)
    img_label_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"], "test_labels.pkl")
    test_dataset = ImgWordEmbDataset(img_ftrs_path, img_ftrs_filename, 0, img_label_path, word_ftrs_path, classes, unseen, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # load a pretrained model
    in_features = config['IMAGE_EMB']['ftrs_size']
    out_features = config['WORD_EMB']['ftrs_size']
    hidden_size = config['MAPPING']['hidden_size']
    model = Img2WordModel(in_features,out_features,hidden_size)
    
    file_name = config['MAPPING']['pretrained_model_name']
    model.load_state_dict(torch.load(file_name))
    model.eval()

    # prepare final feature vectors
    len_train = train_dataset.__len__()
    len_test = test_dataset.__len__()
    train_ftrs = torch.zeros((len_train,out_features))
    test_ftrs = torch.zeros((len_test,out_features))

    # extract feature vectors in the word vector space
    print('Extracting train feature vectors...')
    for i, (data, labels) in enumerate(train_dataloader):
        out = model(data)
        train_ftrs[i*batch_size:min((i+1)*batch_size, len_train), :] = out

    print('Extracting test feature vectors...')
    for i, (data, labels) in enumerate(test_dataloader):
        out = model(data)
        test_ftrs[i*batch_size:min((i+1)*batch_size, len_test), :] = out

    # save extracted features
    print("Saving feature vectors...")
    file_name = config['DATASET']['SEMANTIC']['train_ftrs_path']
    data_utils.save_data(file_name, train_ftrs)
    
    file_name = config['DATASET']['SEMANTIC']['test_ftrs_path']
    data_utils.save_data(file_name, test_ftrs)
    