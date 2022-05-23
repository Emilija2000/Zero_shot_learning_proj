import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
import os
sys.path.append('./dataPrep')
from data_utils import load_config

from final_dataset import SemanticSpaceDataset
from supervised_model import SupervisedModel


if __name__=='__main__':

    config = load_config()

    torch.manual_seed(config['CLASSIFIER']['random_seed'])

    # load dataset
    #test_data_path = config['DATASET']['SEMANTIC']['test_ftrs_unseen_path']
    test_data_path = config['DATASET']['IMAGES']['imgs_seen_ftrs_test_path']
    test_labels_path = config['DATASET']['SEMANTIC']['test_labels_unseen_path']
    #test_data_path = config['DATASET']['IMAGES']['imgs_10cls_ftrs_test_path'] 
    #test_labels_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"],"test_labels.pkl")
    
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    dataset = SemanticSpaceDataset(test_data_path, test_labels_path, word_embs_path)
    
    batch_size = config['CLASSIFIER']['batch_size']
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # initialize the model
    #in_features = config['WORD_EMB']['ftrs_size']
    in_features = config['IMAGE_EMB']['ftrs_size']
    out_features = len(config['DATASET']['classes']) - len(config['DATASET']['unseen'])
    #out_features = len(config['DATASET']['classes']) 
    model = SupervisedModel(in_features,out_features)
    file_name = config['CLASSIFIER']['model_path']
    model.load_state_dict(torch.load(file_name))
    model.eval()


    correct = 0.0
    all = 0.0
    predictions = []        
    
    for i,(data,labels) in enumerate(dataloader):
        with torch.no_grad():
            out = model(data)
            correct += torch.sum(torch.eq(torch.argmax(out, 1), labels))
            all += out.shape[0]
            predictions = predictions + (torch.argmax(out, 1).detach().tolist())


    from sklearn.metrics import classification_report
    print(classification_report(dataset.labels, dataset.get_label(predictions), digits=4))
    
    
    

    