import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import sys
sys.path.append('./dataPrep')
from data_utils import load_config

from img2word_dataset import ImgWordEmbDataset
from img2word_model import Img2WordModel

torch.manual_seed(1)

if __name__=='__main__':

    config = load_config()

    img_ftrs_path = config['DATASET']['IMAGES']["imgs_ftrs_path"]
    img_ftrs_filename = config['DATASET']['IMAGES']["imgs_ftrs_filename"]
    img_batch_num = config['DATASET']['IMAGES']['imgs_batch_num']
    img_label_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"], "train_labels.pkl")
    word_ftrs_path = config['DATASET']['WORDS']['word_ftrs_path']
    classes = config['DATASET']['classes']
    unseen = config['DATASET']['unseen']

    # load dataset
    batch_size = config['MAPPING']['batch_size']
    dataset = ImgWordEmbDataset(img_ftrs_path, img_ftrs_filename, img_batch_num, img_label_path, word_ftrs_path, classes, unseen)

    # train test val split
    total_count = dataset.__len__()
    train_count = int(0.8 * total_count) 
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count),generator=torch.Generator().manual_seed(42))

    dataloader = {'train':DataLoader(train_dataset, batch_size=train_dataset.__len__()),
        'val':DataLoader(val_dataset, batch_size=max(val_dataset.__len__(),batch_size)), 'test':DataLoader(test_dataset, batch_size)}
    do_val = (val_dataset.__len__()>0)

    # initialize the model
    in_features = config['IMAGE_EMB']['ftrs_size']
    out_features = config['WORD_EMB']['ftrs_size']
    hidden_size = config['MAPPING']['hidden_size']
    model = Img2WordModel(in_features,out_features,hidden_size)

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    # prepare training
    num_epochs = config['MAPPING']['num_epochs']
    lr = config['MAPPING']['lr']
    max_iter = config['MAPPING']['max_iter']
    tolerance_grad = config['MAPPING']['tolerance_grad']
    tolerance_change = config['MAPPING']['tolerance_change']
    
    loss_fcn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.LBFGS(model.parameters(),lr,max_iter,tolerance_grad=tolerance_grad,tolerance_change=tolerance_change)
   
    # train
    train_loss = []
    val_loss = []

    for i in range(num_epochs):
        for phase in ['train','val']:

            running_loss = 0.0

            for _,(data,labels) in enumerate(dataloader[phase]):
                data = data.to(device)
                labels = labels.to(device)

                # lbfgs training fcn      
                def loss_closure():
                    optimizer.zero_grad()
                    out = model(data)
                    loss = loss_fcn(out, labels)
                    loss.backward()
                    return loss

                # main training step
                if phase == 'train':
                    model.train()
                    optimizer.step(loss_closure)

                # calculate the loss for monitoring
                model.eval()
                loss = loss_closure()
                running_loss += loss.item()

            if phase == 'train':
                train_loss.append(running_loss)
            else:
                if(do_val):
                    val_loss.append(running_loss)
                    print('Iter {}: train loss {}, val loss {}'.format(i, train_loss[-1], val_loss[-1]))
                else:
                    print('Iter {}: train loss {}'.format(i, train_loss[-1]))
        
                    
    # visualize training
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_loss,'b-')
    if do_val:
        plt.plot(val_loss,'r-')
    plt.show()

    # save model
    model = model.to('cpu')
    file_name = config['MAPPING']['pretrained_model_name']
    torch.save(model.state_dict(), file_name)
