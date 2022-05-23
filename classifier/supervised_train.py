import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./dataPrep')
from data_utils import load_config

from final_dataset import SemanticSpaceDataset
from supervised_model import SupervisedModel



if __name__=='__main__':

    config = load_config()

    torch.manual_seed(config['CLASSIFIER']['random_seed']) 

    # load dataset
     # to test projection into semantic space 
    #train_data_path = config['DATASET']['SEMANTIC']['train_ftrs_path']
    #train_labels_path = config['DATASET']['SEMANTIC']['train_labels_path']
     # to train supervised image classifier based on image features
    train_data_path = config['DATASET']['IMAGES']['imgs_seen_ftrs_train_path']
    train_labels_path = config['DATASET']['SEMANTIC']['train_labels_path']
    # to train supervised 10cls classifier in image space
    #train_data_path = config['DATASET']['IMAGES']['imgs_10cls_ftrs_train_path'] 
    #import os
    #train_labels_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"],"train_labels.pkl")

    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    dataset = SemanticSpaceDataset(train_data_path, train_labels_path, word_embs_path)

    # train test val split
    total_count = dataset.__len__()
    train_count = int(0.8* total_count) 
    test_count = 0
    valid_count = total_count - train_count - test_count
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))

    batch_size = config['CLASSIFIER']['batch_size']
    #dataloader = {'train':DataLoader(train_dataset, shuffle=True, batch_size=train_count),
    #    'val':DataLoader(val_dataset, batch_size=valid_count)}
    dataloader = {'train':DataLoader(train_dataset, shuffle=True, batch_size=batch_size),
        'val':DataLoader(val_dataset, batch_size=batch_size)}


    # initialize the model
    #in_features = config['WORD_EMB']['ftrs_size'] # for semantic space vectors testing
    in_features = config['IMAGE_EMB']['ftrs_size'] # for image space vectors testing
    out_features = len(config['DATASET']['classes']) - len(config['DATASET']['unseen'])
    #out_features=len(config['DATASET']['classes'])
    model = SupervisedModel(in_features,out_features)

    #file_name = config['CLASSIFIER']['model_path']
    #model.load_state_dict(torch.load(file_name))

    # device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    # prepare training
    num_epochs = config['CLASSIFIER']['num_epochs']
    lr = config['CLASSIFIER']['lr']
    weight_decay = config['CLASSIFIER']['weight_decay']
    
    loss_fcn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=weight_decay)
    #optimizer = torch.optim.LBFGS(model.parameters(),lr)
   
   
    # train
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    best_acc = 0
    best_model = model.state_dict()
    
    for i in range(num_epochs):
        for phase in ['train','val']:
            correct = 0.0
            all = 0.0
            running_loss = 0.0

            for this_batch_size,(data,labels) in enumerate(dataloader[phase]):
                data = data.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()      
                
                # main training step
                if phase == 'train':
                    model.train()
                    out = model(data)
                    loss = loss_fcn(out, labels)

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        correct += torch.sum(torch.eq(torch.argmax(out, 1), labels))
                        all += out.shape[0]

                else:
                    model.eval()
                    with torch.no_grad():
                        out = model(data)

                        loss = loss_fcn(out, labels)

                        correct += torch.sum(torch.eq(torch.argmax(out, 1), labels))
                        all += out.shape[0]
                
                '''
                def loss_closure():
                    optimizer.zero_grad()
                    out = model(data)
                    loss = loss_fcn(out, labels)

                    l2_norm = torch.tensor(0.).to(device)
                    for p in model.parameters():
                        l2_norm+=torch.sum(torch.multiply(p,p))
                    
                    loss_nonreg = loss
                    loss = loss + l2_norm*weight_decay

                    loss.backward()
                    return loss_nonreg

                # main training step
                if phase == 'train':
                    model.train()
                    optimizer.step(loss_closure)

                # calculate the loss for monitoring
                model.eval()
                with torch.no_grad():
                    out = model(data)
                    loss = loss_fcn(out, labels)
                    running_loss += loss.item()

                    correct += torch.sum(torch.eq(torch.argmax(out, 1), labels))
                    all += out.shape[0]
                '''
                
                running_loss += loss.item()

            running_loss = running_loss/this_batch_size# for mini-batch
            if phase == 'train':
                train_loss.append(running_loss)
                acc = correct/all
                train_acc.append(acc.item())
            else:
                val_loss.append(running_loss)
                acc = correct/all
                val_acc.append(acc.item())

                if(acc.item()>best_acc):
                    best_acc = acc.item()
                    best_model = model.state_dict()

        print('Iter {}:\ntrain loss {}, val loss {}, \ntrain acc {}, val acc {}'.format(i, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))
              
                    
    # visualize training
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(1,2)
    ax[0].plot(train_loss,'b-')
    ax[0].plot(val_loss,'r-')
    ax[0].set_xlabel('broj epohe')
    ax[0].set_ylabel('kriterijumska funkcija')
    ax[0].set_title('(a)')
    ax[1].plot(train_acc,'b-')
    ax[1].plot(val_acc,'r-')
    ax[1].set_xlabel('broj epohe')
    ax[1].set_ylabel('tacnost')
    ax[1].set_title('(b)')
    plt.show()

    model.load_state_dict(best_model)
    # save model
    model = model.to('cpu')
    file_name = config['CLASSIFIER']['model_path']
    torch.save(model.state_dict(), file_name)
    '''
    model = model.to('cpu')
    file_name = config['CLASSIFIER']['model_path']
    model.load_state_dict(torch.load(file_name))
    '''
    model.eval()
    
    predictions = []   
    labels_all = []
    for i,(data,labels) in enumerate(dataloader['val']):
        with torch.no_grad():
            out = model(data)
            predictions = predictions + (torch.argmax(out, 1).detach().tolist())
            labels_all = labels_all + labels.tolist()


    from sklearn.metrics import classification_report
    print(classification_report(labels_all, predictions))

    