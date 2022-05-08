import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('./dataPrep')
from data_utils import load_config

from final_dataset import SemanticSpaceDataset
from supervised_model import SupervisedModel

torch.manual_seed(43) #42 je teži validacioni mng??

if __name__=='__main__':

    config = load_config()

    # load dataset
    train_data_path = config['DATASET']['SEMANTIC']['train_ftrs_path']
    train_labels_path = config['DATASET']['SEMANTIC']['train_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    dataset = SemanticSpaceDataset(train_data_path, train_labels_path, word_embs_path)

    # train test val split
    total_count = dataset.__len__()
    train_count = int(0.8* total_count) 
    test_count = 0
    valid_count = total_count - train_count - test_count
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))

    batch_size = config['CLASSIFIER']['batch_size']
    dataloader = {'train':DataLoader(train_dataset, shuffle=True, batch_size=train_count),
        'val':DataLoader(val_dataset, batch_size=valid_count)}

    # initialize the model
    in_features = config['WORD_EMB']['ftrs_size']
    out_features = len(config['DATASET']['classes']) - len(config['DATASET']['unseen'])
    model = SupervisedModel(in_features,out_features)

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
    #optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=weight_decay)
    optimizer = torch.optim.LBFGS(model.parameters(),lr)
   
   
    # train
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for i in range(num_epochs):
        for phase in ['train','val']:
            correct = 0.0
            all = 0.0
            running_loss = 0.0

            for _,(data,labels) in enumerate(dataloader[phase]):
                data = data.to(device)
                labels = labels.to(device)
                '''
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
                    loss.backward()
                    return loss

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

                running_loss += loss.item()

            if phase == 'train':
                train_loss.append(running_loss)
                acc = correct/all
                train_acc.append(acc.item())
            else:
                val_loss.append(running_loss)
                acc = correct/all
                val_acc.append(acc.item())

        print('Iter {}:\ntrain loss {}, val loss {}, \ntrain acc {}, val acc {}'.format(i, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))
              
                    
    # visualize training
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(1,2)
    ax[0].plot(train_loss,'b-')
    ax[0].plot(val_loss,'r-')
    ax[0].set_title('Loss')
    ax[1].plot(train_acc,'b-')
    ax[1].plot(val_acc,'r-')
    ax[1].set_title('Accuracy')
    plt.show()

    # save model
    model = model.to('cpu')
    file_name = config['CLASSIFIER']['model_path']
    torch.save(model.state_dict(), file_name)

    from sklearn.metrics import classification_report

    for i, (data,labels) in enumerate(dataloader['val']):
        out = model(data)
        out = torch.argmax(out, 1)

        print(classification_report(labels, out.detach().numpy()))

    