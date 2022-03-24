import json
import pickle
import time
import torch
from torch.utils.data import DataLoader

from image_dataset import ImageDataset
from image_emb_model import OmpT


CONFIG_FILE_NAME = 'config.json'
 
if __name__ == '__main__':
    # read configuration
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)

    in_channels = 3
    out_channels = config['IMAGE_EMB']['encoders']['omp1-t']['ftrs_size']
    patch_size = config['IMAGE_EMB']['encoders']['omp1-t']['patch_size']
    alpha = config['IMAGE_EMB']['encoders']['omp1-t']['alpha']
    stride = config['IMAGE_EMB']['encoders']['omp1-t']['stride']
    padding = config['IMAGE_EMB']['encoders']['omp1-t']['padding']
    pooling = config['IMAGE_EMB']['encoders']['omp1-t']['pooling']
    img_size = config['DATASET']['imgs_size']


    # load pretrained model parameters and init model
    file_name = config['DATASET']['imgs_path']+"\\dict.pkl"
    model = OmpT(in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        pretrained=True,
        weights_path=file_name,
        alpha=alpha,
        stride=stride,
        padding=padding,
        pool_param=pooling)


    # read training data
    file_name_data = config['DATASET']['imgs_path']+"\\train_data.pkl"
    file_name_labels = config['DATASET']['imgs_path']+"\\train_labels.pkl"
    b_size=8 #RAM limited
    dataset = ImageDataset(file_name_data,file_name_labels,img_size,in_channels)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

    embeddings = []

    curr_data_batch = 1
    data_batch_size = dataset.__len__()/5
    print('Extracting image embeddings...')
    for i,(data,labels) in enumerate(dataloader):
        if(i<(curr_data_batch-1)*data_batch_size):
            continue
        if(i>=curr_data_batch*data_batch_size):
            break

        if(i%1000==0):
            print("Extracted {}/{}".format(i-(curr_data_batch-1)*data_batch_size,data_batch_size))

        out=model(data)
        if(b_size>1):
            embeddings = embeddings+out.tolist() 
        else:
            embeddings.append(out.tolist())
    
    file_path = config['DATASET']["imgs_ftrs_path"]+"\\image_embs_omp1t_batch"+str(curr_data_batch)+".pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
        