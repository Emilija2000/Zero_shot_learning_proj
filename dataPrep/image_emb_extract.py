import numpy as np
from torch.utils.data import DataLoader
import os

from image_dataset import ImageDataset
from image_emb_model import OmpT

import data_utils

if __name__ == '__main__':
    # read configuration
    config = data_utils.load_config()

    in_channels = 3
    out_channels = config['IMAGE_EMB']['encoders']['omp1-t']['ftrs_size']
    patch_size = config['IMAGE_EMB']['encoders']['omp1-t']['patch_size']
    alpha = config['IMAGE_EMB']['encoders']['omp1-t']['alpha']
    stride = config['IMAGE_EMB']['encoders']['omp1-t']['stride']
    padding = config['IMAGE_EMB']['encoders']['omp1-t']['padding']
    pooling = config['IMAGE_EMB']['encoders']['omp1-t']['pooling']
    img_size = config['DATASET']['IMAGES']['imgs_size']

    out_img_size = 1+(img_size-patch_size+2*padding)//stride

    emb_size = config['IMAGE_EMB']['ftrs_size'] = 12800

    # load pretrained model parameters 
    file_path = os.path.join(config['DATASET']['IMAGES']['imgs_path'],config['IMAGE_EMB']['dict_file_name'])
    weights,transf_m,transf_wh = data_utils.load_data(file_path)       


    # init model
    model = OmpT(out_channels=out_channels,
        out_img_size=out_img_size,
        pretrained=True,
        weights = weights,
        alpha=alpha,
        pool_param=pooling)

    # read training data
    file_path_data = os.path.join(config['DATASET']['IMAGES']['imgs_path'],"train_data.pkl")
    file_path_labels = os.path.join(config['DATASET']['IMAGES']['imgs_path'],"train_labels.pkl")
    
    b_size=16 
    dataset = ImageDataset(file_path_data, file_path_labels, transf_wh, transf_m, in_channels, img_size, patch_size, stride, padding)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)


    print('Extracting image embeddings...')
    data_batch_size = dataset.__len__()//5
    curr_data_batch = 1
    embeddings = np.zeros((data_batch_size,emb_size), dtype=np.float32)
    print('Batch ',curr_data_batch,':')

    for i,(data,labels) in enumerate(dataloader):
        if(i*b_size>=curr_data_batch*data_batch_size):
            print('Saving batch ',curr_data_batch)
            file_path = os.path.join(config['DATASET']['IMAGES']["imgs_ftrs_path"],
                config['DATASET']['IMAGES']['imgs_ftrs_filename']+"_batch"+str(curr_data_batch)+".pkl")
            data_utils.save_data(file_path,embeddings)

            curr_data_batch = curr_data_batch + 1
            if(curr_data_batch>config['DATASET']['IMAGES']['imgs_batch_num']):
                break
            print('Batch ',curr_data_batch,':')

        if(i*b_size%2000==0):
            print("Extracted {}/{}".format(i*b_size-(curr_data_batch-1)*data_batch_size,data_batch_size))
            
        out=model(data).detach().numpy()
        if(b_size==1):
            embeddings[i*b_size-(curr_data_batch-1)*data_batch_size,:] = out 
        else:
            embeddings[i*b_size-(curr_data_batch-1)*data_batch_size:i*b_size-(curr_data_batch-1)*data_batch_size+b_size,:] = out
    
    print('Saving batch ',curr_data_batch)
    file_path = os.path.join(config['DATASET']['IMAGES']["imgs_ftrs_path"],
        config['DATASET']['IMAGES']['imgs_ftrs_filename']+"_batch"+str(curr_data_batch)+".pkl")
    data_utils.save_data(file_path,embeddings)

    # read test data
    file_path_data = os.path.join(config['DATASET']['IMAGES']['imgs_path'],"test_data.pkl")
    file_path_labels = os.path.join(config['DATASET']['IMAGES']['imgs_path'],"test_labels.pkl")
    
    b_size=16 
    dataset = ImageDataset(file_path_data, file_path_labels, transf_wh, transf_m, in_channels, img_size, patch_size, stride, padding)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

    data_batch_size = dataset.__len__()
    embeddings = np.zeros((data_batch_size,emb_size), dtype=np.float32)
    
    print('Extracting test embeddings...')
    for i,(data,labels) in enumerate(dataloader):
            
        if(i*b_size%1000==0):
            print("Extracted {}/{}".format(i*b_size,data_batch_size))
            
        out=model(data).detach().numpy()
        if(b_size==1):
            embeddings[i,:] = out 
        else:
            embeddings[i*b_size:min((i+1)*b_size,dataset.__len__()),:] = out


    print('Saving test embeddings... ')
    file_path = os.path.join(config['DATASET']['IMAGES']["imgs_ftrs_path"],
            config['DATASET']['IMAGES']['imgs_ftrs_filename']+"_test.pkl")
    data_utils.save_data(file_path,embeddings)
    
