import json
import pickle
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

    out_img_size = 1+(img_size-patch_size+2*padding)//stride

    # load pretrained model parameters 
    file_path = config['DATASET']['imgs_path']+"\\dict.pkl"
    with open(file_path, 'rb') as f:
        weights,transf_m,transf_wh = pickle.load(f, encoding='bytes')       


    # init model
    model = OmpT(out_channels=out_channels,
        out_img_size=out_img_size,
        pretrained=True,
        weights = weights,
        alpha=alpha,
        pool_param=pooling)


    # read training data
    file_path_data = config['DATASET']['imgs_path']+"\\train_data.pkl"
    file_path_labels = config['DATASET']['imgs_path']+"\\train_labels.pkl"
    
    b_size=4 #RAM limit
    dataset = ImageDataset(file_path_data, file_path_labels, transf_wh, transf_m, in_channels, img_size, patch_size, stride, padding)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=False)

    embeddings = []

    for j in range(5):
        curr_data_batch = j
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
        