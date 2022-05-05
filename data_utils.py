import json
import numpy as np
import os
import pickle

CONFIG_FILE_NAME = 'config.json'

def load_config():
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)
    return config

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_image_embeddings(img_emb_path, img_emb_filename, img_batch_num):
    x = []
    for i in range(1,img_batch_num+1):
        file_path = os.path.join(img_emb_path, img_emb_filename+str(i)+".pkl")
        with open(file_path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            if(len(x)==0):
                x = d
            else:
                x = np.concatenate((x,d))
    x = np.float64(x)
    return x
