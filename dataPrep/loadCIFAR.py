import json
import numpy as np
import pickle


CONFIG_FILE_NAME = 'config.json'
 
if __name__ == '__main__':
    # read configuration
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)

    # load cifar training images - merge batches
    batch_size = config['DATASET']['imgs_batch_size']
    batch_num = config['DATASET']['imgs_batch_num']

    data = np.zeros((batch_size*batch_num,3*config['DATASET']['imgs_size']**2))
    labels = np.zeros((batch_size*batch_num,))

    for i in range(1,batch_num+1):
        file_name = config['DATASET']['imgs_batches_path']+"\\data_batch_"+str(i)   
        with open(file_name, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            data[(i-1)*batch_size:i*batch_size,:] = data_dict[b'data']
            labels[(i-1)*batch_size:i*batch_size] = data_dict[b'labels']

    # save whole training set
    file_name = config['DATASET']['imgs_path']+"\\train_data.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    file_name = config['DATASET']['imgs_path']+"\\train_labels.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(labels, f)
    


     

     