import json
import numpy as np
import pickle

CONFIG_FILE_NAME = 'config.json'
np.random.seed(43)
 
if __name__ == '__main__':
    # read configuration
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)


    # read training data
    print('Preparing training images...')
    file_name = config['DATASET']['imgs_path']+"\\train_data.pkl"
    with open(file_name, 'rb') as f:
        x_train = pickle.load(f, encoding='bytes')


    # extract random image patches for dictionary training
    print('Preparing dictionary training patches...')
    num_patches = config['IMAGE_EMB']['encoders']['omp1-t']['num_training_patches']
    patch_size = config['IMAGE_EMB']['encoders']['omp1-t']['patch_size']
    img_size = config['DATASET']['imgs_size']
    
    p_imgs = np.random.randint(0, x_train.shape[0], num_patches)
    r = np.random.randint(0, img_size-patch_size+1, num_patches)
    c = np.random.randint(0, img_size-patch_size+1, num_patches)
    
    imgs = x_train[p_imgs,:].reshape((num_patches, img_size, img_size, 3))
    patches = np.zeros((num_patches,patch_size*patch_size*3))
    for i in range(num_patches):
        patch = imgs[i,r[i]:r[i]+patch_size,c[i]:c[i]+patch_size,:]
        patches[i,:] = patch.flatten()
        
    # normalization
    std = np.std(patches,1).reshape(num_patches,1)
    std[std==0] = 1
    patches = np.divide((patches - np.mean(patches,1).reshape(num_patches,1)),std)

    # zca whitening
    s = np.cov(patches.T)
    m = np.mean(patches.T)
    ph,l,_ = np.linalg.svd(s)
    transf = np.dot(ph, np.dot(np.diag(1.0/np.sqrt(l + 1e-10)), ph.T))
    patches = np.dot(patches, transf)


    # initialize trining parameters
    print('Initializing dictionary training...')
    num_iter = config['IMAGE_EMB']['encoders']['omp1-t']['num_iter']  
    ftrs_size = config['IMAGE_EMB']['encoders']['omp1-t']['ftrs_size']  
    batch_size = config['IMAGE_EMB']['encoders']['omp1-t']['batch_size']  
    num_batches = int(np.ceil(num_patches/(1.0*batch_size)))

    dictionary = np.random.randn(patch_size*patch_size*3,ftrs_size) #random init

    for i in range(num_iter):
        # normalize
        dictionary = np.divide(dictionary,np.std(dictionary,0)+1e-10)

        dict_temp = np.zeros(dictionary.shape)

        for j in range(num_batches):

            # get training samples from batch j 
            data = patches[j*batch_size:min(num_patches, (j+1)*batch_size),:]
            
            # find the most correlated component - omp1
            corr = np.matmul(dictionary.T,data.T) # correlation of dict columns with data vectors - ftrs_size*batch_size
            k = np.argmax(corr, 0) # vector (batch_size,)
            # sparse vectors 
            x_sparse = np.zeros((ftrs_size,batch_size))
            x_sparse[k,np.arange(batch_size)] = corr[k,np.arange(batch_size)]
            
            # update dictionary
            dict_temp = dict_temp + np.matmul(x_sparse,data).T
        dictionary = dict_temp

        if((i+1)%5==0):
            print('Iteration {}/{}'.format(i+1,num_iter))


    # save dict
    print('Saving image dictionary...')
    file_name = config['DATASET']['imgs_path']+"\\dict.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(dictionary, f)
