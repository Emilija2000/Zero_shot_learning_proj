import numpy as np
import os

import data_utils

#bio ovaj seed 1 u ovim trenutnim rezultatima
 
if __name__ == '__main__':
    # read configuration
    config = data_utils.load_config()

    np.random.seed(config['IMAGE_EMB']['random_seed'])

    # read training data
    print('Preparing training images...')
    file_name = os.path.join(config['DATASET']['IMAGES']['imgs_path'],"train_data.pkl")
    x_train = data_utils.load_data(file_name)

    # extract random image patches for dictionary training
    print('Preparing dictionary training patches...')
    num_patches = config['IMAGE_EMB']['encoders']['omp1-t']['num_training_patches']
    patch_size = config['IMAGE_EMB']['encoders']['omp1-t']['patch_size']
    img_size = config['DATASET']['IMAGES']['imgs_size']
    
    p_imgs = np.random.randint(0, x_train.shape[0], num_patches)
    r = np.random.randint(0, img_size-patch_size+1, num_patches)
    c = np.random.randint(0, img_size-patch_size+1, num_patches)
    
    imgs = x_train[p_imgs,:].reshape((num_patches, 3, img_size, img_size))

    patches = np.zeros((num_patches,patch_size*patch_size*3))
    for i in range(num_patches):
        patch = imgs[i,:,r[i]:r[i]+patch_size,c[i]:c[i]+patch_size]
        patches[i,:] = patch.flatten()
      
    # normalization
    std = np.std(patches,1).reshape(num_patches,1)
    std[std==0] = 1
    patches = np.divide((patches - np.mean(patches,1).reshape(num_patches,1)),std)

    # zca whitening
    s = np.cov(patches.T)
    m = np.mean(patches,0)
    l,ph = np.linalg.eig(s)
    transf = np.dot(ph, np.dot(np.diag(1.0/np.sqrt(l + 0.1)), ph.T))
    patches = np.dot(patches-m, transf)


    # initialize trining parameters
    print('Initializing dictionary training...')
    num_iter = config['IMAGE_EMB']['encoders']['omp1-t']['num_iter']  
    ftrs_size = config['IMAGE_EMB']['encoders']['omp1-t']['ftrs_size']  
    batch_size = config['IMAGE_EMB']['encoders']['omp1-t']['batch_size']  
    num_batches = int(np.ceil(num_patches/(1.0*batch_size)))

    # init dict
    dictionary = np.random.randn(ftrs_size,patch_size*patch_size*3) #random init
    # normalize dict
    dictionary = np.divide(dictionary,np.sqrt(np.sum(dictionary*dictionary,1)+1e-20).reshape(-1,1))
    

    for i in range(num_iter):
        dict_temp = np.zeros(dictionary.shape)

        for j in range(num_batches):

            # get training samples from batch j 
            data = patches[j*batch_size:min(num_patches, (j+1)*batch_size),:]
            
            # find the most correlated component - omp1
            corr = np.matmul(dictionary,data.T) # correlation of dict columns with data vectors - ftrs_size*batch_size
            k = np.argmax(np.abs(corr), 0) # vector (batch_size,)
            # sparse vectors 
            x_sparse = np.zeros((ftrs_size,batch_size))
            x_sparse[k,np.arange(batch_size)] = corr[k,np.arange(batch_size)]
            
            # update dictionary
            dict_temp = dict_temp + np.matmul(x_sparse,data)
        
        # reinit empty clusters
        ind = np.sum(dict_temp*dict_temp,1) < 0.001
        newinit = np.random.randn(ftrs_size,patch_size*patch_size*3)
        dict_temp[ind,:] = newinit[ind,:]

        dictionary = dict_temp
        dictionary = np.divide(dictionary,np.sqrt(np.sum(dictionary*dictionary,1)+1e-20).reshape(-1,1))

        if((i+1)%5==0):
            print('Iteration {}/{}'.format(i+1,num_iter))

    # save dict
    print('Saving image dictionary...')
    file_name = os.path.join(config['DATASET']['IMAGES']['imgs_path'],config['IMAGE_EMB']['dict_file_name'])
    data_utils.save_data(file_name, (dictionary,m,transf))
    