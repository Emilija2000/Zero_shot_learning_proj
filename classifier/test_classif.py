import numpy as np
from scipy import spatial
import torch
from torch.utils.data import DataLoader

import sys
import os
sys.path.append('./dataPrep')
from data_utils import load_config,save_data

from final_dataset import AllFtrsDataset
from supervised_model import SupervisedModel
from novelty_detection import LoOP, Gaussian, OneClsSVM


if __name__=='__main__':

    config = load_config()
    classes = config['DATASET']['classes']
    unseen = config['DATASET']['unseen']
    unseen_inds = np.where(np.in1d(classes,unseen))[0]

    torch.manual_seed(config['CLASSIFIER']['random_seed'])

    # load dataset
    img_ftrs_data_path = config['DATASET']['IMAGES']['imgs_10cls_ftrs_test_path'] 
    semantic_ftrs_data_path = config['DATASET']['SEMANTIC']['test_ftrs_path']
    test_labels_path = os.path.join(config['DATASET']['IMAGES']["imgs_path"],"test_labels.pkl")
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    dataset = AllFtrsDataset(img_ftrs_data_path, semantic_ftrs_data_path, test_labels_path, word_embs_path, unseen_inds)
    
    batch_size = config['CLASSIFIER']['batch_size']
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # initialize the models
    in_features = config['IMAGE_EMB']['ftrs_size']
    out_features = len(classes) - len(unseen) 
    model_sup = SupervisedModel(in_features,out_features)
    file_name = config['CLASSIFIER']['model_path']
    model_sup.load_state_dict(torch.load(file_name))
    model_sup.eval()

    file_name = config['CLASSIFIER']['uns_model_path']
    model_uns = torch.load(file_name)

    uns_means = dataset.word_data[np.array(unseen_inds)]

    # test for different thresholds for unsupervised classification
    thresh_num = config['CLASSIFIER']['thresholds_num']
    thresholds_orig = np.linspace(0,1,thresh_num)

    #loop
    #thresholds = thresholds_orig
    # gaussian 
    #thresholds = model_uns.find_thresholds(torch.from_numpy(dataset.semantic), thresholds_orig)
    # svm
    thresholds = model_uns.find_thresholds(dataset.semantic, thresholds_orig[::-1])

    acc_all = []
    acc_sup = []
    acc_uns = []

    #loop - actual percentage
    perc_new = []

    for thresh in thresholds:
        correct = 0.0
        all = 0.0
        correct_sup = 0.0
        all_sup = 0.0
        correct_uns = 0.0
        all_uns = 0.0
        predictions = []  
        
        num_new = 0

        for i,(data_img, data_word, labels) in enumerate(dataloader):
            with torch.no_grad():
                # LOop
                #newcls = model_uns(data_word)>=thresh
                # Gauss or SVM: reversed
                newcls = model_uns(data_word)<thresh

                #Loop or Gauss
                #newcls = newcls.numpy()

                num_new += np.sum(newcls).item()

                unsup_dists = spatial.distance.cdist(data_word, uns_means, metric='euclidean')
                unsup_best = np.argmin(unsup_dists,1)

                sup_out = model_sup(data_img)
                sup_best = np.argmax(sup_out,1)
                sup_best = dataset.get_label(sup_best)

                preds = unseen_inds[unsup_best]*newcls + sup_best*(1-newcls)
                
                labels = labels.numpy()
                correct += np.sum(np.equal(preds, labels))
                all += labels.shape[0]
                predictions = predictions + preds.tolist()

                # divided supervised and novelty classes
                uns_inds = np.in1d(labels,unseen_inds)
                sup_inds = np.logical_xor(True, uns_inds)
                correct_sup += np.sum(np.equal(preds[sup_inds], labels[sup_inds]))
                correct_uns += np.sum(np.equal(preds[uns_inds], labels[uns_inds]))
                all_sup += np.sum(sup_inds)
                all_uns += np.sum(uns_inds)
                
        
        perc_new.append(num_new/dataset.__len__())
        print(perc_new[-1])
        
        acc_all.append(correct/all)
        acc_sup.append(correct_sup/all_sup)
        acc_uns.append(correct_uns/all_uns)
        print('Thresh',thresh,': ',acc_all[-1], acc_sup[-1], acc_uns[-1])

    save_data('svm_accs.pkl', (acc_all,acc_sup,acc_uns))
    #save_data('loop_perc.pkl',perc_new)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(thresholds_orig,acc_sup,'bo')
    plt.plot(thresholds_orig,acc_uns,'rx')
    plt.plot(thresholds_orig,acc_all,'g*')
    #plt.xlabel('granica prepoznavanja novih klasa')
    plt.xlabel('procenat podataka klasifikovanih u poznate klase')
    plt.ylabel('tacnost')
    plt.legend(['poznate klase','nove klase','ukupna tacnost'])
    plt.show()

        
    
    

    