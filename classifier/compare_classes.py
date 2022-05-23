import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from final_dataset import SemanticSpaceDataset
import sys
sys.path.append('./dataPrep')
from data_utils import load_config

if __name__ == '__main__':

    # load data
    config = load_config()

    
    labels_path = config['DATASET']['SEMANTIC']['test_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']
    classes = config['DATASET']['classes']
    combinations = config['DATASET']['TEST_DIF_ZS']['combinations']
    test_paths = config['DATASET']['TEST_DIF_ZS']['test_ftrs']

    accs = []
    descrs =[]

    for i in range(len(combinations)):
        combo = combinations[i]

        data_path = test_paths[i]
        dataset = SemanticSpaceDataset(data_path, labels_path, word_embs_path)

        inds = []
        means = []
        str_descr = ""
        for cls in combo:
            str_descr += classes[cls] + "-"

            center = dataset.word_data[cls,:]
            means.append(center)

            inds1 = np.where(np.sum(dataset.word_emb_labels == center, axis=1) == center.shape[0])[0]
            inds.append(inds1)

        str_descr = str_descr[:-1]
        
        inds = np.array(inds).flatten()
        means = np.array(means)

        data = dataset.data[inds]
        labels = dataset.labels[inds]

        # get word embs
        unsup_dists = spatial.distance.cdist(data, means, metric='euclidean')
        unsup_best = np.argmin(unsup_dists,1).astype(int)

        combo = np.array(combo)
        pred = combo[unsup_best]

        accs.append((np.sum(np.equal(pred,labels))/len(pred)).item())
        descrs.append(str_descr)
        #print(str_descr, np.sum(np.equal(pred,labels))/len(pred))

    
    plt.figure(figsize=(10,4))
    plt.bar(descrs,accs,width=0.6)
    xlocs = [i for i in range(len(accs))]
    for i, v in enumerate(accs):
        plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.ylabel('tacnost')
    plt.show()