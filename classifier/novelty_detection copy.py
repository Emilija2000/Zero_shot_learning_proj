import numpy as np
import torch
import torch.nn as nn

class LoOP(nn.Module):

    def __init__(self, train_points, k, Lambda, sample_size=1000):
        super(LoOP, self).__init__()

        self.train_points = torch.from_numpy(train_points)
        self.k = k
        self.Lambda = Lambda

        self.Z = 1
        self.sample_size = sample_size

    def forward(self,X):
        loop = max(0, torch.erf(self.lof(X)/self.Z/torch.sqrt(torch.Tensor([2]))).item()) # mozda /torch.sqrt(torch.Tensor([2]) mozda ne
        return loop

    def train(self, random_sample=True):
        if random_sample:
            self.Z = self.calc_Z(self.sample(self.sample_size))
        else: 
            self.Z = self.calc_Z(self.train_points)

    def distance(self,X,Y):
        return torch.norm(Y-X, dim=1, p=2)
    
    def knn(self,X):
        dist = self.distance(self.train_points, X)
        knn_dists, indices = dist.topk(self.k, largest=False)
        return self.train_points[indices]

    def pdist(self,X,neighb):
        dists = torch.square(self.distance(neighb, X))
        return self.Lambda * torch.sqrt(torch.sum(dists)/neighb.shape[0])

    def lof(self, X):        
        points = self.knn(X)
        pdist = self.pdist(X, points)

        norm_factor = 0
        
        for point in points:
            norm_factor = norm_factor + self.pdist(point, self.knn(point))
        
        lof = pdist/norm_factor*points.shape[0] - 1
        return lof

    def sample(self, sample_size):
        ind = torch.randint(0, self.train_points.shape[0], size=(sample_size,))
        return self.train_points[ind,:]

    def calc_Z(self, sample):
        sum = torch.zeros((1,))
        i = 0
        for point in sample:
            if(i%50==0):
                print(i,"/",sample.shape[0],": ",sum.item())
            i += 1
            sum += torch.square(self.lof(point))
        Z = sum / sample.shape[0]
        Z = self.Lambda * torch.sqrt(Z)
        return Z


from final_dataset import SemanticSpaceDataset
import sys
sys.path.append('./dataPrep')
from data_utils import load_config


if __name__=='__main__':

    config = load_config()

    torch.manual_seed(config['CLASSIFIER']['random_seed']) 

    # load dataset
    train_data_path = config['DATASET']['SEMANTIC']['train_ftrs_path']
    train_labels_path = config['DATASET']['SEMANTIC']['train_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']

    dataset = SemanticSpaceDataset(train_data_path, train_labels_path, word_embs_path)

    k = config['CLASSIFIER']['uns_k']
    Lambda = config['CLASSIFIER']['uns_lambda']
    sample_size = config['CLASSIFIER']['uns_sample_size']
    
    # train novelty classifier
    uns_clf = LoOP(dataset.data, k, Lambda,sample_size)
    uns_clf.train()

    # load pretrained
    uns_clf = torch.load(config['CLASSIFIER']['uns_model_path'])

    # save clf
    #file_name = config['CLASSIFIER']['uns_model_path']
    #torch.save(uns_clf, file_name)
    #torch.save(uns_clf, file_name)

    # mini test 
    test_data_path = config['DATASET']['SEMANTIC']['test_ftrs_path']
    test_labels_path = config['DATASET']['SEMANTIC']['test_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']
    dataset = SemanticSpaceDataset(test_data_path, test_labels_path, word_embs_path)

    for i in range(30):
        print(dataset.get_label(dataset[i][1]),uns_clf(dataset[i][0]))
    