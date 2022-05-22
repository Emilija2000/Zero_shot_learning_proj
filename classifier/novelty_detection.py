import numpy as np
import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        self.means  = torch.tensor(0.)
        self.sigmas = torch.tensor(1.)
        self.d = 1

    def forward(self, X):
        p = torch.zeros((X.shape[0],self.sigmas.shape[0]))
        
        sigmas = 1/self.sigmas
        sigmas = torch.multiply(sigmas.reshape((-1,1)),torch.ones(self.means.shape))
        sigmas = torch.diag_embed(sigmas)

        coefs = self.d*0.5*torch.log(2*torch.pi*self.sigmas)

        for i in range(len(self.sigmas)):
            mean = self.means[i,:]
            m = X - mean
            
            p_i = torch.matmul(m, sigmas[i,:,:])
            p_i = torch.sum(p_i * m,dim=1)
            
            lnp = -0.5*p_i - coefs[i]
            
            p[:,i] =lnp

        out = torch.max(p,dim=1)[0]
        
        return out

    def find_thresholds(self, data, percents):
        out = self.forward(data)
        out = torch.sort(out)[0]
        
        inds = percents * out.shape[0]

        return out[inds]

    def train(self, train_points, train_word_labels, random_sample=False,sample_size=5000):
        if random_sample:
            inds = np.random.randint(0, train_points.shape[0], size=(sample_size,))
            sample = train_points[inds,:]
            sample_labels = train_word_labels[inds]
        else:
            sample = train_points
            sample_labels = train_word_labels
        
        centers = np.unique(sample_labels, axis = 0)

        dists = sample - sample_labels
        self.d = centers.shape[1]
        dists = np.mean(dists**2,axis=1)
        
        means = []
        sigmas = []
        for center in centers:
            means.append(center)
            
            inds = np.sum(sample_labels == center, axis=1) == self.d
            center_cls = dists[inds]
            sigmas.append(np.mean(center_cls))

        self.means = torch.from_numpy(np.array(means))
        self.sigmas= torch.from_numpy(np.array(sigmas))


class LoOP(nn.Module):

    def __init__(self, train_points, k, Lambda, sample_size=1000):
        super(LoOP, self).__init__()

        self.train_points = torch.from_numpy(train_points)
        self.train_points = self.train_points.reshape((1,self.train_points.shape[0],self.train_points.shape[1]))
        self.k = k
        self.Lambda = Lambda

        self.Z = 1
        self.sample_size = sample_size

    def forward(self,X):
        X = X.reshape((X.shape[0],1,X.shape[1]))
        loop = torch.erf(self.lof(X)/self.Z/torch.sqrt(torch.Tensor([2]))) # mozda /torch.sqrt(torch.Tensor([2])) mozda ne
        loop = torch.max(torch.Tensor([0]),loop)
        return loop

    def train(self, random_sample=True):
        if random_sample:
            self.Z = self.calc_Z(self.sample(self.sample_size))
        else: 
            self.Z = self.calc_Z(self.train_points)

    def distance(self,X,Y):
        return torch.norm(Y-X, dim=2, p=2)
    
    def knn(self,X):
        dist = self.distance(self.train_points, X)
        knn_dists, indices = dist.topk(self.k, largest=False)
        return self.train_points[:,indices,:]

    def pdist(self,X,neighb):
        dists = torch.square(self.distance(neighb, X))
        return self.Lambda * torch.sqrt(torch.sum(dists,dim=1)/neighb.shape[1])

    def lof(self, X):      
        points = self.knn(X)[0]
        pdist = self.pdist(X, points)
    
        sh1 = points.shape[0]
        points = points.reshape((-1,1,points.shape[-1]))
        neighbs = self.knn(points)[0]
        pdists = self.pdist(points, neighbs)
        pdists = pdists.reshape((sh1,-1))
        norm_factors = torch.mean(pdists,1)
        
        lof = torch.divide(pdist,norm_factors) - 1
        
        return lof

    def sample(self, sample_size):
        ind = torch.randint(0, self.train_points.shape[1], size=(sample_size,))
        data = self.train_points[:,ind,:]
        data = data.reshape((data.shape[1],1,-1))
        return data

    def calc_Z(self, sample):
        sum = torch.zeros((1,))
        step=50
        for i in range(0,sample.shape[0],step):
            points = sample[i:i+step,:,:]
            
            lofs = self.lof(points)
            lofs =  torch.square(lofs)
            sum +=torch.sum(lofs)

            print(i,"/",sample.shape[0],": ",sum.item())
            
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
    #uns_clf = LoOP(dataset.data, k, Lambda,sample_size)
    #uns_clf.train()

    uns_clf = Gaussian()
    uns_clf.train(dataset.data,dataset.word_emb_labels)

    # load pretrained
    #uns_clf = torch.load(config['CLASSIFIER']['uns_model_path'])

    # save clf
    file_name = config['CLASSIFIER']['uns_model_path']
    torch.save(uns_clf, file_name)

    # mini test 
    test_data_path = config['DATASET']['SEMANTIC']['test_ftrs_path']
    test_labels_path = config['DATASET']['SEMANTIC']['test_labels_path']
    word_embs_path = config['DATASET']['WORDS']['word_ftrs_path']
    dataset = SemanticSpaceDataset(test_data_path, test_labels_path, word_embs_path)

    data = dataset[0:64][0]
    print(dataset.get_label(dataset[0:64][1]),uns_clf(data))

    X = torch.from_numpy(dataset.data)
    threshs = uns_clf.find_thresholds(X, np.arange(0,1,0.1))
    print(threshs)
    
    data = dataset[1][0]
    data = data.reshape((1,data.shape[0]))
    print(dataset.get_label(dataset[1][1]),uns_clf(data))
    