import torch.nn as nn

class SupervisedModel(nn.Module):

    def __init__(self, in_features, out_features):
        super(SupervisedModel, self).__init__()

        self.layer1 = nn.Linear(in_features, out_features)
        #self.softmax = nn.Softmax()  #included in cross entropy loss pytorch

    def forward(self, X):
        y = self.layer1(X)
        #y = self.softmax(y)
        return y