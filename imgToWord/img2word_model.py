import torch.nn as nn

class Img2WordModel(nn.Module):

    def __init__(self, in_features, out_features, hidden_size):
        super(Img2WordModel, self).__init__()

        self.layer1 = nn.Linear(in_features, hidden_size)
        self.activation = nn.Tanh()
        self.layer2 = nn.Linear(hidden_size, out_features)

 
    def forward(self, X):
        y = self.layer1(X)
        y = self.activation(y)
        y = self.layer2(y)
        return y