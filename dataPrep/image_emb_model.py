from numpy import pad
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

class OmpT(nn.Module):

    def __init__(self, in_channels=3, out_channels=1600, patch_size=6, pretrained=False, weights_path=None, alpha=0.25, stride=1, padding=0, pool_param=8):
        super(OmpT, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, patch_size, stride=stride, padding=padding, bias=False)

        # load pretrained dictionary
        if pretrained:
            if weights_path!=None:
                with open(weights_path, 'rb') as f:
                    weights = pickle.load(f, encoding='bytes')
                weights = torch.from_numpy(weights.T)
                weights = weights.reshape([out_channels,patch_size,patch_size,in_channels])
                weights = weights.permute(0,3,1,2)

                with torch.no_grad():
                    self.conv.weight = nn.Parameter(weights)
            else:
                ('Please provide weights of the pretrained network...')

        # soft threshold value
        self.activation = nn.Softshrink(alpha)

        # pooling layer
        self.pool = nn.AvgPool2d(kernel_size=pool_param, stride=pool_param)
        
 
    def forward(self, X):
        # apply dictionary on image patches
        X = self.conv(X) 
        # soft threshold
        X = self.activation(X)
        # split positive and negative features
        ftrs_pos = torch.abs(X)
        ftrs_neg = -torch.abs(-X)
        # average pooling
        ftrs_pos = self.pool(ftrs_pos)
        ftrs_neg = self.pool(ftrs_neg)
        # final embedding
        if(len(X.shape)==3):
            emb = torch.cat((ftrs_pos.flatten(), ftrs_neg.flatten()))
        else:
            emb = torch.cat((ftrs_pos.reshape(ftrs_pos.shape[0],-1),ftrs_neg.reshape(ftrs_neg.shape[0],-1)), 1)
        return emb

