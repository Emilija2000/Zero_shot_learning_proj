import torch
import torch.nn as nn
import torch.nn.functional as F

class OmpT(nn.Module):

    def __init__(self, out_channels=1600, out_img_size = 27, pretrained=False, weights=[], alpha=0.25, pool_param=8):
        super(OmpT, self).__init__()

        # self.conv = nn.Conv2d(in_channels, out_channels, patch_size, stride=stride, padding=padding, bias=False)
        # note: using conv does not allow individual patch preprocessing

        # load pretrained dictionary
        if pretrained:
            if len(weights)>0:
                self.weights = torch.from_numpy(weights).T.float()
                self.weights = self.weights.reshape((1,1,self.weights.shape[0],self.weights.shape[1]))
            else:
                ('Please provide weights of the pretrained network...')
        self.shape = [-1, out_channels, out_img_size, out_img_size]
        
        # soft threshold value
        self.activation = nn.Softshrink(alpha)

        # pooling layer
        self.pool = nn.AvgPool2d(kernel_size=pool_param, stride=pool_param)
        
 
    def forward(self, X):
        # apply dictionary on image patches
        X = torch.matmul(X,self.weights)

        # reshape into [out_channels, out_img_size, out_img_size]
        X = X.reshape(self.shape)

        # soft threshold
        X = self.activation(X)

        # split positive and negative features
        ftrs_pos = X * (X>=0)
        ftrs_neg = X * (X<0)

        # average pooling
        ftrs_pos = self.pool(ftrs_pos)
        ftrs_neg = self.pool(ftrs_neg)

        # final embedding - return flattened vector
        if(len(X.shape)==3):
            emb = torch.cat((ftrs_pos.flatten(), ftrs_neg.flatten()))
        else:
            emb = torch.cat((ftrs_pos.reshape(ftrs_pos.shape[0],-1),ftrs_neg.reshape(ftrs_neg.shape[0],-1)), 1)
        return emb

