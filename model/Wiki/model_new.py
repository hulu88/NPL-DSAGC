import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class GalaLayer(nn.Module):
    def __init__(self,in_features,out_features,bias=False):
        super(GalaLayer,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.gain=nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.weight.data,gain=self.gain)
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj,multi=False,num=2):
        support = torch.mm(x, self.weight)
        if multi==True:
            temp=torch.eye(adj.shape[0])
            for i in range(num):
                temp=torch.mm(adj,temp)
            output=torch.mm(temp,support)
        else:
            output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GaLa(nn.Module):
    def __init__(self,nfeat,nhid,embed,dropout,alpha=0.2):
        super(GaLa,self).__init__()
        self.encoder1=GalaLayer(nfeat,nhid)
        self.encoder2=GalaLayer(nhid,embed)

        self.dropout=nn.Dropout(p=dropout)
       
        self.decoder1=GalaLayer(embed,nhid)
        self.decoder2=GalaLayer(nhid,nfeat)
       
        self.leakyrelu=nn.LeakyReLU(alpha)

    def forward(self,x,asm,asp):
        h1=self.leakyrelu(self.encoder1(x,asm))
        h1=self.dropout(h1)
        z=self.encoder2(h1,asm)
        h2=self.decoder1(z,asp)
        x=self.leakyrelu(self.decoder2(h2,asp))
        x=self.dropout(x)
        return z,x,h1,h2
    
    def getEmbedding(self, x, asm):
        h=self.encoder1(x,asm)
        h=self.encoder2(h,asm)
        z=F.normalize(h,p=2,dim=1)
        return z


