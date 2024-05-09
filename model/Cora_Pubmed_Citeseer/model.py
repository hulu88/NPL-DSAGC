import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import MixedDropout, MixedLinear,GATLayer
from torch.nn import Linear
import math
import scipy.sparse as sp
import numpy as np
import utils
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class AdaGCN(nn.Module):
    def __init__(self, nfeat,  nhid, nclass, dropout, dropout_adj):
        super().__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False)]
        fcs.append(nn.Linear(nhid, nclass, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if dropout is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout) 
        if dropout_adj is 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj) 
        self.act_fn = nn.ReLU()

    def _transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, x, adj, idx):  
        logits = self._transform_features(x) 
        return F.log_softmax(logits, dim=-1)[idx]

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)#spmm-->mm
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):#打印出维度的变化
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = torch.relu(self.gc1(x, adj))
        x = self.dropout(x)
        x = self.gc2(x, adj)
        return torch.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def getEmbedding(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)
        return z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred


class DNN2(nn.Module):  # 3327->500->32->6
    def __init__(self, num_features, hidden_size1, hidden_size2, nclass, dropout=0.2):
        super(DNN2, self).__init__()
        self.num_features = num_features
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.nclass = nclass

        self.a = DNNLayer(num_features, hidden_size1)
        self.b = DNNLayer(hidden_size1, hidden_size2)
        self.c = DNNLayer(hidden_size2, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        H = F.relu(self.a(x))
        H = self.dropout(H)
        H = F.relu(self.b(H))
        H = F.relu(self.scale(self.c(H)))

        return torch.log_softmax(H, dim=1)

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
class DNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)#1.414

    def forward(self, x):
        Wh = torch.mm(x, self.W)
        return Wh

    def __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+'->'+str(self.out_features)+')'

class DNN3(nn.Module):
    def __init__(self, num_features, hidden_size, nclass, dropout=0.5):
        super(DNN3, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.nclass = nclass

        self.a = DNNLayer(num_features, hidden_size)
        self.b = DNNLayer(hidden_size, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x,norm='True'):
        H = F.relu(self.a(x))
        H = self.dropout(H)
        if norm=='True':
            H = self.b(H)
            H = self.scale(H)
        else:
            H = self.b(H)
            if self.check(H)==1:
                H=self.repair(H)
            H = self.scale(H)
        return torch.log_softmax(H, dim=1)

    def check(self,z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        idx = (zmax == zmin).cpu().detach().numpy().flatten()
        if idx.sum()!=0:
            return 1
        return 0
    def repair(self,z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        idx = (zmax == zmin).squeeze()
        idx1=[]
        for i in range(idx.shape[0]):
            if idx[i]==True:
                idx1.append(i)
        for j in idx1:
            z[j]=torch.rand(1,z.shape[1])
        return z


    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

class DNN4(nn.Module):
    def __init__(self, num_features, hidden_size, nclass,dropout=0.5):
        super(DNN4, self).__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.nclass = nclass

        self.a = DNNLayer(num_features, hidden_size)
        self.b = DNNLayer(hidden_size, nclass)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x,adj,norm='True'):
        H = F.relu(self.a(x))
        H = self.dropout(H)
        H=torch.mm(adj,H)
        if norm=='True':
            H = self.scale(self.b(H))
        else:
            H = self.b(H)
        return torch.log_softmax(H, dim=1)


    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
