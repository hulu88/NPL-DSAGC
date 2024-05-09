import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.preprocessing import normalize
import sklearn.preprocessing as preprocess

def load_wiki():
    f = open('dataset/Wiki/graph.txt','r')
    adj, xind, yind = [], [], []
    for line in f.readlines():
        line = line.split()
        
        xind.append(int(line[0]))
        yind.append(int(line[1]))
        adj.append([int(line[0]), int(line[1])])
    f.close()

    f = open('dataset/Wiki/group.txt','r')
    label = []
    for line in f.readlines():
        line = line.split()
        label.append(int(line[1]))
    f.close()

    f = open('dataset/Wiki/tfidf.txt','r')
    fea_idx = []
    fea = []
    adj = np.array(adj)
    adj = np.vstack((adj, adj[:,[1,0]]))
    adj = np.unique(adj, axis=0)
    
    labelset = np.unique(label)
    labeldict = dict(zip(labelset, range(len(labelset))))
    label = np.array([labeldict[x] for x in label])
    adj = sp.csr_matrix((np.ones(len(adj)), (adj[:,0], adj[:,1])), shape=(len(label), len(label)))

    for line in f.readlines():
        line = line.split()
        fea_idx.append([int(line[0]), int(line[1])])
        fea.append(float(line[2]))
    f.close()

    fea_idx = np.array(fea_idx)
    features = sp.csr_matrix((fea, (fea_idx[:,0], fea_idx[:,1])), shape=(len(label), 4973)).toarray()
    scaler = preprocess.MinMaxScaler()
    
    features = scaler.fit_transform(features)
    features = torch.FloatTensor(features)
    
    label=torch.from_numpy(label).to(dtype=torch.float)
    
    adj1=adj.todense()
    adj1+=torch.eye(adj1.shape[0])
    adj1=normalize(adj,norm="l1")
    adj1=torch.from_numpy(adj1.toarray()).to(dtype=torch.float)
    
    adj_label= torch.from_numpy(adj.toarray()).to(dtype=torch.float)
    
    return adj1,adj_label, features, label




def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


