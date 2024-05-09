import argparse
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans,SpectralClustering

import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scipy.sparse.csgraph import connected_components

from sklearn.manifold import TSNE

import utils
from model import GAT
from model_new import GaLa
from evaluation import eva
from WikiUtil import load_wiki
import time
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
def scale(z):
    zmax = np.amax(z,axis=1)
    zmin = np.amin(z,axis=1)
    zmax = np.expand_dims(zmax,axis=1)
    zmin = np.expand_dims(zmin,axis=1)
    z_std = (z - zmin) / (zmax - zmin+0.01)
    return z_std
def check(z):
    zmax = np.amax(z,axis=1)
    zmin = np.amin(z,axis=1)
    if (zmax == zmin).sum()!=0:
        return 1
    return 0
def repair(z):
    zmax = np.amax(z,axis=1)
    zmin = np.amin(z,axis=1)
    idx = (zmax == zmin)
    idx1=[]
    for i in range(idx.shape[0]):
        if idx[i]==True:
            idx1.append(i)
    for j in idx1:
        z[j]=np.random.rand(1,z.shape[1])
    return z
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def KnnGraph(k,n,adj):
    A=torch.zeros(n,n)
    test=adj
    for i in range(n):
        temp=test[i]*(-1)
        ind=np.argpartition(temp,k)[:k]
        A[i][ind]=1
    A=(A+A.T)/2
    return A

def KnnGraph1(k,n,adj):
    A=np.zeros((n,n))
    test=adj
    for i in range(n):
        temp=test[i]*(-1)
        ind=np.argpartition(temp,k)[:k]
        A[i][ind]=test[i][ind]
    A=(A+A.T)/2
    return A

def gussian_simility(n,z):
    A=torch.zeros(n,n)
    test=z
    for i in range(n):
        for j in range(n):
            if i!=j:
                A[i][j]=torch.exp((-1)*torch.square(test[i]-test[j]).sum()/2)
    return A

def pretrain(x, y, A, adj_label):
    res=[]
    for i in range(1):
        model = GaLa(
            nfeat=args.input_dim,
            nhid=args.hidden_size,
            embed=args.embedding_size,
            dropout=0,           
        ).to(device)
        print(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        adj_label = adj_label
        adj_label = adj_label-torch.eye(x.shape[0])

        asm=utils.SMnormalize_adj(adj_label)
        asp=utils.SPnormalize_adj(adj_label)
        asm=utils.sparse_matrix_to_torch(asm).to(device)
        asp=utils.sparse_matrix_to_torch(asp).to(device)

        x = torch.Tensor(x).to(device)
        y = y.numpy()
        result = [0, 0, 0, 0]

        for epoch in range(args.max_epoch):
            model.train()
            z,X,_,_ = model(x, asm,asp)
            loss = 0.5*(torch.square((X.view(-1))-(x.view(-1))).sum())/(X.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                z,_,_,_ = model(x, asm,asp)
                z=F.normalize(z,p=2,dim=1)

                A=kneighbors_graph(z.detach().cpu(),args.k,mode='connectivity',include_self=False)
                A=A.toarray()
                f_adj=0.5*(A+A.T)                     

                cluster=SpectralClustering(n_clusters=args.n_clusters,affinity='precomputed',random_state=0)
                y_pred=cluster.fit_predict(f_adj)
                acc, nmi, ari, f1 = eva(y,y_pred, epoch)

                v_loss.append(acc)
                if result[0] < acc:
                    result = [acc, nmi, ari, f1]
        print("Result:\nacc:{}; nmi:{}; ari:{}; f1:{}".format(result[0], result[1], result[2], result[3]))
        res.append(result[0])
    print("avg acc{}".format(np.array(res).sum()/1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="Wiki")  
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    args = parser.parse_args()

    if os.path.exists('pretrain/') == False:  
        os.makedirs('pretrain/')

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.name == "Wiki":
        args.input_dim = 4973
        args.max_epoch = 200
        args.lr = 0.001
        args.k = 20
        args.n_clusters = 17
        A,adj_label,x,y = load_wiki()
        print(args)
        pretrain(x, y, A, adj_label)
        exit()
    else:
        args.k = None
        

