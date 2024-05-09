import argparse
import os
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans,SpectralClustering

import random
import torch
import torch.nn.functional as F
from torch.optim import Adam


import utils

from model_new import GaLa
from evaluation import eva
import scipy.sparse as sp
import numpy as np
import time
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def KnnGraph(k,n,adj):
    A=torch.zeros(n,n)
    for i in range(n):
        temp=adj[i]*(-1)
        ind=np.argpartition(temp,k)[:k]
        A[i][ind]=1
    A=(A+A.T)/2

    return A



def pretrain(dataset):
    res=[]
    for i in range(10):
        model = GaLa(
            nfeat=args.input_dim,
            nhid=args.hidden_size,
            embed=args.embedding_size,
            dropout=0,
            
        ).to(device)
        print(model)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)#args.weight_decay

        
        dataset = utils.data_preprocessing(dataset)
        adj = dataset.adj.to(device)
        adj_label = dataset.adj_label

        adj_label = adj_label-torch.eye(dataset.x.shape[0])

        asm=utils.SMnormalize_adj(adj_label)
        asp=utils.SPnormalize_adj(adj_label)

        asm=utils.sparse_matrix_to_torch(asm).to(device)
        asp=utils.sparse_matrix_to_torch(asp).to(device)
        
        x = torch.Tensor(dataset.x).to(device)
        y = dataset.y.cpu().numpy()
        result = [0, 0, 0, 0]
        
        for epoch in range(args.max_epoch):

            model.train()
            _,X,_,_ = model(x, asm,asp)

            loss = 0.5*(torch.square((X.view(-1))-(x.view(-1))).sum())/(X.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

             with torch.no_grad():
                z,_,_,_ = model(x, asm,asp)
                z=F.normalize(z,p=2,dim=1)

                f_adj=torch.softmax(torch.mm(z,z.t()),dim=-1)          
                f_adj=KnnGraph(args.k,f_adj.shape[0],f_adj.detach().numpy())
                   
                cluster=SpectralClustering(n_clusters=args.n_clusters,affinity='precomputed',random_state=0)
                y_pred=cluster.fit_predict(f_adj)
                acc, nmi, ari, f1 = eva(y,y_pred, epoch)

                if result[0] < acc:
                    result = [acc, nmi, ari, f1]

        print("Result:\nacc:{}; nmi:{}; ari:{}; f1:{}".format(result[0], result[1], result[2], result[3]))
        res.append(result[0])

    print("avg acc{}".format(np.array(res).sum()/10))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default="Cora")  
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.2, help="Alpha for the leaky_relu.")
    args = parser.parse_args()


    if os.path.exists('pretrain/') == False:  #
        os.makedirs('pretrain/')

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.hidden_size=500
        args.k = 20
        args.n_clusters = 6
        args.embedding_size=32
        args.max_epoch=70
    elif args.name == "Cora":
        args.lr = 0.005
        args.hidden_size=500
        args.k = 100
        args.n_clusters = 7
        args.embedding_size=32
        args.max_epoch = 100
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = 800
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)



