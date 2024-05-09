import os
import argparse
from sklearn.cluster import KMeans,SpectralClustering
import copy
from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import TSNE
import utils
from model_new import GaLa
from evaluation import eva

def get_ind(y,ci):
    data=[]
    for j in range(y.shape[0]):
        if y[j]==ci:
            data.append(j)
    return data

def target_adj(D,adj,n,t):
    d=copy.copy(D)
    A=adj
    others=list(range(n))
    for i in range(n):
        if i in d:
            others.remove(i)
    for j in d:
        A[j][d]=adj[j][d]*(1+t)
        A[j][others]=adj[j][others]*(1-t)
    return A
def KnnGraph(k,n,adj):
    A=torch.zeros(n,n)
    for i in range(n):
        temp=adj[i]*(-1)
        ind=np.argpartition(temp,k)[:k]
        A[i][ind]=1
    A=(A+A.T)/2
    return A

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
class IGALA(nn.Module):
    def __init__(self,input_dim,hidden_size,embedding_size,num_clusters,dropout=0,v=1):
        super(IGALA,self).__init__()
        self.num_clusters=num_clusters
        self.gala=GaLa(input_dim,hidden_size,embedding_size,dropout)

    def forward(self,x,asm,asp):
        z,X,_,_=self.gala(x,asm,asp)
        return X,z
      

def trainer(dataset):
    model=IGALA(
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_clusters=args.n_clusters
    ).to(device)
    print(model)

    optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)    

    dataset=utils.data_preprocessing(dataset)
    adj=dataset.adj.to(device)
    adj_label=dataset.adj_label
    adj_label=adj_label-torch.eye(dataset.x.shape[0])

    asm=utils.SMnormalize_adj(adj_label)
    asp=utils.SPnormalize_adj(adj_label)
    asm=utils.sparse_matrix_to_torch(asm).to(device)
    asp=utils.sparse_matrix_to_torch(asp).to(device)

    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    result = [0, 0, 0, 0]
    
    model.gala.load_state_dict(torch.load(args.pretrain_path,map_location='cpu'))
    
    with torch.no_grad():
        z,_,_,_=model.gala(x,asm,asp)

    z=F.normalize(z,p=2,dim=1)
    f_adj=torch.softmax(torch.mm(z,z.t()),dim=-1)          
    f_adj=KnnGraph(args.k,f_adj.shape[0],f_adj.detach().numpy())
    
    cluster=SpectralClustering(n_clusters=args.n_clusters,affinity='precomputed',random_state=0)
    y_pred=cluster.fit_predict(f_adj.detach().numpy())
    eva(y,y_pred,'pretrain',show=True)

    for epoch in range(args.max_epoch):
        model.train()
        if epoch%args.update_interval==0:
            _,z=model(x,asm,asp)
            z=F.normalize(z,p=2,dim=1)
            f_adj=torch.softmax(torch.mm(z,z.t()),dim=-1)
            A=KnnGraph(args.k,f_adj.shape[0],f_adj.detach().numpy())

            cluster=SpectralClustering(n_clusters=args.n_clusters,affinity='precomputed',random_state=0)
            y_pred=cluster.fit_predict(A.detach().numpy())
            with torch.no_grad():
                for i in range(args.n_clusters):
                    data=get_ind(y_pred,i)
                    f_adj=target_adj(data,f_adj,f_adj.shape[0],0.5)
            f_adj=((f_adj.t()/torch.sum(f_adj,1)).t()).detach()

        X,z=model(x,asm,asp)
        z=F.normalize(z,p=2,dim=1)
        a0=torch.softmax(torch.mm(z,z.t()),dim=-1)
        adj=KnnGraph(args.k,a0.shape[0],a0.detach().numpy()).to(device)

        cluster=SpectralClustering(n_clusters=args.n_clusters,affinity='precomputed',random_state=0)     
        y_pred=cluster.fit_predict(adj.detach().numpy())
        acc,nmi,ari,f1=eva(y,y_pred,epoch)
        
        se_loss=F.kl_div(a0.log(),f_adj,reduction='batchmean')
        re_loss=0.5*(torch.square((X.view(-1))-(x.view(-1))).sum())/(X.shape[0])
        loss =5*se_loss + re_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if result[0]<acc:
            result=[acc,nmi,ari,f1]

    print("Result:\nacc:{}; nmi:{}; ari:{}; f1:{}".format(result[0], result[1], result[2], result[3]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=5, type=int)  
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
        args.lr = 0.005
        args.hidden_size=500
        args.embedding_size=32
        args.k = 20
        args.n_clusters = 6
        args.max_epoch=50
    elif args.name == 'Cora':
        args.lr = 0.005
        args.hidden_size=500
        args.embedding_size=32
        args.max_epoch=50
        args.k = 100
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None
 
    args.pretrain_path =f'pretrain/GALA_{args.name}.pkl'
    args.input_dim = dataset.num_features

    print(args)
    trainer(dataset)





























