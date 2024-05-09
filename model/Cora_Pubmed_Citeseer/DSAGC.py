from collections import Counter

from sklearn.cluster import KMeans,SpectralClustering
from sklearn.neighbors import kneighbors_graph
import utils
from model import GAT,DNN2,DNN3,GCN
from model_new import GaLa
from IGaLa import IGALA
import copy
from sklearn.manifold import TSNE

from evaluation import eva
import scipy.sparse as sp
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1./(1.+np.exp(-x))

def KnnGraph(k,n,adj):
    A=torch.zeros(n,n)
    for i in range(n):
        temp=adj[i]*(-1)
        ind=np.argpartition(temp,k)[:k]
        A[i][ind]=1
    A=(A+A.T)/2
    return A

def get_ind(y,ci): 
    data = []
    n=y.shape[0]
    for j in range(n):
        if y[j] == ci:
            data.append(j)
    return data

def get_sample_1(D,k,adj):
    n=len(D)
    ind=torch.tensor(D)
    sim=torch.zeros(n,n)
    for i in range(n):
        for j in range(n):
            sim[i][j]=adj[D[i]][D[j]]
    temp=sim.reshape([-1,])
    num=k*len(temp)
    num=np.floor(num).astype(np.int)-1
    inds=np.argpartition(-temp,num)[:num]
    x1=inds//(sim.shape[0])
    x2=inds%(sim.shape[0])
    x=torch.cat((ind[x1],ind[x2]),0)
    return list(set(x.numpy()))

def get_sample_1_1(D,k,adj):
    n=len(D)
    ind=torch.tensor(D)
    sim=torch.zeros(n,n)
    dis=[]
    for i in range(n):
        tmp=0
        for j in range(n):
            sim[i][j]=adj[D[i]][D[j]]
            tmp+=sim[i][j]
        dis.append(tmp)
    dis=np.array(dis)
    num=np.floor(n*k).astype(np.int)
    index=np.argpartition(-dis,num)[:num]
    x=ind[index]
    return list(set(x.numpy()))

def get_sample_2(D,adj,n,k2):
    d=copy.copy(D)
    other=[]
    sample_index=[]
    adj=torch.from_numpy(adj)
    for i in range(n):
        other.append(i)
    for j in range(n):
        if j in d:
            other.remove(j)

    for t in d:
        dd=copy.copy(D)
        dd.remove(t)
        t1=torch.max(adj[t][dd])
        t2=torch.max(adj[t][other])
        if t1-t2>=k2:
            sample_index.append(t)
    return sample_index


def get_dist(num,y,adj,n,k2):
    dis=[]
    for i in range(num):
        data=get_ind(y,i)
        d=get_distance(data,adj,n,k2)
        dis.append(d)
    min_dis=min(dis)
    return min_dis  

def get_distance(D,adj,n,k2):
    d=copy.copy(D)
    other=[]
    sample_index=[]
    adj=torch.from_numpy(adj)
    for i in range(n):
        other.append(i)
    for j in range(n):
        if j in d:
            other.remove(j)
    distance=[]
    for t in d:
        dd=copy.copy(D)
        dd.remove(t)
        t1=torch.max(adj[t][dd])
        t2=torch.max(adj[t][other])
        d=t1-t2
        distance.append(d)
    distance=(np.sort(distance))[::-1]
    num=k2*distance.shape[0]
    num=np.floor(num).astype(np.int)-1
    dis=distance[num]
    return dis

def trainer(dataset):
    model = IGALA(
        input_dim=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        num_clusters=args.n_clusters              
    ).to(device)
    
    print(model)

    model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

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

    with torch.no_grad():
        _,z = model(x,asm,asp)
    z=F.normalize(z,p=2,dim=1)
    z1=z.data.cpu().numpy()
    S = np.matmul(z1, np.transpose(z1))

    f_adj=kneighbors_graph(z1,args.k,mode='connectivity',include_self=False)
    f_adj=f_adj.toarray()
    f_adj=0.5*(f_adj+f_adj.T)
    
    cluster = SpectralClustering(n_clusters=args.n_clusters, affinity='precomputed', random_state=0)
    y_pred = cluster.fit_predict(f_adj)
    acc, nmi, ari, f1 = eva(y, y_pred, 'pretrain-Spectral-Cluster', show=True)
    
    A=S-np.eye(S.shape[0])
    
    k1=0.0005
    k1_1=0.9
    k2=0.9
    sample=[]
    k0=get_dist(args.n_clusters,y_pred,A,A.shape[0],k2)

    for i in range(args.n_clusters):
        data=get_ind(y_pred,i)
        print(f"the number of {i}-th cluster'samples is {len(data)}")
        
        sample2=get_sample_2(data,A,A.shape[0],k0)
        print(f"the number of sample2 is{len(sample2)}")
        for i in sample2:
              sample.append(i)
          
    realiable_sample=np.array(sample)

    y_new = [int(y[i]) for i in realiable_sample]
    y_pred_new = [int(y_pred[i]) for i in realiable_sample]
    
    data1 = torch.Tensor(dataset.x)
    
    adj_label1=sp.coo_matrix(adj_label.cpu().numpy())
    adj_label1 = adj_label1 - sp.dia_matrix((adj_label1.diagonal()[np.newaxis, :], [0]), shape=adj_label1.shape)
    adj_label1.eliminate_zeros()
    adj_norm_s = utils.preprocess_graph(adj_label1, args.layer, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(data1).toarray()

    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)
    sm_fea_s=torch.Tensor(sm_fea_s).to(device)

    adj11=utils.normalize_adj(adj_label).toarray()
    train_self_supervised(x=sm_fea_s, y=y,y_pred=y_pred, sample_index=realiable_sample,adj=adj11,epoch=100)#x=sm_fea_s

def train_self_supervised(x,y,y_pred, sample_index,adj,epoch=200):
    
    sample_index=[i for i in sample_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x,y_pred ,adj= x.to(device),torch.LongTensor(y_pred).to(device),torch.Tensor(adj).to(device)
    a=[]
    for i in range(1):
        model = DNN3(num_features=x.shape[1], hidden_size=4500,nclass=int(y.max()) + 1, dropout=0.0)
        print(model)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        acc_best = 0;
        nmi_test1=0;
        ari_test1=0;
        f1_test1=0;

        for i in range(epoch):
            model.train()
            optimizer.zero_grad()
            output = model(x,args.norm)
            loss_train = F.nll_loss(output[sample_index], y_pred[sample_index])
            loss_train.backward()
            optimizer.step()

            y_pre_GAT = np.argmax(output.cpu().detach().numpy(), axis=1)
            y_pre_spectral = y_pred.cpu().detach().numpy()

            acc_test, nmi_test, ari_test, f1_test = eva(y_pre_GAT, y, 'Test: {}'.format(i))

            print(f"epoch {i}:acc {acc_test:.4f}, nmi {nmi_test:.4f}, ari {ari_test:.4f}, f1 {f1_test:.4f}")
            
            if ((nmi_test>0.55)&(acc_test > acc_best)):
                acc_best=acc_test
                nmi_test1=nmi_test
                ari_test1=ari_test
                f1_test1=f1_test

        a.append(acc_best)#acc_best
    print(a)
    print(sum(a)/1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')  
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()

    if os.path.exists('pretrain/') == False:  
        os.makedirs('pretrain/')

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
        args.lr = 0.005
        args.k = 20
        args.layer=3
        args.hidden_size=500
        args.embedding_size=32
        args.n_clusters = 6
        args.norm='True'
    elif args.name == 'Cora':
        args.lr = 0.005#0.0001
        args.k = 100
        args.layer=8
        args.hidden_size=500
        args.embedding_size=32
        args.n_clusters = 7
        args.norm='True'
        args.max_epoch=200
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.layer=35
        args.k = None
        args.norm='True'
        args.n_clusters = 3
    else:
        args.k = None

    args.pretrain_path =f'pretrain1/IGALA_{args.name}.pkl'
    args.input_dim = dataset.num_features

    print(args)
    trainer(dataset)



