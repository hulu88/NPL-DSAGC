import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid
from collections import Counter

def get_dataset(dataset):                            
    datasets = Planetoid(root='./dataset', name=dataset)
    return datasets

def data_preprocessing(dataset):                       
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj
    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def SMnormalize_adj(adj):                           
    adj = sp.coo_matrix(adj)
    asm = adj + sp.eye(adj.shape[0])
    asm = sp.coo_matrix(asm)
    rowsum = np.array(asm.sum(1))  
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
    return asm.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  

def SPnormalize_adj(adj):                          
    adj = sp.coo_matrix(adj)
    asp = 2*sp.eye(adj.shape[0])-adj
    asp = sp.coo_matrix(asp)
    temp= 2*sp.eye(adj.shape[0])+adj
    rowsum = np.array(temp.sum(1))    
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
    return asp.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  



def preprocess_graph(adj, layer, norm='sym', renorm=True):     
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj
    rowsum = np.array(adj_.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [2 / 3] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    return adjs




def sparse_matrix_to_torch(X):                 
   # coo = X.tocoo()
    if(sp.isspmatrix_coo(X)==False):
        coo=X.tocoo()
    else:
        coo=X
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices.astype(np.float32)),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)

def spectral_radius(M):
    a,b=np.linalg.eig(M)
    return np.max(np.abs(a))

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj=adj.cpu().numpy() + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()  



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


