import os
import torch
import logging
import numpy as np
from cytoolz import curry
import multiprocessing as mp
from scipy import sparse as sp
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize, StandardScaler
from torch_scatter import scatter
from torch.utils.data import random_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, \
    add_remaining_self_loops, remove_self_loops, get_laplacian, to_undirected, to_dense_adj, to_networkx


def setup_logger(log_file_name):
    log_dir = "./logs"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = os.path.join(log_dir, log_file_name)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def standardize(feat, mask):
    scaler = StandardScaler()
    scaler.fit(feat[mask])
    new_feat = torch.FloatTensor(scaler.transform(feat))
    return new_feat
    
    
def preprocess(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return torch.tensor(features)


class PPR:
    #Node-wise personalized pagerank
    def __init__(self, adj_mat, maxsize=200, n_order=2, alpha=0.85):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    @curry
    def process(self, path, seed):
        ppr_path = os.path.join(path, 'ppr{}'.format(seed))
        if not os.path.isfile(ppr_path) or os.stat(ppr_path).st_size == 0:
            # print ('Processing node {}.'.format(seed))
            neighbor = self.search(seed)
            torch.save(neighbor, ppr_path)
        else :
            print ('File of node {} exists.'.format(seed))
    
    def search_all(self, node_num, path):
        neighbor  = {}
        # if os.path.isfile(path+'_neighbor') and os.stat(path+'_neighbor').st_size != 0:
        #     print ("Exists neighbor file")
        #     neighbor = torch.load(path+'_neighbor')
        # else :
        print ("Extracting subgraphs")
        os.system('mkdir {}'.format(path))
        with mp.Pool() as pool:
            list(pool.imap_unordered(self.process(path), list(range(node_num)), chunksize=1000))
            
        print ("Finish Extracting")
        for i in range(node_num):
            neighbor[i] = torch.load(os.path.join(path, 'ppr{}'.format(i)))
        torch.save(neighbor, path+'_neighbor')
        os.system('rm -r {}'.format(path))
        print ("Finish Writing")
        return neighbor

    
class Subgraph:
    #Class for subgraph extraction
    
    def __init__(self, x, edge_index, path, maxsize=50, n_order=10):
        self.x = x
        self.path = path
        edge_index = edge_index.clone().cpu()
        self.edge_index = np.array(edge_index)
        self.edge_num = edge_index[0].size(0)
        self.node_num = x.size(0)
        self.maxsize = maxsize
        
        self.sp_adj = sp.csc_matrix((np.ones(self.edge_num), (edge_index[0], edge_index[1])), 
                                    shape=[self.node_num, self.node_num])
        self.ppr = PPR(self.sp_adj, n_order=n_order)
        
        self.neighbor = {}
        self.adj_list = {}
        self.subgraph = {}
        
    def process_adj_list(self):
        for i in range(self.node_num):
            self.adj_list[i] = set()
        for i in range(self.edge_num):
            u, v = self.edge_index[0][i], self.edge_index[1][i]
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)
            
    def adjust_edge(self, idx):
        #Generate edges for subgraphs
        dic = {}
        for i in range(len(idx)):
            dic[idx[i]] = i
            
        new_index = [[], []]
        nodes = set(idx)
        for i in idx:
            edge = list(self.adj_list[i] & nodes)
            edge = [dic[_] for _ in edge]
            #edge = [_ for _ in edge if _ > i]
            new_index[0] += len(edge) * [dic[i]]
            new_index[1] += edge
        return torch.LongTensor(new_index)

    def adjust_x(self, idx):
        #Generate node features for subgraphs
        return self.x[idx]            
    
    def build(self):
        #Extract subgraphs for all nodes
        # if os.path.isfile(self.path+'_subgraph') and os.stat(self.path+'_subgraph').st_size != 0:
        #     print ("Exists subgraph file")
        #     self.subgraph = torch.load(self.path+'_subgraph')
        #     return 
        
        self.neighbor = self.ppr.search_all(self.node_num, self.path)
        self.process_adj_list()
        for i in range(self.node_num):
            nodes = self.neighbor[i][:self.maxsize]
            x = self.adjust_x(nodes)
            edge = self.adjust_edge(nodes)
            self.subgraph[i] = Data(x, edge)
        torch.save(self.subgraph, self.path+'_subgraph')
        
    def search(self, node_list):
        #Extract subgraphs for nodes in the list
        batch = []
        index = []
        size = 0
        for node in node_list:
            batch.append(self.subgraph[node])
            index.append(size)
            size += self.subgraph[node].x.size(0)
        index = torch.tensor(index)
        batch = Batch().from_data_list(batch)
        return batch, index