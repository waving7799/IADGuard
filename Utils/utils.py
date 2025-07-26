import os
from scipy import sparse

import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def degree_centrality(adjacent_matrix, sen_idx):
    centrality = (adjacent_matrix.sum(axis=0) + adjacent_matrix.sum(axis=1).transpose()) / (
        adjacent_matrix.shape[0] - 1)

    centrality = np.array(centrality)
    centrality = np.squeeze(centrality)
    idx_matrix = np.zeros((len(sen_idx), adjacent_matrix.shape[0]))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature


def katz_feature(graph, sen_idx, alpha=0.1, beta=1.0, normalized=True, weight=None):
    graph = graph.T
    n = graph.shape[0]
    b = np.ones((n, 1)) * float(beta)
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * graph), b)
    if normalized:
        norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
    else:
        norm = 1.0
    centrality = centrality / norm
    idx_matrix = np.zeros((len(sen_idx), n))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature



def to_adjmatrix(adj_sparse, adj_size):
    A = torch.sparse_coo_tensor(adj_sparse[:, :2].T, adj_sparse[:, 2],
                                size=[adj_size, adj_size]).to_dense()
    return A


def degree_centrality_torch(adj, sen_api_idx):
    adj_size = adj.shape[0]
    idx_matrix = np.zeros((len(sen_api_idx), adj_size))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1

    idx_matrix = torch.from_numpy(idx_matrix).to(device)
    all_degree = torch.div((torch.sum(adj, 0) + torch.sum(adj, 1)).float(),
                           float(adj.shape[0] - 1))
    degree_centrality = torch.matmul(
        idx_matrix, all_degree.type_as(idx_matrix))
    return degree_centrality


def katz_feature_torch(graph, sen_api_idx, alpha=0.1, beta=1.0, device='cuda:0', normalized=True):
    n = graph.shape[0]
    graph = graph.T
    b = torch.ones((n, 1)) * float(beta)
    b = b.to(device)
    graph = graph.to(device)
    A = torch.eye(n, n).to(device).float() - (alpha * graph.float())
    # L, U = torch.solve(b, A)
    L = torch.linalg.solve(A, b)
    if normalized:
        norm = torch.sign(sum(L)) * torch.norm(L)
    else:
        norm = 1.0
    centrality = torch.div(L, norm.to(device)).to(device)
    idx_matrix = np.zeros((len(sen_api_idx), n))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)
    katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
    return katz_centrality

def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis

def extract_sensitive_api(sensitive_api_list, nodes_list):
    sample_sensitive_api = []
    for x in sensitive_api_list:
        if x in nodes_list:
            sample_sensitive_api.append(nodes_list.index(x))
        else:
            sample_sensitive_api.append(-1)
    return np.array(sample_sensitive_api)


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
