import os
import pickle

import numpy as np
import torch
from scipy import sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_nn_torch(Q, X, y, k=1):
    dist = torch.sum((np.squeeze(X) - np.squeeze(Q)).pow(2.), 1)
    ind = torch.argsort(dist)
    label = y[ind[:k]]
    unique_label = torch.unique(y)
    unique_label = unique_label.long()
    count = np.zeros((1,unique_label.shape[0]))
    for i in label:
        count[0,unique_label[i.long()]] += 1
    count_torch = torch.from_numpy(count).float()
    ii = torch.argmax(count_torch)
    pred_prob = count_torch/torch.sum(count_torch)
    final_label = unique_label[ii]
    return pred_prob


def to_adjmatrix(adj_sparse, adj_size):
    A = torch.sparse_coo_tensor(adj_sparse[:, :2].T, adj_sparse[:, 2],
                                size=[adj_size, adj_size]).to_dense()
    return A


def extrac_degree_centrality_sparse(adj_sparse, sen_api_idx, adj_size):
    adj_tmp = to_adjmatrix(adj_sparse)
    adj_tmp = adj_tmp.float()
    all_degree = torch.div((torch.sum(adj_tmp, 0) + torch.sum(adj_tmp, 1)), (adj_tmp.shape[0] - 1))
    feature = torch.matmul(sen_api_idx, all_degree.float())
    # print("\t\t max feature:", torch.max(feature))
    return feature


def extract_degree_centrality(adj, sen_api_idx):
    adj_tmp = adj.float()
    all_degree = torch.div((torch.sum(adj_tmp, 0) + torch.sum(adj_tmp, 1)), (adj_tmp.shape[0] - 1))
    feature = torch.matmul(sen_api_idx, all_degree.float())
    # print("\t\t max feature:", torch.max(feature))
    return feature


def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


def load_train_data(data_path):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    X_train = data3["adjacent_matrix"]
    Idx_train = data3["sensitive_api_index"]
    y_train = data3["label"]
    train_feature = data3["degree_feature_data"]
    return X_train, Idx_train, y_train, train_feature


def degree_centrality_extraction(adjacent_matrix, sen_idx):
    centrality = (adjacent_matrix.sum(axis=0) + adjacent_matrix.sum(axis=1).transpose()) / (
            adjacent_matrix.shape[0] - 1)

    centrality = np.array(centrality)
    centrality = np.squeeze(centrality)
    idx_matrix = np.zeros((len(sen_idx), adjacent_matrix.shape[0]))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)

    return feature


def extract_sensitive_api(sensitive_api_list, nodes_list):
    sample_sensitive_api = []
    for x in sensitive_api_list:
        if x in nodes_list:
            sample_sensitive_api.append(nodes_list.index(x))
        else:
            sample_sensitive_api.append(-1)
    return np.array(sample_sensitive_api)


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


def load_test_data(data_path, idx):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    apk_name_list = list(data3)

    X_test_sha256 = data3["sha256"][idx]
    X_test_am = data3["adjacent_matrix"][idx]
    X_test_sen_idx = data3["sensitive_api_idx"][idx]
    y_test = 0

    X_test_degree_cen = degree_centrality_extraction(X_test_am, X_test_sen_idx)
    return X_test_sha256, X_test_am, X_test_sen_idx, y_test, X_test_degree_cen


def load_all_test_data(data_path):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    apk_name_list = list(data3)

    X_test_sha256 = data3["sha256"]
    X_test_am = data3["adjacent_matrix"]
    X_test_sen_idx = data3["sensitive_api_idx"]

    return X_test_sha256, X_test_am, X_test_sen_idx


def find_2nd_nn_l2(Q, y_Q, X, y_X, k):
    if type(Q) is list:
        Q = np.array(Q)
    if type(X) is list:
        X = np.array(X)
    assert Q.shape[0] == X.shape[1]
    class_num = 2
    nn = np.zeros((1, k), dtype=np.int32)
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    # for i, q in enumerate(Q):
    dist = np.sum(np.square(X - Q), axis=axis)
    dist = np.squeeze(dist)
    ind = np.argsort(dist)
    ind = np.squeeze(ind).transpose()
    mean_dist = np.zeros((class_num,))
    for j in range(class_num):
        ind_j = ind[y_X[ind] == j]
        dist_j = dist[ind_j]
        mean_dist[j] = np.mean(dist_j)

    ind_dist = np.argsort(mean_dist)
    if ind_dist[0] == y_Q:
        nn = ind[y_X[ind] == ind_dist[1]][:k]
    else:
        nn = ind[y_X[ind] == ind_dist[0]][:k]
    return nn


def trans2triple_rw(adjacent_matrix, sha256, triple_path, overwrite_flag=True):
    '''

    :param adjacent_matrix:
    :param sha256:
    :param overwrite_flag: 决定是否重新计算triple matrix
    :return: triple matrix
    '''
    file_name = triple_path + "/" + sha256 + ".npy"
    # print(file_name)
    triple = []
    if type(adjacent_matrix) is sparse.coo_matrix:
        adjacent_matrix = adjacent_matrix.tocsr()

    if not overwrite_flag:
        if os.path.exists(file_name):
            print("\t loading data...")
            fsize = os.path.getsize(file_name)
            fsize = fsize / float(1024 * 1024)
            if fsize > 500:
                return None
            triple = np.load(file_name)
        else:
            print("\t getting triple data")
            node_number = adjacent_matrix.shape[0]
            print("\t\t node number ", node_number)
            triple = []
            for zi in range(node_number):
                # triple.append([zi, zi, adjacent_matrix[zi, zi]])
                for zj in range(zi + 1, node_number):
                    triple.append([zi, zj, adjacent_matrix[zi, zj]])
                    triple.append([zj, zi, adjacent_matrix[zj, zi]])
            triple = np.array(triple)
            np.save(file_name, triple)
            fsize = os.path.getsize(file_name)
            fsize = fsize / float(1024 * 1024)
            if fsize > 500:
                return None
    else:
        node_number = adjacent_matrix.shape[0]
        triple = []
        for zi in range(node_number):
            # triple.append([zi, zi, adjacent_matrix[zi, zi]])
            for zj in range(zi + 1, node_number):
                triple.append([zi, zj, adjacent_matrix[zi, zj]])
                triple.append([zj, zi, adjacent_matrix[zj, zi]])
        triple = np.array(triple)
        np.save(file_name, triple)
        fsize = os.path.getsize(file_name)
        fsize = fsize / float(1024 * 1024)
        if fsize > 500:
            return None
    return triple

def get_mamafeatures(adjs,node_index,state_category="families",ifnodes=False):
        test_adj = adjs
        pack_idx = node_index
        # # print(test_adj)
        # pack_idx =np.array(node_index[i])
        train_feature = extract_feature_torch(test_adj, pack_idx, state_category,ifnodes)
            # print(train_features)
        return train_feature


def extract_feature_torch(adj, pack_idx, type,ifnodes=False):
    '''
    adj: csr_matrix: adjacent matrix
    pack_idx: nparray: node number * 1, the package index of each node
    num: package: 11; package: 446
    '''
    if type == "families":
        nn = 11
    else:
        nn = 446
        # idx_one_hot = torch.zeros((pack_idx.size, nn)).to(torch.float64)
    if ifnodes:
        adjcopy = adj.copy().toarray()
        adjcopy = torch.from_numpy(adjcopy).to(torch.float64).cuda()
        nodes_num = adjcopy.size(0)
        feature = torch.zeros((nodes_num,nn**2)).cuda().to(torch.float64)
        for i in range(nodes_num):
            adj_nodes = torch.zeros((nodes_num,nodes_num)).to(torch.float64).cuda()
            adj_nodes[:,i] = adjcopy[:,i]
            adj_nodes[i,:] = adjcopy[i,:]
            idx_one_hot = torch.zeros((pack_idx.size, nn)).to(torch.float64).cuda()
            idx_one_hot[np.arange(pack_idx.size), pack_idx] = 1
            # adj = torch.from_numpy(adj).to(torch.int32)
            call_relation = idx_one_hot.T.matmul(adj_nodes.matmul(idx_one_hot))

            # MarkovFeats = np.zeros((max(pack_idx)+1, max(pack_idx)+1))
            MarkovFeats = torch.zeros((nn, nn))
            Norma_all = torch.sum(call_relation, axis=1)
            for i in range(0, len(call_relation)):
                Norma = Norma_all[i]
                if (Norma == 0):
                    MarkovFeats[i] = call_relation[i]
                else:
                    MarkovFeats[i] = call_relation[i] / Norma

            feature_nodes = MarkovFeats.flatten()
            feature[i,:] = feature_nodes
    else:
        adjcopy = adj.copy().toarray()
        adjcopy= torch.from_numpy(adjcopy).to(torch.float64).cuda()
        idx_one_hot = torch.zeros((pack_idx.size, nn)).to(torch.float64).cuda()
        idx_one_hot[np.arange(pack_idx.size), pack_idx] = 1
        # adj = torch.from_numpy(adj).to(torch.int32)
        call_relation = idx_one_hot.T.matmul(adjcopy.matmul(idx_one_hot))

        # MarkovFeats = np.zeros((max(pack_idx)+1, max(pack_idx)+1))
        MarkovFeats = torch.zeros((nn, nn))
        Norma_all = torch.sum(call_relation, axis=1)
        for i in range(0, len(call_relation)):
            Norma = Norma_all[i]
            if (Norma == 0):
                MarkovFeats[i] = call_relation[i]
            else:
                MarkovFeats[i] = call_relation[i] / Norma

        feature = MarkovFeats.flatten().detach()
    return feature


