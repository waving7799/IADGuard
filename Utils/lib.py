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
    
    label_total = y[ind]
    list_label = label_total.cpu().numpy()
    benign_idx = np.argwhere(list_label==1)
    min_dist = dist[ind[benign_idx[0][0]]]

    unique_label = torch.unique(y)
    unique_label = unique_label.long()
    count = np.zeros((1,unique_label.shape[0]))
    for i in label:
        count[0,unique_label[i.long()]] += 1
    count_torch = torch.from_numpy(count).float()
    ii = torch.argmax(count_torch)
    pred_prob = count_torch/torch.sum(count_torch)
    final_label = unique_label[ii]
    return pred_prob,final_label, min_dist
