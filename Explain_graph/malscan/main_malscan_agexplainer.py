import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from scipy.sparse.coo import coo_matrix
import torch
from PGExplainer import PGExplainer
from GNNexplainer import GNNexplainer
from torch import nn
import pickle
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_all_test_data(data_path):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    X_test_sha256 = []
    X_test_am = []
    X_test_sen_idx = []
    for name in list(data3):
    # X_test_sha256 = data3["sha256"]
        item = data3[name]
        X_test_sha256.append(name)
        X_test_am.append(item["adjacent_matrix"])
        X_test_sen_idx.append(item["sensitive_api_list"])

    return X_test_sha256, X_test_am, X_test_sen_idx


def load_all_explain_data(data_path):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    explain_sha256 = data3["sha256"]
    explain_adj = data3["adjacent_matrix"]
    explain_subgraph = data3["subgraph"]
    explain_sens = data3["sensitive_api_list"]

    return explain_sha256, explain_adj, explain_subgraph, explain_sens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../../data/malscan/traindata_fea_data.pkl")
    parser.add_argument("--test_file", type=str, default="../../data/malscan/test_dict_data.pkl")
    parser.add_argument("--subgraph_file", type=str, default="../../data/malscan/subgraph_dict_malscan.pkl")
    parser.add_argument("--explainer_name", type=str, default="malscan") 
    parser.add_argument("--task", type=str, default="graph")
    parser.add_argument("--train", type=bool, default=True, help="whether to train the explainer")
    parser.add_argument("--ifsp", type=bool, default=True, help="whether to train in supervised mode")
    parser.add_argument("--k", type=int, default=1, help="k for knn classifier")
    parser.add_argument("--output", type=str, default="malscan_explain.npz")

    return parser.parse_args()


if __name__ == '__main__':
    # load data
    args = get_args()

    task = args.task
    explainer_name = args.explainer_name
    dataname = "adv"  
    k = args.k
    ifsp = args.ifsp
    train = args.train
    output_file = args.output
    
    train_path = args.train_file
    df = open(train_path, "rb")
    data_train_dict = pickle.load(df)
    df.close()
    train_sha256 = data_train_dict["sha256"]
    train_feature = np.array(data_train_dict["feature"])
    train_label = np.array(data_train_dict["label"])  
    
    X_train_torch = torch.from_numpy(train_feature.copy()).to(device, torch.float64)
    y_train_torch = torch.from_numpy(train_label.copy()).to(device)
    
    explain_sha256, explain_adj, explain_subgraph, explain_sens= load_all_explain_data(args.subgraph_file)


    # For PGExplainer
    explainer = PGExplainer(explain_sha256,explain_adj,explain_sens,explain_subgraph,task,X_train_torch,y_train_torch,explainer_name,k)
    if ifsp and train:
        indices = range(0,len(explain_sha256), 1)
        explainer.prepare(indices,ifsp=ifsp,train=True)
    elif ifsp==False:
        indices = range(0,len(explain_sha256), 5)
        explainer.prepare(indices,train=True)
    elif train==False:
        model_path = "./explainer_"+ explainer_name +".pkl"
        indices = range(0,len(explain_sha256), 5)
        explainer.prepare(indices,model_path=model_path)
    
    # select a graph to explain, this needs to be part of the list of indices
    X_adv_test_sha256, X_adv_test_adj, X_adv_test_sens = load_all_test_data(args.test_file)

    explain_name = []
    explain_feature = []
    explain_score = []
    print("explaining samples:"+str(len(X_adv_test_sha256)))
    for i in range(len(X_adv_test_sha256)):
        if len(X_adv_test_adj[i].data)<2:
            continue
        exp_edge, exp_score = explainer.explain(X_adv_test_adj[i],X_adv_test_sens[i])
        explain_feature.append(exp_edge)
        explain_score.append(exp_score)
        explain_name.append(X_adv_test_sha256[i])


    np.savez(output_file, sha256_name=explain_name, explain_feature=explain_feature,explain_score=explain_score,dtype=object)

