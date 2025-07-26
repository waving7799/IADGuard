import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from scipy.sparse.coo import coo_matrix
# import torch
from PGExplainer import PGExplainer
import torch
import pickle
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import argparse
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse


def load_all_test_data(data_path):
    df = open(data_path, "rb")
    data = pickle.load(df)
    test_sha256_all = data["sha256"]
    test_adj_all = data["adjacent_matrix"]
    test_idx_all = data["node_idx"]
    return test_sha256_all, test_adj_all, test_idx_all

def load_all_explain_data(data_path):
    df = open(data_path, "rb")
    data3 = pickle.load(df)
    explain_sha256 = data3["sha256"]
    explain_adj = data3["adjacent_matrix"]
    explain_subgraph = data3["subgraph"]
    explain_idx = data3["node_idx"]

    return explain_sha256, explain_adj, explain_subgraph, explain_idx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../../data/mamadroid/train_fea_families.pkl")
    parser.add_argument("--test_file", type=str, default="../../data/mamadroid/test_dict_families.pkl")
    parser.add_argument("--subgraph_file", type=str, default="../../data/mamadroid/subgraph_dict_mama.pkl")
    parser.add_argument("--explainer_name", type=str, default="mamadroid") 
    parser.add_argument("--task", type=str, default="graph")
    parser.add_argument("--train", type=bool, default=True, help="whether to train the explainer")
    parser.add_argument("--k", type=int, default=1, help="k for knn classifier")
    parser.add_argument("--state_cate", type=str, default="families")
    parser.add_argument("--output", type=str, default="mama_explain.npz")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    k = args.k
    train = args.train
    state_category = args.state_cate
    task = args.task
    output_path = args.output
    
    ori_train = open(args.train_file, "rb")
    data = pickle.load(ori_train)
    ori_train.close()
    train_sha256 = np.array(data["sha256"])
    train_feature = np.array(data["feature"])
    train_label = np.array(data["label"])
    
    X_train_torch = torch.from_numpy(train_feature.copy()).to(device, torch.float64)
    y_train_torch = torch.from_numpy(train_label.copy()).to(device)
    
    explain_sha256, explain_adj, explain_subgraph, explain_idx = load_all_explain_data(args.subgraph_file)
       
    # For PGExplainer
    explainer = PGExplainer(explain_sha256,explain_adj,explain_idx,explain_subgraph,state_category,task,X_train_torch,y_train_torch,args.explainer_name,k)
    if train:
        indices = range(0,len(explain_sha256), 1)
        explainer.prepare(indices,ifsp=True,train=True)
    else:
        model_path = "./explainer_"+ args.explainer_name+".pkl"
        indices = range(0,len(explain_sha256), 3)
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

        explain_name.append(X_adv_test_sha256[i])
        exp_edge, exp_score = explainer.explain(X_adv_test_adj[i],X_adv_test_sens[i])
        explain_feature.append(exp_edge)
        explain_score.append(exp_score)

    np.savez(output_path, sha256_name=explain_name, explain_feature=explain_feature,explain_score=explain_score,dtype=object)

