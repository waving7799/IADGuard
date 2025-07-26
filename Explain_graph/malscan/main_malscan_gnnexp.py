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



def load_all_test_data(data_path ):
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

def arg_parse():
    parser = argparse.ArgumentParser(description="GNN Explainer arguments.")
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument(
        "--bmname", dest="bmname", help="Name of the benchmark dataset"
    )
    io_parser.add_argument("--pkl", dest="pkl_fname", help="Name of the pkl data file")

    parser.add_argument(
        "--gpu",
        dest="gpu",
        action="store_const",
        const=True,
        default=True,
        help="whether to use GPU.",
    )
    parser.add_argument(
        "--epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output-dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--no-writer",
        dest="writer",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    # Explainer
    parser.add_argument("--mask-act", dest="mask_act", type=str, help="sigmoid, ReLU.")
    parser.add_argument(
        "--mask-bias",
        dest="mask_bias",
        action="store_const",
        const=True,
        default=False,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--explain-node", dest="explain_node", type=int, help="Node to explain."
    )
    parser.add_argument(
        "--graph-idx", dest="graph_idx", type=int, help="Graph to explain."
    )
    parser.add_argument(
        "--graph-mode",
        dest="graph_mode",
        action="store_const",
        const=True,
        default=False,
        help="whether to run Explainer on Graph Classification task.",
    )
    parser.add_argument(
        "--multigraph-class",
        dest="multigraph_class",
        type=int,
        help="whether to run Explainer on multiple Graphs from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--multinode-class",
        dest="multinode_class",
        type=int,
        help="whether to run Explainer on multiple nodes from the Classification task for examples in the same class.",
    )
    parser.add_argument(
        "--align-steps",
        dest="align_steps",
        type=int,
        help="Number of iterations to find P, the alignment matrix.",
    )

    parser.add_argument(
        "--method", dest="method", type=str, help="Method. Possible values: base, att."
    )

    # TODO: Check argument usage
    parser.set_defaults(
        opt="adam",  
        opt_scheduler="none",
        cuda="0",
        lr=0.1,
        clip=2.0,
        batch_size=20,
        num_epochs=100,
        hidden_dim=20,
        output_dim=20,
        num_gc_layers=3,
        dropout=0.0,
        method="base",
        name_suffix="",
        explainer_suffix="",
        align_steps=1000,
        explain_node=None,
        graph_idx=-1,
        mask_act="sigmoid",
        multigraph_class=-1,
        multinode_class=-1,
    )

    parser.add_argument("--train_file", type=str, default="../../data/malscan/traindata_fea_data.pkl")
    parser.add_argument("--test_file", type=str, default="../../data/malscan/test_dict_data.pkl")
    parser.add_argument("--k", type=int, default=1, help="k for knn classifier")
    parser.add_argument("--task", type=str, default="graph")
    parser.add_argument("--output", type=str, default="malscan_explain.npz")

    return parser.parse_args()    
    
    
if __name__ == '__main__':
    prog_args = arg_parse()
    output_path = prog_args.output
    k = prog_args.k
    task = prog_args.task

    
    train_path = prog_args.train_file
    df = open(train_path, "rb")
    data_train_dict = pickle.load(df)
    df.close()
    train_sha256 = data_train_dict["sha256"]
    train_feature = data_train_dict["feature"]
    train_label = data_train_dict["label"]  
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_feature, train_label)
    
    X_adv_test_sha256, X_adv_test_adj, X_adv_test_sens = load_all_test_data(prog_args.test_file)


    # For unsupersived
    explain_sha256 = np.array(X_adv_test_sha256)
    explain_train = np.array(X_adv_test_adj)
    explain_sens = np.array(X_adv_test_sens)
    
    #For GNNexplainer
    explainer = GNNexplainer(model,explain_train,explain_sens,task,train_feature,train_label,prog_args)
    explain_idx = []
    explain_feature = []
    explain_score = []

    for i in range(len(explain_sha256)):
        if len(explain_train[i].data)<2:
            continue
        name = explain_sha256[i]
        masked_adj = explainer.gnn_explain(name,graph_idx=i)
        explain_idx.append(explain_sha256[i])
        exp_edge, exp_score = explainer.explain(masked_adj)
        explain_feature.append(exp_edge)
        explain_score.append(exp_score)

    np.savez(output_path, sha256_name= explain_idx, explain_feature=explain_feature,explain_score=explain_score,dtype=object)
    
 