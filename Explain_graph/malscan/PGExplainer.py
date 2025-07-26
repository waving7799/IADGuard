import os
import sys
sys.path.append("../../")
sys.path.append("../")
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from BaseExplainer import BaseExplainer
from scipy import sparse
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.sparse import coo_matrix

from Utils.lib import find_nn_torch
import gc

class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """
    def __init__(self,explain_sha256, graphs, features,subgraph_dict, \
        task, X_train_torch, y_train_torch,expname,k,\
        epochs=1, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(graphs, X_train_torch, y_train_torch, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = 43972
        self.subgraph_dict = subgraph_dict
        self.explain_sha256 = explain_sha256
        self.expname = expname
        self.k = k

    def _get_sensi_edges(self, graph, sens):
        adj_sparse = graph.copy().toarray()
        sens_nodes_all = []
        ii = np.where(sens != -1)
        sens_nodes = sens[ii].tolist()
        sens_nodes_all += sens_nodes
        
        for a in sens_nodes:
            x_callee = adj_sparse[a,:]
            caller_x = adj_sparse[:,a]
            sens_nodes_all += np.where(x_callee==1)[0].tolist()
            sens_nodes_all += np.where(caller_x==1)[0].tolist()
        return sens_nodes_all

    def _create_explainer_input(self, graph, sens):
        """
        Given the embedding of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = graph.row
        cols = graph.col
        new_rows= []
        new_cols= []
        sens_nodes_all = self._get_sensi_edges(graph, sens)
        graph_edge = graph.copy()
        data = graph_edge.data
        
        input_expl = self.malscan_feature(graph_edge,sens).detach()
        input_expl = input_expl.cpu().unsqueeze(0)
        input_expl_ori = input_expl
        for i in range(len(data)):
            if rows[i] in sens_nodes_all or cols[i] in sens_nodes_all:
                new_rows.append(rows[i])
                new_cols.append(cols[i])
                data_edge = data.copy()
                data_edge[i] = 0
                graph_edge.data = data_edge
                feature_edge = self.malscan_feature(graph_edge,sens).detach().cpu()
                feature_delta = input_expl_ori-feature_edge
                norm = torch.norm(feature_delta)
                if norm != 0:
                    feature_delta = torch.div(feature_delta,torch.norm(feature_delta))
                input_expl = torch.cat((input_expl, feature_delta), 0)
            

        return input_expl,new_rows,new_cols


    def _sample_graph(self, sampling_weights, temperature=2.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = ((bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias))
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph


    def _loss(self, masked_pred,newgraph, original_pred, mask, reg_coefs,true_subgraph=None,ifsp=False):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        mask = mask.cuda()
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask).cuda() * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg).cuda()

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred).cuda()
        
        if ifsp:
            # Mask loss
            masked_graph = torch.from_numpy(newgraph.toarray()).cuda()
            true_subgraph = torch.from_numpy(true_subgraph).cuda()
            distance = F.pairwise_distance(masked_graph,true_subgraph,p=2).cuda()
            p2_loss = torch.sum(distance).cuda()
            print("cce_loss:"+str(10*cce_loss))
            print("size_loss:"+str(size_loss*0.1))
            print("mask_ent_loss:"+str(mask_ent_loss*10))
            print("p2_loss:"+str(0.2*p2_loss))
            return 10*cce_loss + 0.1*size_loss + 10*mask_ent_loss+ 0.2*p2_loss
        
        print("cce_loss:"+str(10*cce_loss))
        print("size_loss:"+str(0.1*size_loss))
        print("mask_ent_loss:"+str(mask_ent_loss*10))
        return 10*cce_loss + 0.1*size_loss + mask_ent_loss*10 

    def prepare(self, indices=None,ifsp=False,train=False,model_path=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 1000),
            nn.ReLU(),
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))
        if train:
            self.train(indices=indices,ifsp=ifsp)
        else:
            self.explainer_model.load_state_dict(torch.load(model_path))

    def mask_edges(self,graph,sens_idx):
        ii = np.where(sens_idx != -1)
        ii_idx = sens_idx[ii]
        
        data_len = len(graph.data)
        random_num =int(data_len*0.2)
        perturb_idx = np.random.choice(ii_idx,1)
        
        # 随机mask一部分边
        graph_mask = graph.copy()
        mask_col_del = np.random.choice(data_len,random_num)
        graph_mask.data[mask_col_del]=0
        
        #随机添加一部分边
        rows = graph.row
        cols = graph.col
        graph_mask = graph_mask.toarray()
        # graph_len = graph_mask.shape[0]        
        mask_col_add = np.random.choice(cols,random_num)
        mask_row_add = np.random.choice(rows,random_num)
        graph_mask[perturb_idx,mask_col_add]=1
        graph_mask[mask_row_add,perturb_idx]=1
        graph_mask = sparse.coo_matrix(graph_mask)
        # print(mask_col_del,mask_col_add,mask_row_add)
        return graph_mask
    
    def _masked_adj(self,mask,graph,new_rows,new_cols):
        
        sym_mask = mask.detach().cpu().numpy()
        sym_mask = coo_matrix((sym_mask, (new_rows, new_cols)), shape=graph.shape)

        sparseadj = graph.copy().toarray()

        masked_adj = sparseadj*sym_mask

        num_nodes = sparseadj.shape[0]
        torchones = np.ones((num_nodes, num_nodes))
        diag_mask = torchones - np.eye(num_nodes)
        masked_graph = masked_adj*diag_mask
        
        masked_graph = coo_matrix(masked_graph)

        return masked_graph

    def train(self, indices = None,ifsp=False):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0]*((self.temp[1]/self.temp[0])**(e/self.epochs))

        # Start training loop
        for e in range(0, self.epochs):
            print("Epochs:",str(e)) 
            t = temp_schedule(e)
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).cuda().detach()
            
            for n in tqdm(indices):
                n = int(n)

                graph_0 = self.graphs[n].copy()
                if len(graph_0.data)<2:
                    continue
                for i in range(1):
                    
                    sens = self.sen_api_idx[n]
                    graph = graph_0.copy()
                    features = self.malscan_feature(graph_0,sens).detach()

                    # Sample possible explanation
                    input_expl,new_rows,new_cols = self._create_explainer_input(graph, sens)
                    input_expl = input_expl.unsqueeze(0).cpu()
                    sampling_weights = self.explainer_model(input_expl)
                    sampling_weights = sampling_weights.reshape(1,-1).squeeze()
                    mask = self._sample_graph(sampling_weights[1:], t, bias=self.sample_bias)

                    
                    masked_graph = self._masked_adj(mask,graph,new_rows,new_cols)
                    mask_embed = self.malscan_feature(masked_graph,sens).detach()
                    
                    
                    masked_pred,_,_ = find_nn_torch(mask_embed.unsqueeze(0), self.X_train_torch, self.y_train_torch,self.k)
                    original_pred,_,_ = find_nn_torch(features.unsqueeze(0), self.X_train_torch, self.y_train_torch,self.k)

                    # release the cuda memory
                    
                    mask_embed = mask_embed.cpu()
                    features = features.cpu()
                    masked_pred = masked_pred.cpu()
                    original_pred = original_pred.cpu()
                    torch.cuda.empty_cache()
                    
                    del mask_embed
                    del features

                    print("adv label: " + str(original_pred))
                    print("masked label: " + str(masked_pred))

                    if self.type == 'node': # we only care for the prediction of the node
                        masked_pred = masked_pred[n].unsqueeze(dim=0)
                        original_pred = original_pred[n]
                        
                    if ifsp:
                        true_subgraph = self.subgraph_dict[n]
                        id_loss = self._loss(masked_pred,masked_graph, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs,true_subgraph,ifsp=True)
                    else:
                        id_loss = self._loss(masked_pred,masked_graph, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                    loss += id_loss
                    
            print("Epoch:"+str(e)+"\t"+"Total loss:"+str(loss))
            loss.backward()
            optimizer.step()
            del loss
            torch.cuda.empty_cache()
            gc.collect()
        self.explainer_model.eval()
        torch.save(self.explainer_model.state_dict(), 'new_explainer_'+self.expname+'.pkl') 

    def get_label(self, graph,sens):
        features = self.malscan_feature(graph,sens).detach()
        _,label,_ = find_nn_torch(features.unsqueeze(0), self.X_train_torch, self.y_train_torch,self.k)
        label = label.cpu().numpy()
        return label


    def explain(self, graph,sens):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """

        # Use explainer mlp to get an explanation
        input_expl,new_rows,new_cols = self._create_explainer_input(graph, sens)
        input_expl = input_expl.squeeze(0).cpu()
        sampling_weights = self.explainer_model(input_expl)
        sampling_weights = sampling_weights.reshape(1,-1).squeeze()
        print(sampling_weights)
        mask = self._sample_graph(sampling_weights[1:]).squeeze()
        print(mask)
        
        explain = []
        for i in range(len(new_rows)):
            edge_exp = []
            edge_exp.append(new_rows[i])
            edge_exp.append(new_cols[i])
            edge_exp.append(mask[i].cpu().detach().numpy())
            explain.append(edge_exp)
        explain.sort(key=lambda x: abs(x[2]), reverse=True)
        explain_edge =np.array([x[:2] for x in explain])
        explain_score = np.array([x[2] for x in explain])

        return explain_edge, explain_score
    
    
    def index_edge(self,graph, pair):
        return torch.where((graph.T == pair).all(dim=1))[0]
    
    def to_adjmatrix(self, adj_sparse):
        adj_size = adj_sparse.shape[0] 
        indices = torch.tensor([adj_sparse.row, adj_sparse.col])
        value = torch.tensor(adj_sparse.data)
        A = torch.sparse_coo_tensor(indices,value,
                                        size=[adj_size, adj_size]).to_dense()

        return A
            
    def Degree_Centrality_torch(self, adj, sen_api_idx,ifnode=False):
        
        idx_matrix = np.zeros((len(sen_api_idx), adj.shape[0]))
        ii = np.where(sen_api_idx != -1)
        idx_matrix[ii, sen_api_idx[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix).cuda()
        adj_dense = self.to_adjmatrix(adj)
        if ifnode:
            all_degree = torch.div(adj_dense, (adj_dense.shape[0] - 1))
            all_degree = all_degree.T
        else:
            all_degree = torch.div((torch.sum(adj_dense, 0) + torch.sum(adj_dense, 1)), (adj_dense.shape[0] - 1))            
        degree_centrality = torch.matmul(idx_matrix, all_degree.type_as(idx_matrix))
        return degree_centrality
    
    
    def katz_feature_torch(self,graph,sen_api_idx,ifnode=False,alpha=0.1, beta=1.0, normalized=True):
        n = graph.shape[0] 
        graph = graph.T.cuda()
        
        if ifnode:
            b = torch.ones((n, n)).cuda() * float(beta)
            A = torch.eye(n, n).cuda().float() - (alpha * graph.float())
            L,U= torch.solve(b, A)
            if normalized:
                norm = torch.sign(sum(L)) * torch.norm(L)
            else:
                norm = 1.0
            centrality = torch.div(L, norm.cuda()).cuda()
        else:
            b = torch.ones((n, 1)).cuda() * float(beta)
            A = torch.eye(n, n).cuda().float() - (alpha * graph.float())
            L = torch.linalg.solve(A, b)
            if normalized:
                norm = torch.sign(sum(L)) * torch.norm(L)
            else:
                norm = 1.0
            centrality = torch.div(L, norm.cuda()).cuda()

        idx_matrix = np.zeros((len(sen_api_idx), n))

        ii = np.where(sen_api_idx != -1)
        idx_matrix[ii, sen_api_idx[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix).cuda()
        katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
        return katz_centrality
    
    def malscan_feature(self, graph,sen_api_idx):
        feature = self.Degree_Centrality_torch(graph, sen_api_idx)
        densegraph = self.to_adjmatrix(graph)
        feature_katz = self.katz_feature_torch(densegraph, sen_api_idx)

        feature = torch.cat((feature, np.squeeze(feature_katz)), 0)
        return feature.to(torch.float32)      
    
    def node_feature(self, graph,sen_api_idx):
        # extract centrality embedding for every edge
        feature = self.Degree_Centrality_torch(graph, sen_api_idx,ifnode=True)
        densegraph = self.to_adjmatrix(graph)
        feature_katz = self.katz_feature_torch(densegraph, sen_api_idx,ifnode=True)
        feature = torch.cat((feature, np.squeeze(feature_katz)), 0)
        return feature.to(torch.float32) 
    

