import sys
sys.path.append("../../")
import torch
# import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
# from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from BaseExplainer import BaseExplainer
from scipy import sparse
from torch.autograd import Variable
import torch.nn.functional as F
from lib import find_nn_torch

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
    def __init__(self, explain_sha256, graphs, sens_idx,subgraph_dict, state_category,\
        task,X_train_torch,y_train_torch,expname, k,\
        epochs=10, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0):
        super().__init__(graphs, sens_idx, X_train_torch,y_train_torch,  task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = 121
        self.state_categroy = state_category
        self.subgraph_dict = subgraph_dict
        self.explain_sha256 = explain_sha256
        self.expname = expname
        self.k =k

    def _create_explainer_input(self, graph, sens):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        
        graph_edge = graph.copy()
        data = graph_edge.data
        
        input_expl = self.get_mamafeatures(graph_edge,sens).detach()
        input_expl = input_expl.unsqueeze(0)
        input_expl_ori = input_expl
        for i in range(len(data)):
            data_edge = data.copy()
            data_edge[i] = 0
            graph_edge.data = data_edge
            feature_edge = self.get_mamafeatures(graph_edge,sens).detach()
            feature_delta = input_expl_ori-feature_edge
            norm = torch.norm(feature_delta)
            if norm != 0:
                feature_delta = torch.div(feature_delta,torch.norm(feature_delta))
            input_expl = torch.cat((input_expl, feature_delta), 0)
        
        return input_expl.to(torch.float32).cuda()


    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
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
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()).cuda() + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs).cuda()
        else:
            graph = torch.sigmoid(sampling_weights).cuda()

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
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)
        
        if ifsp:
            # Mask loss
            masked_graph = torch.from_numpy(newgraph.toarray())
            true_subgraph = torch.from_numpy(true_subgraph)
            distance = F.pairwise_distance(masked_graph,true_subgraph,p=2)
            p2_loss = torch.sum(distance)
            print("cce_loss:"+str(10*cce_loss))
            print("size_loss:"+str(size_loss*0.1))
            print("mask_ent_loss:"+str(mask_ent_loss*0.5))
            print("p2_loss:"+str(0.2*p2_loss))
            # return 10*cce_loss + 0.5*mask_ent_loss+ 0.2*p2_loss
            return 20*cce_loss + 0.1*size_loss + 0.5*mask_ent_loss+ 0.1*p2_loss
        
        print("cce_loss:"+str(10*cce_loss))
        print("size_loss:"+str(0.1*size_loss))
        print("mask_ent_loss:"+str(mask_ent_loss*0.5))
        return cce_loss + 0.1*size_loss + mask_ent_loss 

    def prepare(self, indices=None,ifsp=False,train=False,model_path=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).cuda()


        if train:
            self.train(indices=indices,ifsp=ifsp)
        else:
            self.explainer_model.load_state_dict(torch.load(model_path))

    def mask_edges(self,graph):
        # 随机mask一部分边
        data_len = len(graph.data)
        random_num =int(data_len*0.2)
        mask_col_del = np.random.choice(data_len,random_num)
        graph.data[mask_col_del]=0
        
        #随机添加一部分边
        rows = graph.row
        cols = graph.col
        graph_mask = graph.toarray()
        mask_col_add = np.random.choice(cols,random_num)
        mask_row_add = np.random.choice(rows,random_num)
        graph_mask[mask_col_add,mask_row_add]=1
        graph_mask = sparse.coo_matrix(graph_mask)
        return graph_mask

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

                for i in range(1):
                    sens = self.sen_api_idx[n]
                    graph = graph_0.copy()
                    
                    features = self.get_mamafeatures(graph_0,sens)

                    # Sample possible explanation
                    input_expl = self._create_explainer_input(graph, sens)
                    sampling_weights = self.explainer_model(input_expl)
                    sampling_weights = sampling_weights.reshape(1,-1).squeeze()
                    print(sampling_weights)
                    mask = self._sample_graph(sampling_weights[1:], t, bias=self.sample_bias)
                    newgraph = graph.copy()
                    newgraph.data = mask.detach().cpu().numpy()

                    mask_embed = self.get_mamafeatures(newgraph,sens)
                    
                    masked_pred = find_nn_torch(mask_embed.cuda().unsqueeze(0), self.X_train_torch, self.y_train_torch,self.k)
                    original_pred = find_nn_torch(features.cuda().unsqueeze(0), self.X_train_torch, self.y_train_torch,self.k)
                    
                    print("original label: "+str(original_pred))
                    print("masked label: "+str(masked_pred))

                    if ifsp:
                        true_subgraph = self.subgraph_dict[n]
                        id_loss = self._loss(masked_pred,newgraph, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs, true_subgraph,ifsp=True)
                    else:
                        id_loss = self._loss(masked_pred,newgraph, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                    loss += id_loss


            print("Epoch:"+str(e)+"\t"+"Total loss:"+str(loss))
            loss.backward()
            optimizer.step()
        self.explainer_model.eval()
        torch.save(self.explainer_model.state_dict(), 'explainer_'+self.expname+'.pkl') 

    def explain(self, graph,sens):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, sens).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights).squeeze()
        print(mask)
        
        explain = []
        rows = graph.row
        cols = graph.col
        for i in range(len(rows)):
            edge_exp = []
            edge_exp.append(rows[i])
            edge_exp.append(cols[i])
            edge_exp.append(mask[i].cpu().detach().numpy())
            explain.append(edge_exp)
        explain.sort(key=lambda x: abs(x[2]), reverse=True)
        explain_edge =np.array([x[:2] for x in explain])
        explain_score = np.array([x[2] for x in explain])

        return explain_edge, explain_score
    
    
    def get_mamafeatures(self,adjs,node_index,state_category="families",ifnodes=False):
        test_adj = adjs
        pack_idx = node_index
        train_feature = self.extract_feature_torch(test_adj, pack_idx, state_category,ifnodes)
        return train_feature


    def extract_feature_torch(self,adj, pack_idx, type,ifnodes=False):
        '''
        adj: csr_matrix: adjacent matrix
        pack_idx: nparray: node number * 1, the package index of each node
        num: package: 11; package: 446
        '''
        if type == "families":
            nn = 11
        else:
            nn = 446
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
                call_relation = idx_one_hot.T.matmul(adj_nodes.matmul(idx_one_hot))

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
            call_relation = idx_one_hot.T.matmul(adjcopy.matmul(idx_one_hot))

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
    
