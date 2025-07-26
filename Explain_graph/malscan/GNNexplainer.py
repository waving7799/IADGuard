import torch
# import torch_geometric as ptgeom
from torch import nn
# from torch.optim import Adam
# from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from BaseExplainer import BaseExplainer
from scipy import sparse
from torch.autograd import Variable
import torch.nn.functional as F
from explain_module import ExplainModule
import time
from scipy.sparse.coo import coo_matrix

class GNNexplainer(BaseExplainer):
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
    def __init__(self, model_to_explain,  graphs, features, task,X_train_torch, y_train_torch, prog_args, epochs=10, lr=0.003, temp=(5.0, 2.0), reg_coefs=(0.05, 1.0),sample_bias=0):
        super().__init__(graphs, X_train_torch, y_train_torch,features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.expl_embedding = 43972
        self.model_to_explain = model_to_explain
        # self.subgraph_dict = subgraph_dict

        self.traindata = X_train_torch
        self.y = y_train_torch
        self.args = prog_args

    def _create_explainer_input(self, pair, embeds):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair.row
        cols = pair.col
        row_embeds = embeds[:,rows]
        col_embeds = embeds[:,cols]
        input_expl =  row_embeds+col_embeds
        return input_expl.T


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
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size()) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph =  torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        return graph


    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
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

        return 0.5*cce_loss + 0.5*size_loss + mask_ent_loss


    # Main method
    def gnn_explain(
        self,name,node_idx=0, graph_idx=0, graph_mode=True, unconstrained=False, model="exp"
    ):
        """Explain a single node prediction
        """
        # index of the query node in the new adj
        if graph_mode:
            node_idx_new = node_idx
            ori_adj = self.graphs[graph_idx]
            sub_adj = self.graphs[graph_idx].toarray()
            sens = self.sen_api_idx[graph_idx]
            sub_label = self.label

        sub_adj = np.expand_dims(sub_adj, axis=0)
        adj   = torch.tensor(sub_adj, dtype=torch.float)
        label = torch.tensor(sub_label, dtype=torch.long)
        pred_label = sub_label


        explainer = ExplainModule(
            ori_adj = ori_adj,
            adj=adj,
            model=self.model_to_explain,
            label=label,
            args=self.args,
            sens_api = sens,
            graph_idx= graph_idx,
            graph_mode=graph_mode,
        )
        explainer = explainer.cuda()


        # gradient baseline
        if model == "grad":
            explainer.zero_grad()
            # pdb.set_trace()
            adj_grad = torch.abs(
                explainer.adj_feat_grad(node_idx_new, pred_label[node_idx_new])[0]
            )[graph_idx]
            masked_adj = adj_grad + adj_grad.t()
            masked_adj = nn.functional.sigmoid(masked_adj)
            masked_adj = masked_adj.cpu().detach().numpy() * sub_adj.squeeze()
        else:
            explainer.train()
            begin_time = time.time()
            for epoch in range(self.epochs):
                explainer.zero_grad()
                explainer.optimizer.zero_grad()
                ypred = explainer(unconstrained=unconstrained)
                loss = explainer.loss(ypred, pred_label, node_idx_new, epoch)
                loss.backward()

                explainer.optimizer.step()
                if explainer.scheduler is not None:
                    explainer.scheduler.step()

                mask_density = explainer.mask_density()
                print(
                    "epoch: ",
                    epoch,
                    "; loss: ",
                    loss.item(),
                    "; mask density: ",
                    mask_density.item(),
                    "; pred: ",
                    ypred,
                )


            print("finished training in ", time.time() - begin_time)
            if model == "exp":
                masked_adj = (
                    explainer.masked_adj[0].cpu().detach().numpy() * sub_adj.squeeze()
                )

        return masked_adj


    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None: # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)


    def explain(self, masked_adj):
        mask_sparse = coo_matrix(masked_adj) 
        explain = []
        rows = mask_sparse.row
        cols = mask_sparse.col
        mask = mask_sparse.data
        for i in range(len(rows)):
            edge_exp = []
            edge_exp.append(rows[i])
            edge_exp.append(cols[i])
            edge_exp.append(mask[i])
            explain.append(edge_exp)
        explain.sort(key=lambda x: abs(x[2]), reverse=True)
        explain_edge =np.array([x[:2] for x in explain])
        explain_score = np.array([x[2] for x in explain])

        return explain_edge, explain_score
    