import torch
import torch.nn as nn
from torch.autograd import Variable
import train_utils as train_utils
import math
import numpy as np
import copy
import torch.nn.functional as F

class ExplainModule(nn.Module):
    def __init__(
        self,
        ori_adj,
        adj,
        model,
        label,
        args,
        sens_api,
        graph_idx=0,
        use_sigmoid=True,
        graph_mode=False,
    ):
        super(ExplainModule, self).__init__()
        self.ori_adj = ori_adj
        self.adj = adj
        self.model = model
        self.label = label
        self.graph_idx = graph_idx
        self.sens = sens_api
        self.args = args
        self.mask_act = args.mask_act
        self.use_sigmoid = use_sigmoid
        self.graph_mode = graph_mode
        # self.subgraph = subgraph

        init_strategy = "normal"
        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(
            num_nodes, init_strategy=init_strategy
        )

        params = [self.mask]
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = train_utils.build_optimizer(args, params)

        self.coeffs = {
            "size": 0.005,
            "feat_size": 1.0,
            "ent": 1.0,
            "feat_ent": 0.1,
            "grad": 0,
            "lap": 1.0,
        }

    def construct_feat_mask(self, feat_dim, init_strategy="normal"):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == "normal":
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
                # mask[0] = 2
        return mask

    def construct_edge_mask(self, num_nodes, init_strategy="normal", const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(
                2.0 / (num_nodes + num_nodes)
            )
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == "const":
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == "sigmoid":
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        adj = self.mask_edges(adj,self.sens)
        # sym_mask = sym_mask.cuda() if self.args.gpu else sym_mask
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum
    
    def to_adjmatrix(self, adj_sparse):
        A = torch.sparse_coo_tensor(adj_sparse[:, :2].T, adj_sparse[:, 2],
                                        size=[adj_sparse.shape[0], adj_sparse.shape[0]]).to_dense()

        return A
            
    def getDegreeCentrality(self, adj, sen_api_idx,ifnode=False):
        
        idx_matrix = np.zeros((len(sen_api_idx), adj.shape[0]))
        ii = np.where(sen_api_idx != -1)
        idx_matrix[ii, sen_api_idx[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix)
        adj_dense = self.to_adjmatrix(adj)
        if not ifnode:
            all_degree = torch.div((torch.sum(adj_dense, 0) + torch.sum(adj_dense, 1)), (adj_dense.shape[0] - 1))
        else:
            all_degree = torch.div(adj_dense, (adj_dense.shape[0] - 1))            
        degree_centrality = torch.matmul(idx_matrix, all_degree.type_as(idx_matrix))
        return degree_centrality
    
    
    def katz_feature_torch(self,graph,sen_api_idx,ifnode=False,alpha=0.1, beta=1.0, normalized=True):
        n = graph.shape[0] 
        graph = graph.T
        
        if ifnode:
            b = torch.ones((n, n)) * float(beta)
            A = torch.eye(n, n).float() - (alpha * graph.float())
            L,U= torch.solve(b, A)
            if normalized:
                norm = torch.sign(sum(L)) * torch.norm(L)
            else:
                norm = 1.0
            centrality = torch.div(L, norm)
        else:
            b = (torch.ones((n, 1)) * float(beta)).cuda()
            A = torch.eye(n, n).float().cuda() - (alpha * graph.float())
            L = torch.linalg.solve(A, b)
            if normalized:
                norm = torch.sign(sum(L)) * torch.norm(L)
            else:
                norm = 1.0
            centrality = torch.div(L, norm)

        idx_matrix = np.zeros((len(sen_api_idx), n))

        ii = np.where(sen_api_idx != -1)
        idx_matrix[ii, sen_api_idx[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix)
        katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
        return katz_centrality
    
    def malscan_feature(self, graph,sen_api_idx):
        graph = graph.squeeze(0)
        feature = self.getDegreeCentrality(graph, sen_api_idx)
        densegraph = self.to_adjmatrix(graph)
        feature_katz = self.katz_feature_torch(densegraph, sen_api_idx)
        feature = torch.cat((feature, np.squeeze(feature_katz)), 0)
        return feature.to(torch.float32)      

    def mask_edges(self,graph,sens_idx):
        ii = np.where(sens_idx != -1)
        ii_idx = sens_idx[ii]
        
        data_len = len(graph.data)
        random_num =int(data_len*0.2)
        perturb_idx = np.random.choice(ii_idx,1)
        
        # 随机mask一部分边
        data_len = len(self.ori_adj.data)
        rows = graph.size(1)
        cols = graph.size(1)
        graph_perturb = copy.copy(graph)
        random_num =int(data_len*0.2)
        
        mask_col_del = np.random.choice(cols,random_num)
        mask_row_del = np.random.choice(rows,random_num)
        graph_perturb[:,mask_col_del,mask_row_del]=0
        
        #随机添加一部分边      
        mask_col_add = np.random.choice(cols,random_num)
        mask_row_add = np.random.choice(rows,random_num)
        graph_perturb[:,perturb_idx,mask_col_add]=1
        graph_perturb[:,mask_row_add,perturb_idx]=1
        return graph_perturb

    def forward(self,unconstrained=False, mask_features=True, marginalize=False):

        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = (
                torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            )
        else:
            self.masked_adj = self._masked_adj()
            
        mask_embed = self.malscan_feature(self.masked_adj,self.sens).detach()
        ypred = self.model.predict_proba(mask_embed.unsqueeze(0))
        # if self.graph_mode:
        #     res = nn.Softmax(dim=0)(ypred[0])
        ypred = torch.from_numpy(ypred)
        return ypred

    def loss(self, pred, pred_label, node_idx, epoch):
        """
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        """
        mi_obj = False
        if mi_obj:
            pred_loss = -torch.sum(pred * torch.log(pred))
        else:
            gt_label_node = self.label if self.graph_mode else self.label[0][node_idx]
            logit = pred[0][gt_label_node]
            pred_loss = -torch.log(logit)
        # size
        mask = self.mask
        if self.mask_act == "sigmoid":
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == "ReLU":
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs["size"] * torch.sum(mask)


        # entropy
        mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = self.coeffs["ent"] * torch.mean(mask_ent)


        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj if self.graph_mode else self.masked_adj[self.graph_idx]
        L = D - m_adj
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        if self.graph_mode:
            lap_loss = 0
        else:
            lap_loss = (self.coeffs["lap"]
                * (pred_label_t @ L @ pred_label_t)
                / self.adj.numel()
            )
        print("cce_loss:"+str(10*pred_loss))
        print("size_loss:"+str(0.1*size_loss))
        print("mask_ent_loss:"+str(mask_ent_loss*0.5))
        loss = 10*pred_loss + 0.1*size_loss + mask_ent_loss*0.5

        return loss
