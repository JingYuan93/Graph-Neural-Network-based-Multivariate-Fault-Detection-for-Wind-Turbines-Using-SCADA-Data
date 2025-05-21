import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.env import get_device
from models.graph_layer import GraphLayer

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()
    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num
    return batch_edge_index.long()

class OutLayer(nn.Module):
    def __init__(self, in_num, layer_num, inter_num, output_dim):
        super(OutLayer, self).__init__()
        modules = []
        for i in range(layer_num):
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, output_dim))
            else:
                layer_in = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)
        return out

class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, heads=1):
        super(GNNLayer, self).__init__()
        self.gnn = GraphLayer(in_channel, out_channel, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        np.save('att_weight_1.npy', att_weight.detach().cpu().numpy())
        np.save('edge_index_1.npy', new_edge_index.detach().cpu().numpy())
        out = self.bn(out)
        return self.relu(out)

class GRN(nn.Module):
    def __init__(self, edge_index_sets, node_num, config, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super(GRN, self).__init__()
        self.edge_index_sets = edge_index_sets
        self.config = config
        device = get_device()
        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([GNNLayer(input_dim, dim, heads=1) for _ in range(edge_set_num)])
        self.lin = nn.Linear(dim, config["slide_win"] // 2)
        if config["loss_type"] == "pred":
            self.out_layer = OutLayer(dim * edge_set_num, out_layer_num, inter_num=out_layer_inter_dim, output_dim=1)
        else:
            self.out_layer = OutLayer(config["slide_win"] // 2, out_layer_num, inter_num=out_layer_inter_dim, output_dim=config["slide_win"])
        self.topk = topk
        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        x = x.reshape(-1, all_feature)
        gcn_outs = []
        for edge_index in self.edge_index_sets:
            all_embeddings = self.embedding(torch.arange(node_num).to(data.device))
            weights = all_embeddings.detach().clone()
            all_embeddings = weights.repeat(batch_num, 1)
            cos_mat = torch.matmul(weights, weights.T)
            norms = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_mat = cos_mat / norms
            topk_num = self.topk
            topk_idx = torch.topk(cos_mat, topk_num, dim=-1)[1]
            gated_i = torch.arange(node_num).unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0).to(data.device)
            gated_j = topk_idx.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            batch_edge = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(data.device)
            gcn_outs.append(self.gnn_layers[0](x, batch_edge, embedding=all_embeddings))
        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)
        idx = torch.arange(node_num).to(data.device)
        out = x * self.embedding(idx)
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)
        out = self.lin(out)
        out = self.out_layer(out)
        if self.config["loss_type"] == "pred":
            out = out.view(-1, node_num)
        return out
