import torch
from typing import Optional, Union
from torch.nn import Parameter
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros



class GraphLayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int=1,
        concat: bool=True,
        negative_slope: float=0.2,
        dropout: float=0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool=True,
        share_weights: bool = True,
        **kwargs
    ):

        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self._alpha = None

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')
        if self.share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.att_em)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):

        if torch.is_tensor(x):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        else:
            x_l, x_r = x[0], x[1]
            assert x_l.dim() == 2
            x_l = self.lin_l(x_l).view(-1, self.heads, self.out_channels)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, self.heads, self.out_channels)

        x = (x_l, x_r)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=num_nodes)

        out = self.propagate(
            edge_index,
            x=x,
            embedding=embedding,
            edges=edge_index,
            return_attention_weights=return_attention_weights,
        )

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out

    def message(
        self, x_i, x_j, edge_index_i, edge_index_j, size_i, embedding, edges, return_attention_weights
    ):

        if embedding is not None:
            embedding_i = embedding[edge_index_i].unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding[edges[0]].unsqueeze(1).repeat(1, self.heads, 1)
            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att = torch.cat((self.att, self.att_em), dim=-1)
        key = key_i + key_j
        key = F.leaky_relu(key, self.negative_slope)

        alpha = (key * cat_att).sum(dim=-1)
        alpha = softmax(alpha, index=edge_index_i, num_nodes=size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"
