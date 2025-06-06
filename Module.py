import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from einops import rearrange
from mamba import MambaConfig, Mamba
import torch.nn.functional as F
class T_convolution(nn.Module):
    def __init__(self, kernel_size, inc):
        super(T_convolution, self).__init__()
        self.num_windows = inc
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            in_channels=self.num_windows,
            out_channels=self.num_windows,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0,0),  #
            groups=self.num_windows,
            bias=False
        )
        self.bn = nn.BatchNorm2d(self.num_windows)
        self.elu = nn.ELU()
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        self.bn.reset_parameters()

    def forward(self, x):
        B, W, N, T = x.shape
        x = F.pad(x, pad=( self.kernel_size - 1,0, 0, 0))

        x = self.conv(x)  # (B, W, N, T)
        x = self.bn(x)
        x = self.elu(x)
        return x  # B, W, N, T)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,nums_graph,in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.nums_graph = nums_graph
        self.in_features = in_features
        self.out_features = out_features
        self.linear_layer = nn.Linear(in_features, out_features)
        self.elu = nn.ReLU()
        self.weight = Parameter(torch.Tensor(nums_graph,out_features, out_features))
        # self.mlp = nn.Sequential(nn.Linear(out_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU())
        self.dropout = nn.Dropout(0.3)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def get_degree_matrix(self,adj):
        """
        get the degree matrix
        :param adj: B, N, N
        :return: B, N, N
        """
        batch_size, num_nodes, _ = adj.shape
        degrees = torch.sum(adj, dim=2)  # (B, N)
        degrees_inv = torch.where(degrees > 0, 1.0 / degrees, torch.zeros_like(degrees))  # (B, N)
        degree_matrix = torch.diag_embed(degrees_inv)  # (B, N, N)
        return degree_matrix
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #input:b w n t
        #adj: (b w) n t
        batch_size, num_windows,num_nodes,num_ftrs = input.shape[:4]
        input =  rearrange(input, 'b w n c -> (b w) n c', w=num_windows, b=batch_size, n=num_nodes)
        # input_linear = self.linear_layer(input) #(b w) n out_features
        degree_matrix = self.get_degree_matrix(adj)
        support1 = torch.bmm(degree_matrix,adj)
        support2 = torch.bmm(support1, self.elu(input))  # (b w) n out_features
        weight_all = self.weight.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (b, w, n, n)
        weight_all = rearrange(weight_all, 'b w n c -> (b w) n c', w=num_windows, b=batch_size)
        output = torch.bmm(support2, weight_all)  #(b w) n out_features
        output = self.elu(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class DynamicGraphConstruction(nn.Module):
    def __init__(self,):
        super(DynamicGraphConstruction, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        self.eps = nn.Parameter(torch.FloatTensor(1))
        self.reset_parameters()
        self.edgedrop =  nn.Dropout(0.3)

    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def dropedge(self,dynamic_ftr_map, a_percent):
        a_percent = max(0, min(100, a_percent))
        b, n, _ = dynamic_ftr_map.shape
        result = dynamic_ftr_map.clone()
        for i in range(b):
            matrix = result[i]
            total_elements = matrix.numel()
            num_to_zero = int(total_elements * a_percent / 100)
            if num_to_zero == 0:
                continue
            flat_matrix = matrix.view(-1)
            _, indices = torch.abs(flat_matrix).sort()
            indices_to_zero = indices[:num_to_zero]
            flat_matrix[indices_to_zero] = 0
            result[i] = flat_matrix.view(n, n)
        return result

    def pearson_correlation(self,x):
        """
        Compute the Pearson correlation matrix for the last two dimensions of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (b, n, c).

        Returns:
            torch.Tensor: Pearson correlation matrix of shape (b, n, n).
        """
        b, n, c = x.size()
        x_mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - x_mean
        covariance = torch.matmul(x_centered, x_centered.transpose(-1, -2))
        covariance = covariance / (c - 1)
        std = torch.sqrt(torch.var(x, dim=-1, unbiased=True, keepdim=True))
        std_matrix = std * std.transpose(-1, -2)
        correlation = covariance / std_matrix
        correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        return correlation
    def forward_construct_dynamic_ftr_map(self, x):
        #x b w n h
        # minibatch_size, num_windows, num_nodes = x.shape[:3]
        x_glb = self.gap(rearrange(x, 'b w n h -> (b w) n h')).squeeze(-1)  # (b w) n
        x_max = self.max(rearrange(x, 'b w n h -> (b w) n h')).squeeze(-1)  # (b w) n
        adj = torch.bmm(x_glb.unsqueeze(-1), x_max.unsqueeze(-1).transpose(1, 2))
        dynamic_ftr_map = torch.sigmoid(adj)
        #edge dropout
        dynamic_ftr_map = self.dropedge(dynamic_ftr_map,30)
        return dynamic_ftr_map

    def forward(self, x):
        #x: b w n h
        dynamic_ftr_map = self.forward_construct_dynamic_ftr_map(x) #(b w) n n
        x = rearrange(x, 'b w n h -> (b w) n h')
        return x,dynamic_ftr_map
class ModuleMeanReadout(nn.Module):
    """
    A simple mean readout module.
    """
    def __init__(self):
        super(ModuleMeanReadout, self).__init__()

    def forward(self, x, node_axis=1):
        return x.mean(node_axis), torch.zeros(size=[1, 1, 1], dtype=torch.float32)

class BARO(nn.Module):
    """
    BARO (Brain-regions Attention Readout) module with attention mechanism for brain regions.
    """
    def __init__(self, hidden_dim, input_dim,topk, upscale=1.0):
        super(BARO, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, round(upscale * hidden_dim)),
            nn.BatchNorm1d(round(upscale * hidden_dim)),
            nn.GELU()
        )
        self.attend = nn.Linear(round(upscale * hidden_dim), input_dim)
        self.topk = topk
        self.linear = nn.Sequential(nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden_dim * topk, hidden_dim))

    def forward(self, x):
        # x: b w n h
        x_readout1 = x.mean(2)  #b w
        x_readout2 = x.max(2) #b w
        x_readout = x_readout1 + x_readout2.values #b w
        x_shape = x_readout.shape
        x_embed = self.embed(x_readout.reshape(-1, x_shape[-1])) #b w h
        x_graphattention = torch.sigmoid(self.attend(x_embed)).view(*x_shape[:-1], -1) #b w n
        x_attention_applied = x * x_graphattention.unsqueeze(-1)
        output = x_attention_applied.mean(2)
        b,w, n, h = x_attention_applied.shape
        topk_indices = torch.topk(x_graphattention, self.topk, dim=-1).indices  # (b, w, k)
        selected_features = torch.gather(x_attention_applied, dim=2,index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, h))  # (b, w, k, h)
        print("top10-ROIs",topk_indices[0,0,:])
        selected_features = selected_features.sum(2)  # (b, w,h)

        return output,selected_features, x_graphattention.permute(1, 0, 2)

class ModuleMamba(nn.Module):
    def __init__(self, input_dim, hidden_dim,n_layers):
        super().__init__()
        self.config = MambaConfig(d_model=hidden_dim, n_layers=n_layers)
        self.mamba = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Mamba(self.config),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        x = self.mamba(x)
        return x
