import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

from layers import TransformerEncoder
from torch_scatter import scatter_mean, scatter_sum



############################################################################################################################
## GraphNetwork2 for phononDoS
############################################################################################################################
from e3nn.nn.models.gate_points_2101 import smooth_cutoff

class Graphnetwork2_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, dim_out, device):
        super(Graphnetwork2_phonon, self).__init__()

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 51))
        self.device = device

    def forward(self, g):

        # Added
        edge_length = g.edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.)[:, None] * g.edge_shift
        x, edge_attr = self.GN_encoder(x = g.x, edge_attr = edge_attr, batch = g.batch)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr

        dos_vector1 = self.out_layer(x)
        dos_vector3 = scatter_sum(dos_vector1, g.batch, dim=0)

        return dos_vector3

############################################################################################################################
## mlp2 for phononDoS
############################################################################################################################
from e3nn.nn.models.gate_points_2101 import smooth_cutoff

class mlp2_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, dim_out, device):
        super(mlp2_phonon, self).__init__()

        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_hidden)
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 51))
        self.device = device

    def forward(self, g):

        # Added
        edge_length = g.edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.)[:, None] * g.edge_shift
        x, edge_attr  = self.GN_encoder(x = g.x, edge_attr = edge_attr, batch = g.batch)

        dos_vector1 = self.out_layer(x)
        dos_vector3 = scatter_sum(dos_vector1, g.batch, dim=0)

        return dos_vector3
############################################################################################################################
## Graph Neural Network
############################################################################################################################

class Encoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats,n_hidden):
        super(Encoder, self).__init__()
        self.node_encoder = nn.Sequential(nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        # self.global_encoder = nn.Sequential(nn.Linear(n_global_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_attr, batch):
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        # energies = energies.reshape(energies.shape[0], 1, energies.shape[1]).expand(energies.shape[0], len(batch.unique()), energies.shape[1])
        # glob = glob.reshape(-1, 2)
        # u = self.global_encoder(glob)

        return x, edge_attr


class Processor(nn.Module):
    def __init__(self, edge_model = None, node_model = None):
        super(Processor, self).__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()
    
    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr)
        
        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)
        
        return x, edge_attr


class Decoder(nn.Module):
    def __init__(self, n_hidden):
        super(Decoder, self).__init__()
        # self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        # self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden))
        self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden))
    def forward(self, x, batch):
        
        # glob = torch.cat([glob, scatter_sum(x, batch, dim=0)], dim = 1)
        # glob = self.mlp(glob)
        output = scatter_sum(x, batch, dim = 0)
        output = self.mlp(output)
        
        return output


############################################################################################################################
## Basic Building Blocks
############################################################################################################################

class EdgeModel(nn.Module):
    def __init__(self, n_hidden):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(n_hidden*3, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))

    def forward(self, src, dest, edge_attr):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1) # u.shape(16, 201, 128) else.shape(34502, 128)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, n_hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))
        self.node_mlp_2 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        # torch_scatter.scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0)
        # averages all values from src into out at the indices specified in the index
        out = scatter_sum(edge_attr, col, dim=0, dim_size=x.size(0)) 
        out = torch.cat([x, out], dim=1)

        return self.node_mlp_2(out)