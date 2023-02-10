import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_sum
from e3nn.nn.models.gate_points_2101 import smooth_cutoff

############################################################################################################################
## mlp2 for phDoS (not use Energy embedding)
############################################################################################################################

class mlp2_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, dim_out, device):
        super(mlp2_phonon, self).__init__()
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_hidden)
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 51))
        self.device = device

    def forward(self, g):

        edge_length = g.edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.)[:, None] * g.edge_shift
        x, _  = self.GN_encoder(x = g.x, edge_attr = edge_attr, batch = g.batch)

        dos_vector1 = self.out_layer(x)
        dos_vector3 = scatter_sum(dos_vector1, g.batch, dim=0)

        return dos_vector3
############################################################################################################################
## Encoder
############################################################################################################################

class Encoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats,n_hidden):
        super(Encoder, self).__init__()
        self.node_encoder = nn.Sequential(nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_attr, batch):
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)


        return x, edge_attr



class Decoder(nn.Module):
    def __init__(self, n_hidden):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden))
    def forward(self, x, batch):
        
        output = scatter_sum(x, batch, dim = 0)
        output = self.mlp(output)

        return output

