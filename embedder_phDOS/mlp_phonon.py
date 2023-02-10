import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_sum
from e3nn.nn.models.gate_points_2101 import smooth_cutoff

############################################################################################################################
## mlp for phDoS
############################################################################################################################

class mlp_phonon(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_hidden, dim_out, device):
        super(mlp_phonon, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(51, n_hidden)
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_hidden)
        self.GN_decoder = Decoder(n_hidden)
        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden//2), nn.LayerNorm(n_hidden//2), nn.PReLU(), nn.Linear(n_hidden//2, 1))
        self.device = device

    def forward(self, g):

        input_ids = torch.tensor(np.arange(51)).to(self.device)
        energies = self.embeddings(input_ids)

        edge_length = g.edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.)[:, None] * g.edge_shift
        x, _, energies = self.GN_encoder(x = g.x, edge_attr = edge_attr, batch = g.batch, energies = energies)

        graph = self.GN_decoder(x, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(51, graph.shape[0], graph.shape[1])
        dos = self.out_layer(energies + self.alpha * graph)    
        dos = dos.squeeze(2).T      

        return dos

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
    
    def forward(self, x, edge_attr, batch, energies):
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        energies = energies.reshape(energies.shape[0], 1, energies.shape[1]).expand(energies.shape[0], len(batch.unique()), energies.shape[1])

        return x, edge_attr,energies



class Decoder(nn.Module):
    def __init__(self, n_hidden):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden))
    def forward(self, x, batch):
        
        output = scatter_sum(x, batch, dim = 0)
        output = self.mlp(output)

        return output

