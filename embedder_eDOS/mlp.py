import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

############################################################################################################################
## MLP with Energy embedding
############################################################################################################################

class mlp(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):
        super(mlp, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(201, n_hidden)
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.GN_decoder = Decoder(n_hidden)
        self.device = device
        self.out_layer = nn.Sequential(nn.Linear(n_hidden*2,n_hidden), nn.LeakyReLU(), nn.Linear(n_hidden,1))

    def forward(self, g):

        input_ids = torch.tensor(np.arange(201)).to(self.device)
        energies = self.embeddings(input_ids)

        x, _, glob, energies = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch, energies = energies)
        
        graph = self.GN_decoder(x, glob, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(201, graph.shape[0], graph.shape[1])
        dos_input = torch.cat([energies, graph], dim=2)
        dos = self.out_layer(dos_input)
        dos = dos.squeeze(2).T

        return dos
    
############################################################################################################################
## MLP without Energy embedding
############################################################################################################################

class mlp2(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):
        super(mlp2, self).__init__()

        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.device = device
        self.out_layer = nn.Sequential(nn.Linear(n_hidden*2, n_hidden), nn.LeakyReLU(), nn.Linear(n_hidden, 201))      

    def forward(self, g):

        x, _, glob = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch)
        dos_input = scatter_sum(x, g.batch, dim=0) 
        dos_input = torch.cat([dos_input, glob], dim=1)
        dos = self.out_layer(dos_input)

        return dos
############################################################################################################################
## Encdoer & Decoder
############################################################################################################################

class Encoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, n_global_feats, n_hidden):
        super(Encoder, self).__init__()
        self.node_encoder = nn.Sequential(nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.node_encoder_prompt = nn.Sequential(nn.Linear(n_atom_feats+n_hidden//2, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.global_encoder = nn.Sequential(nn.Linear(n_global_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_attr, glob, batch, energies):
        
        if x.shape[1] == 200:
            x = self.node_encoder(x)
        else:
            x = self.node_encoder_prompt(x)

        edge_attr = self.edge_encoder(edge_attr)
        energies = energies.reshape(energies.shape[0], 1, energies.shape[1]).expand(energies.shape[0], len(batch.unique()), energies.shape[1])
        glob = glob.reshape(-1, 2)
        u = self.global_encoder(glob)

        return x, edge_attr, u, energies

class Decoder(nn.Module):
    def __init__(self, n_hidden):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden))
    
    def forward(self, x, glob, batch):
        
        glob = torch.cat([glob, scatter_sum(x, batch, dim=0)], dim = 1)
        glob = self.mlp(glob)
        
        return glob




