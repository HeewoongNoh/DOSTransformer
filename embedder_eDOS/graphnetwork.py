import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

############################################################################################################################
## Graphnetwork Energy embedding
############################################################################################################################
class Graphnetwork(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):

        super(Graphnetwork, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(201, n_hidden)

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])
        self.GN_decoder = Decoder(n_hidden)
        self.device = device
        self.out_layer = nn.Sequential(nn.Linear(n_hidden*2,n_hidden), nn.LeakyReLU(), nn.Linear(n_hidden,1))


    def forward(self, g):

        input_ids = torch.tensor(np.arange(201)).to(self.device)
        energies = self.embeddings(input_ids)

        x, edge_attr, glob, energies = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch, energies = energies)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr
        
        graph = self.GN_decoder(x, glob, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(201, graph.shape[0], graph.shape[1])
        dos = self.out_layer(torch.cat([energies, graph], dim=2))
        dos = dos.squeeze(2).T

        return dos, x

############################################################################################################################
## Graphnetwork without Energy embedding
############################################################################################################################
class Graphnetwork2(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):

        super(Graphnetwork2, self).__init__()

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])
        self.GN_decoder = Decoder(n_hidden)
        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Sequential(nn.Linear(n_hidden*2, n_hidden), nn.LeakyReLU(), nn.Linear(n_hidden, 201))      
        self.device = device

    def forward(self, g):


        x, edge_attr, glob = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob)   
        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr

        dos_input = scatter_sum(x, g.batch, dim=0)
        dos_input = torch.cat([dos_input, glob], dim=1)
        dos = self.out_layer(dos_input)
   
        return dos, x
    
############################################################################################################################
## Graph Neural Network
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
        self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden))
    
    def forward(self, x, glob, batch):
        
        glob = torch.cat([glob, scatter_sum(x, batch, dim=0)], dim = 1)
        glob = self.mlp(glob)
        
        return glob


############################################################################################################################
## Basic Building Blocks
############################################################################################################################

class EdgeModel(nn.Module):
    def __init__(self, n_hidden):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(n_hidden*3, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))

    def forward(self, src, dest, edge_attr):
        out = torch.cat([src, dest, edge_attr], 1) 
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, n_hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))
        self.node_mlp_2 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))

    def forward(self, x, edge_index, edge_attr):

        row, col = edge_index

        out = scatter_sum(edge_attr, col, dim=0, dim_size=x.size(0)) 
        out = torch.cat([x, out], dim=1)

        return self.node_mlp_2(out)



