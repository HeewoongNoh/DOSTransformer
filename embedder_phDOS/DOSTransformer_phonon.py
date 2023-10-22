import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from layers import TransformerEncoder
from torch_scatter import scatter_mean, scatter_sum
from e3nn.nn.models.gate_points_2101 import smooth_cutoff
from e3nn import o3
from torch_cluster import radius_graph
############################################################################################################################
## DOSTransformer for phononDoS
############################################################################################################################
class DOSTransformer_phonon(nn.Module):
    def __init__(self, layers, t_layers, n_atom_feats, n_bond_feats, n_hidden, device, attn_drop):
        super(DOSTransformer_phonon, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(51, n_hidden) 
        # Prompt embeddings
        self.prompt_token = nn.Embedding(7, n_hidden//2)
        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])
        
        # Cross attention & Self attention
        self.transformer = TransformerEncoder(embed_dim = n_hidden,
                                        num_heads = 1,
                                        layers = t_layers, attn_dropout=attn_drop)
        
        self.transformer_self = TransformerEncoder(embed_dim = n_hidden,
                                num_heads = 1,
                                layers = t_layers, attn_dropout=attn_drop)
        
        self.transformer_source = TransformerEncoder(embed_dim = n_hidden,
                                num_heads = 1,
                                layers = t_layers, attn_dropout=attn_drop)
        
        self.GN_decoder = Decoder(n_hidden)
        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Linear(n_hidden, 1)
        self.fc = nn.Linear(n_hidden*2,n_hidden)
        self.fc_prompt = nn.Linear((n_hidden*2 + n_hidden//2),n_hidden)
        self.device = device

    def preprocess(self, data):
        
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0], dtype=torch.long)

        if 'edge_index' in data:
            edge_src = data['edge_index'][0]  # edge source
            edge_dst = data['edge_index'][1]  # edge destination
            edge_vec = data['edge_vec']
        
        else:
            edge_index = radius_graph(data['pos'], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data['pos'][edge_src] - data['pos'][edge_dst]

        return edge_vec
    
    def forward(self, g):

        input_ids = torch.tensor(np.arange(51)).to(self.device)
        token_ids = torch.tensor(np.arange(7)).to(self.device)

        energies = self.embeddings(input_ids)
        prompt_token = self.prompt_token(token_ids)

        edge_vec = self.preprocess(g)
        edge_sh = o3.spherical_harmonics(o3.Irreps.spherical_harmonics(1), edge_vec, True, normalization='component')
        edge_length = edge_vec.norm(dim=1)
        edge_attr = smooth_cutoff(edge_length / 4.)[:, None] * edge_sh
        
        x, edge_attr, energies = self.GN_encoder(x = g.x, edge_attr = edge_attr, batch = g.batch, energies = energies)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr

        x_dense, _ = to_dense_batch(x, batch = g.batch)
        x_dense = x_dense.transpose(0, 1)
        energies = self.transformer(energies, x_dense, x_dense)  
        
        graph = self.GN_decoder(x, g.batch)  
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(51,graph.shape[0], graph.shape[1])

        dos_input  = torch.cat([energies, graph], dim=2 )

        dos_input = F.leaky_relu(self.fc(dos_input)) 

        dos_global = self.transformer_self(dos_input, dos_input, dos_input)

        dos_global = self.transformer_source(dos_global, x_dense, x_dense)

        dos_global = self.out_layer(dos_global)  
   
        dos_global = dos_global.squeeze(2).T     

        prompt_token = prompt_token[g.system].repeat(51,1,1) 
    
        dos_input  = torch.cat([energies, graph, prompt_token], dim=2 )  

        dos_input = F.leaky_relu(self.fc_prompt(dos_input)) 

        dos_system = self.transformer_self(dos_input, dos_input, dos_input)

        dos_system = self.transformer_source(dos_system, x_dense, x_dense) 

        dos_system = self.out_layer(dos_system) 
   
        dos_system = dos_system.squeeze(2).T 
        
        return dos_global, x, dos_system


############################################################################################################################
## Graph Neural Network
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
        self.mlp = nn.Sequential(nn.Linear(n_hidden, n_hidden))
    def forward(self, x, batch):
        
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
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, n_hidden):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))
        self.node_mlp_2 = nn.Sequential(nn.Linear(n_hidden*2, n_hidden*2), nn.LayerNorm(n_hidden*2), nn.PReLU(), nn.Linear(n_hidden*2, n_hidden))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index

        out = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0)) 
        out = torch.cat([x, out], dim=1)

        return self.node_mlp_2(out)