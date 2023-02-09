import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

from layers import TransformerEncoder
from torch_scatter import scatter_mean, scatter_sum

# GNN(encoder + decoder) before applying Transformer
class graph_encoder(nn.Module):
    def __init__(self, layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):
        """
        Construct a MulT model.
        """
        super(graph_encoder, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(201, n_hidden)

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])
        self.GN_decoder = Decoder(n_hidden)

        self.device = device


        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
        self.spark_layer = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Sigmoid())


    def forward(self, g):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        input_ids = torch.tensor(np.arange(201)).to(self.device)
        energies = self.embeddings(input_ids)

        x, edge_attr, glob, energies = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch, energies = energies)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr
        
        x_dense, _ = to_dense_batch(x, batch = g.batch)
        x_dense = x_dense.transpose(0, 1)
        
        # energies = self.transformer(energies, x_dense, x_dense)
        padding = torch.zeros(1, energies.shape[1], energies.shape[2]).to(self.device)

        energies1 = torch.cat([energies, padding], dim = 0)
        energies2 = torch.cat([padding, energies], dim = 0)
        pred = torch.cat([energies1, energies2], dim = 2)[1: -1]
        spark = self.spark_layer(pred)
        
        graph = self.GN_decoder(x, glob, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(201, graph.shape[0], graph.shape[1])
        dos = self.out_layer(energies + self.alpha * graph)
        # dos = self.out_layer(energies)
        dos = dos.squeeze(2).T

        return dos, x, spark

## Add before decoder
class GNTransformer_joint2(nn.Module):
    def __init__(self, layers, t_layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):
        """
        Construct a MulT model.
        """
        super(GNTransformer_joint2, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(201, n_hidden)

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])

        # Crossmodal Attentions
        self.transformer = TransformerEncoder(embed_dim = n_hidden,
                                            num_heads = 1,
                                            layers = t_layers)
        
        self.GN_decoder = Decoder(n_hidden)
        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))

        self.spark_layer1 = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.spark_layer2 = nn.Sequential(nn.Linear(n_hidden, 1))
        
        self.sigmoid = nn.Sigmoid()
        
        self.device = device

    def forward(self, g):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        input_ids = torch.tensor(np.arange(201)).to(self.device)
        energies = self.embeddings(input_ids)

        x, edge_attr, glob, energies = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch, energies = energies)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr
        
        x_dense, _ = to_dense_batch(x, batch = g.batch)
        x_dense = x_dense.transpose(0, 1)
        
        energies = self.transformer(energies, x_dense, x_dense)    #[201, 8, 64]

        
        # default window_size:2 
        # padding1 = torch.zeros(1, energies.shape[1], energies.shape[2]).to(self.device)
        # padding2 = torch.zeros(1, energies.shape[1], 1).to(self.device)
        
        # energies1 = torch.cat([energies, padding1], dim = 0)
        # energies2 = torch.cat([padding1, energies], dim = 0)

        # pred1= torch.cat([energies1, energies2], dim = 2)[1: -1]
        # spark1 = self.spark_layer1(pred1)   

        graph = self.GN_decoder(x, glob, g.batch)
        graph = graph.reshape(-1, graph.shape[0], graph.shape[1]).expand(201, graph.shape[0], graph.shape[1])
        
        dos = self.out_layer(energies + self.alpha * graph)   #[201,8,1]    
        
        spark_embed = energies[:-1]
        spark = self.spark_layer1(spark_embed)
        # energies3 = torch.cat([dos, padding2], dim = 0)
        # energies4 = torch.cat([padding2, dos], dim = 0)
        # spark2 = torch.abs(torch.subtract(energies3, energies4))[1:-1]
        # spark2 = torch.where(spark2 < 0.1, -10*spark2, 10*spark2)
        # spark2 = self.sigmoid(spark2)
        # output = torch.abs(torch.subtract(energies2, energies1))[1:-1]
        # output = torch.where(output < 0.01, -1000*(output),100*(output))
        # output = self.sigmoid(output)
        # output = torch.where(output >= 0.5, 10*(output-0.5), output)
        # output = self.relu(output)
        dos = dos.squeeze(2).T      #[8, 201]
        return dos, x, spark
    
    
class GNTransformer_joint(nn.Module):
    def __init__(self, layers, t_layers, n_atom_feats, n_bond_feats, n_glob_feats, n_hidden, dim_out, device):
        """
        Construct a MulT model.
        """
        super(GNTransformer_joint, self).__init__()

        # Energy embeddings
        self.embeddings = nn.Embedding(201, n_hidden)

        # Graph Neural Network
        self.GN_encoder = Encoder(n_atom_feats, n_bond_feats, n_glob_feats, n_hidden)
        self.stacked_processor = nn.ModuleList([Processor(EdgeModel(n_hidden), NodeModel(n_hidden)) for i in range(layers)])

        # Crossmodal Attentions
        self.transformer = TransformerEncoder(embed_dim = n_hidden,
                                            num_heads = 1,
                                            layers = t_layers)
        
        self.GN_decoder = Decoder(n_hidden)
        self.alpha = nn.Parameter(torch.rand(1))
        self.out_layer = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, 1))
        self.spark_layer = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.Sigmoid())
        # self.spark_layer = nn.Sequential(nn.Linear(n_hidden * 2, 1), nn.ReLU())
        self.device = device

    def forward(self, g):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        input_ids = torch.tensor(np.arange(201)).to(self.device)
        energies = self.embeddings(input_ids)

        x, edge_attr, glob, energies = self.GN_encoder(x = g.x, edge_attr = g.edge_attr, glob = g.glob, batch = g.batch, energies = energies)

        for processor in self.stacked_processor:
            out_x, out_edge_attr = processor(x = x, edge_index = g.edge_index, edge_attr = edge_attr)
            x = x + out_x
            edge_attr = edge_attr + out_edge_attr
        
        x_dense, _ = to_dense_batch(x, batch = g.batch)
        x_dense = x_dense.transpose(0, 1)
        
        energies = self.transformer(energies, x_dense, x_dense)    #[201, 8, 64]
        # print(f'energies:{energies.shape}')
        padding = torch.zeros(1, energies.shape[1], energies.shape[2]).to(self.device)
        energies1 = torch.cat([energies, padding], dim = 0)
        energies2 = torch.cat([padding, energies], dim = 0)
        # original_pred = torch.cat([energies1, energies2], dim = 2)
        pred = torch.cat([energies1, energies2], dim = 2)[1: -1]
        spark = self.spark_layer(pred)
        
        graph_decoded = self.GN_decoder(x, glob, g.batch)
        graph = graph_decoded.reshape(-1, graph_decoded.shape[0], graph_decoded.shape[1]).expand(201, graph_decoded.shape[0], graph_decoded.shape[1])
        dos = self.out_layer(energies + self.alpha * graph)   #[201,8,1]
        # print(f'dos_shape:{dos.shape}')     
        dos = dos.squeeze(2).T      #[8, 201]
        # print(f'transposed_dos:{dos.shape}')
        return dos,x, graph_decoded


############################################################################################################################
## Graph Neural Network
############################################################################################################################

class Encoder(nn.Module):
    def __init__(self, n_atom_feats, n_bond_feats, n_global_feats, n_hidden):
        super(Encoder, self).__init__()
        self.node_encoder = nn.Sequential(nn.Linear(n_atom_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.edge_encoder = nn.Sequential(nn.Linear(n_bond_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.global_encoder = nn.Sequential(nn.Linear(n_global_feats, n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
        self.reset_parameters()
        
    
    def reset_parameters(self):
        for item in [self.node_encoder, self.edge_encoder]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
    
    def forward(self, x, edge_attr, glob, batch, energies):
        
        x = self.node_encoder(x)
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
        # self.mlp = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden), nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden, n_hidden))
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