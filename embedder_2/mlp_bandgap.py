import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class mlp_bandgap(nn.Module):
    def __init__(self,  n_hidden, device):
        super(mlp_bandgap, self).__init__()
        self.device = device
        self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))

        
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap
        
        
class mlp_bandgap_test(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_bandgap_test, self).__init__()
        self.device = device
        self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.to(self.device)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap
    
class mlp_efermi(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_efermi, self).__init__()
        self.device = device
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi
    
class mlp_efermi_test(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_efermi_test, self).__init__()
        self.device = device
        self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))

    def forward(self, g):
        
        input_dos = g.to(self.device)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi


class mlp_efermi2(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_efermi2, self).__init__()
        self.device = device
        self.linear = nn.Sequential(nn.Linear(201, n_hidden*2),nn.LayerNorm(n_hidden*2), nn.PReLU(),\
        nn.Linear(n_hidden*2, n_hidden),nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden,n_hidden//2),nn.LayerNorm(n_hidden//2), nn.PReLU(),\
        nn.Linear(n_hidden//2, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))

    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi
    
class mlp_efermi_test2(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_efermi_test2, self).__init__()
        self.device = device
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        self.linear = nn.Sequential(nn.Linear(201, n_hidden*2),nn.LayerNorm(n_hidden*2), nn.PReLU(),\
        nn.Linear(n_hidden*2, n_hidden),nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden,n_hidden//2),nn.LayerNorm(n_hidden//2), nn.PReLU(),\
        nn.Linear(n_hidden//2, 1))

    def forward(self, g):
        
        input_dos = g.to(self.device)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi



class mlp_bandgap2(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_bandgap2, self).__init__()
        self.device = device
        self.linear = nn.Sequential(nn.Linear(201, n_hidden*2),nn.LayerNorm(n_hidden*2), nn.PReLU(),\
        nn.Linear(n_hidden*2, n_hidden),nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden,n_hidden//2),nn.LayerNorm(n_hidden//2), nn.PReLU(),\
        nn.Linear(n_hidden//2, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))

    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap
    
class mlp_bandgap_test2(nn.Module):
    def __init__(self, n_hidden, device):
        super(mlp_bandgap_test2, self).__init__()
        self.device = device
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        # self.linear = nn.Sequential(nn.Linear(201, n_hidden*2), nn.ReLU(),nn.Linear(n_hidden*2, n_hidden), nn.ReLU(), nn.Linear(n_hidden, 1))
        self.linear = nn.Sequential(nn.Linear(201, n_hidden*2),nn.LayerNorm(n_hidden*2), nn.PReLU(),\
        nn.Linear(n_hidden*2, n_hidden),nn.LayerNorm(n_hidden), nn.PReLU(), nn.Linear(n_hidden,n_hidden//2),nn.LayerNorm(n_hidden//2), nn.PReLU(),\
        nn.Linear(n_hidden//2, 1))

    def forward(self, g):
        
        input_dos = g.to(self.device)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap














'''
class mlp_bandgap(nn.Module):
    def __init__(self,  device):
        super(mlp_bandgap, self).__init__()
        self.device = device
        self.linear = nn.Linear(201, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.linear.weight,std=0.05)
        
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap
        
        
class mlp_bandgap_test(nn.Module):
    def __init__(self, device):
        super(mlp_bandgap_test, self).__init__()
        self.device = device
        self.linear = nn.Linear(201, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.linear.weight,std=0.05)
        
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.to(self.device)
        pred_bandgap = self.linear(input_dos)
        
        return pred_bandgap
    
class mlp_efermi(nn.Module):
    def __init__(self, device):
        super(mlp_efermi, self).__init__()
        self.device = device
        self.linear = nn.Linear(201, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.linear.weight,std=0.05)
        
    def forward(self, g):
        
        # batch = g.batch
        input_dos = g.y_ft.reshape(len(g.mp_id),-1)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi
    
class mlp_efermi_test(nn.Module):
    def __init__(self, device):
        super(mlp_efermi_test, self).__init__()
        self.device = device
        self.linear = nn.Linear(201, 1)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.linear.weight,std=0.05)

    def forward(self, g):
        
        input_dos = g.to(self.device)
        pred_efermi = self.linear(input_dos)
        
        return pred_efermi
    
'''