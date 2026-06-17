import torch
from torch import nn
class VNN(nn.Module):
    def __init__(self, input_features , output_features , hidden_units=10):
        super().__init__()
        self.X0=nn.Parameter(torch.tensor(0.0))
        self.linear_layer_stack=nn.Sequential(
            nn.Linear(in_features=input_features , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=output_features),
            nn.Sigmoid())
    def forward(self , x):
         return self.linear_layer_stack(x)
    

class SNN(nn.Module):
    def __init__(self, input_features , output_features , hidden_units=10):
        super().__init__()
        self.X0=nn.Parameter(torch.tensor(0.0))
        self.linear_layer_stack=nn.Sequential(
            nn.Linear(in_features=input_features , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=hidden_units),
            nn.Tanh(),
            nn.Linear(in_features=hidden_units , out_features=output_features),
            nn.Sigmoid())
    def forward(self , x):
         return self.linear_layer_stack(x)
    

class RNN(nn.Module):
    def __init__(self , input_features , output_features , hidden_units=10):
        super().__init__()
        self.X0=nn.Parameter(torch.tensor(0.0))
        self.rec_layer=nn.LSTM(input_features , hidden_units , 3)
        self.output_layer = nn.Sequential(nn.Linear(hidden_units, output_features), nn.Sigmoid())
    def forward (self , x):
        output , _ = self.rec_layer(x)
        return self.output_layer(output)    

