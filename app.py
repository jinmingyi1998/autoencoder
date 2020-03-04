import torch
from mf import MF
net = MF()
net.load_state_dict(torch.load('net.pkl'))
print("hello")