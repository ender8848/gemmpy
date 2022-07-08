import torch

class Intv:
    def __init__(self):
        self.upper = float(0)
        self.lower = float(0)

A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]], dtype=Intv)