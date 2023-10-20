import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList

from einops import rearrange, repeat

# functions

def exists(v):
    return v is not None

# class

class PauseTransformer(Module):
    raise NotImplementedError
