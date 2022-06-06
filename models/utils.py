import math

import torch
from torch import Tensor
from torch import nn


class QueryKeyAttention(nn.Module):

    def __init__(self, d_dest: int, d_source: int, d_qk: int, softmax: bool=True):
        super(QueryKeyAttention, self).__init__()
        self.w_q = nn.Linear(d_dest, d_qk)
        self.w_k = nn.Linear(d_source, d_qk)
        self.temperature = math.sqrt(d_qk)
        self.softmax = softmax

    def forward(self, dest: Tensor, source: Tensor):
        q, k = self.w_q(dest), self.w_k(source)
        attn = torch.bmm(q, k.permute(0, 2, 1)) / self.temperature
        if self.softmax:
            attn = torch.softmax(attn, dim=-1)
        return attn


def argmax_onehot(x: Tensor, dim: int):
    idx = x.argmax(dim=dim)
    onehot = torch.zeros_like(x).scatter_(dim, idx.unsqueeze(dim), 1.0)
    return onehot
