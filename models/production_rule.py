from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch import nn


class ProductionRule(nn.Module, ABC):

    def __init__(self, embedding_size: int):
        super(ProductionRule, self).__init__()

        self.embedding_size = embedding_size
        self.embedding = nn.Parameter(torch.randn(embedding_size))

    def forward(self,
                var_primary: Tensor,
                var_context: Tensor) -> Tensor:
        return var_primary + self.rule(var_primary, var_context)

    @abstractmethod
    def rule(self,
             var_primary: Tensor,
             var_context: Tensor) -> Tensor:
        pass


class NullRule(ProductionRule):

    def forward(self,
                var_primary: Tensor,
                var_context: Tensor) -> Tensor:
        return var_primary

    def rule(self,
             var_primary: Tensor,
             var_context: Tensor) -> Tensor:
        raise NotImplementedError()


# -------------------------------------------------------------
# ---------------------- Implementations ----------------------
# -------------------------------------------------------------

class MLPRule(ProductionRule):

    def __init__(self,
                 embedding_size: int,
                 state_size: int,
                 hidden_sizes: Tuple[int, ...] = (128,),
                 dropout_p: float = 0.0):
        assert 0 <= dropout_p <= 1
        super(MLPRule, self).__init__(embedding_size=embedding_size)

        layers = []
        feature_sizes = (2 * state_size,) + hidden_sizes
        for in_feats, out_feats in zip(feature_sizes[:-1], feature_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_feats, out_feats),
                nn.ReLU(inplace=True)
            ))
        layers.append(nn.Linear(feature_sizes[-1], state_size))
        self.layers = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def rule(self,
             var_primary: Tensor,
             var_context: Tensor) -> Tensor:
        x = torch.concat((var_primary, var_context), dim=1)
        x = self.layers(x)
        if self.dropout:
            x = self.dropout(x)
        return x
