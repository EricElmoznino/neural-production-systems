import pytest

import torch
from torch import Tensor
from torch import nn

from models.neural_production_system import SequentialNPS
from models.production_rule import MLPRule


class DummyEncoder(nn.Module):

    def __init__(self, batch_size: int, n_slots: int, var_size: int):
        super(DummyEncoder, self).__init__()
        self.batch_size = batch_size
        self.n_slots = n_slots
        self.var_size = var_size

    def forward(self, x: Tensor, vars: Tensor):
        if vars is not None:
            return vars
        return torch.randn(self.batch_size, self.n_slots, self.var_size)


def create_nps(batch_size, n_rules, n_stages, n_slots, var_size):
    encoder = DummyEncoder(batch_size=batch_size,
                           n_slots=n_slots,
                           var_size=var_size)
    rules = [MLPRule(embedding_size=10,
                     state_size=var_size,
                     hidden_sizes=(6,))
             for _ in range(n_rules)]
    nps = SequentialNPS(encoder=encoder,
                        rules=rules,
                        n_stages=n_stages,
                        n_slots=n_slots,
                        var_size=var_size)
    return nps


class TestSequentialNPS:

    @pytest.mark.parametrize('n_stages', [1, 2])
    @pytest.mark.parametrize('n_slots', [3])
    @pytest.mark.parametrize('var_size', [4])
    def test_shapes(self, n_stages, n_slots, var_size):
        batch_size = 5
        n_rules = 2
        nps = create_nps(batch_size, n_rules, n_stages, n_slots, var_size)

        vars = nps(x=torch.randn(batch_size), vars=None)

        assert vars.shape == (batch_size, n_slots, var_size)

    def test_backward(self):
        batch_size = 5
        n_slots = 3
        var_size = 4
        nps = create_nps(batch_size, n_rules=2, n_stages=2, n_slots=n_slots, var_size=var_size)

        vars = torch.randn(batch_size, n_slots, var_size)
        vars.requires_grad = True
        out_vars = nps(x=torch.randn(batch_size), vars=vars)
        loss = out_vars.sum()
        loss.backward()

        assert vars.grad is not None
        assert not vars.grad.isnan().any()
        assert not vars.grad.isinf().any()


