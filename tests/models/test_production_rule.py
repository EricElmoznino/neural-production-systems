import pytest

import torch

from models.production_rule import MLPRule


class TestMLPRule:

    @pytest.mark.parametrize('embedding_size', [10])
    @pytest.mark.parametrize('state_size', [5])
    @pytest.mark.parametrize('hidden_sizes', [(), (6,), (6, 7)])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    def test_shapes(self, embedding_size, state_size, hidden_sizes, dropout_p):
        rule = MLPRule(embedding_size=embedding_size,
                       state_size=state_size,
                       hidden_sizes=hidden_sizes,
                       dropout_p=dropout_p)

        batch_size = 5
        v_primary, v_context = torch.randn(batch_size, state_size), \
                               torch.randn(batch_size, state_size)
        v_primary = rule(v_primary, v_context)

        assert rule.embedding.shape == (embedding_size,)
        assert v_primary.shape == (batch_size, state_size)

    def test_backward(self):
        state_size = 5
        rule = MLPRule(embedding_size=10,
                       state_size=state_size,
                       hidden_sizes=(6, 7),
                       dropout_p=0.5)

        batch_size = 5
        v_primary, v_context = torch.randn(batch_size, state_size), \
                               torch.randn(batch_size, state_size)
        v_context.requires_grad = True
        v_primary = rule(v_primary, v_context)
        loss = v_primary.sum()
        loss.backward()

        assert v_context.grad is not None
        assert not v_context.grad.isnan().any()
        assert not v_context.grad.isinf().any()
