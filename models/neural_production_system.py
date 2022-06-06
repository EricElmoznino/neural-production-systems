from typing import List, Optional

import torch
from torch import Tensor, LongTensor
from torch import nn
from torch.nn import functional as F

from models.production_rule import ProductionRule, NullRule
from models.utils import QueryKeyAttention, argmax_onehot


class SequentialNPS(nn.Module):

    def __init__(self,
                 encoder: nn.Module,
                 rules: List[ProductionRule],
                 n_stages: int,
                 n_slots: int,
                 var_size: int,
                 d_qk: int = 32):
        super(SequentialNPS, self).__init__()
        assert len(rules) > 0
        for r in rules[1:]:
            assert r.embedding_size == rules[0].embedding_size

        # Properties
        self.rule_embedding_size = rules[0].embedding_size
        self.n_rules = len(rules)
        self.n_stages = n_stages
        self.n_slots = n_slots
        self.var_size = var_size
        self.d_qk = d_qk

        # Submodules
        self.encoder = encoder
        self.rules = nn.ModuleList(rules)
        self.var_rule_attention = QueryKeyAttention(d_dest=self.var_size,
                                                    d_source=self.rule_embedding_size,
                                                    d_qk=self.d_qk,
                                                    softmax=False)
        self.primary_context_attention = QueryKeyAttention(d_dest=self.var_size,
                                                           d_source=self.var_size,
                                                           d_qk=self.d_qk,
                                                           softmax=False)

    def forward(self, x: Tensor, vars: Optional[Tensor]):
        if vars is None:
            vars = torch.randn(x.size(0), self.n_slots, self.var_size)  # todo: verify this initialization method

        # Step 1: update the variable slots using the input
        vars = self.encoder(x, vars)

        for _ in range(self.n_stages):
            # Step 2: select a primary slot to update and a rule to update it with
            primary_rule_mask = self.select_primary_and_rule(vars)
            primary_mask, rule_mask = primary_rule_mask.sum(dim=2), primary_rule_mask.sum(dim=1)
            var_primary = (vars * primary_mask.unsqueeze(dim=2)).sum(dim=1)

            # Step 3: select a context slot
            context_mask = self.select_context(vars, var_primary)
            var_context = (vars * context_mask.unsqueeze(dim=2)).sum(dim=1)

            # Step 4: update the primary variable
            var_primary = self.apply_rule(var_primary, var_context, rule_mask)
            primary_mask = primary_mask.detach().bool()
            new_vars = torch.zeros_like(vars)
            new_vars[~primary_mask] = vars[~primary_mask]
            new_vars[primary_mask] = var_primary
            vars = new_vars

        return vars

    def select_primary_and_rule(self, vars: Tensor) -> LongTensor:
        batch_size, n_slots, var_size = vars.size()

        # Get attention scores between variables and rules
        rule_embeddings = torch.stack([r.embedding for r in self.rules])
        rule_embeddings = rule_embeddings.repeat(batch_size, 1, 1)
        primary_rule_logits = self.var_rule_attention(vars, rule_embeddings)

        # Select rules that win the attention competition
        primary_rule_logits = primary_rule_logits.view(batch_size, -1)
        if self.training:
            primary_rule_mask = F.gumbel_softmax(primary_rule_logits, tau=1.0, hard=True, dim=-1)
        else:
            primary_rule_mask = argmax_onehot(primary_rule_logits, dim=-1)
        primary_rule_mask = primary_rule_mask.view(batch_size, n_slots, self.n_rules)

        return primary_rule_mask

    def select_context(self, vars: Tensor, var_primary: Tensor) -> LongTensor:
        context_logits = self.primary_context_attention(var_primary.unsqueeze(dim=1), vars)
        context_logits = context_logits.squeeze(dim=1)

        if self.training:
            context_mask = F.gumbel_softmax(context_logits, tau=1.0, hard=True, dim=-1)
        else:
            context_mask = argmax_onehot(context_logits, dim=-1)

        return context_mask

    def apply_rule(self, var_primary: Tensor, var_context: Tensor, rule_mask: LongTensor) -> Tensor:
        rule_outputs = [rule(var_primary, var_context) for rule in self.rules]
        rule_outputs = torch.stack(rule_outputs, dim=2)
        v_primary = (rule_outputs * rule_mask.unsqueeze(dim=1)).sum(dim=2)
        return v_primary
