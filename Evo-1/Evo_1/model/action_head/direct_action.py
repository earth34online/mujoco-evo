"""Trainable action chunks decoded from native Pi-MEM observations."""
import torch
from torch import nn

from model.action_head.flow_matching import FlowmatchingActionHead


class DirectActionHead(nn.Module):
    _expand_action_mask = FlowmatchingActionHead._expand_action_mask

    def __init__(self, config):
        super().__init__()
        self.horizon = config["horizon"]
        self.per_action_dim = config["per_action_dim"]
        self.action_dim = self.horizon * self.per_action_dim
        self.use_state = bool(config.get("use_state", True))
        width = 256
        self.visual = nn.Linear(config.get("embed_dim", 896), width)
        self.state_encoder = (
            nn.Linear(config.get("state_dim", 7), width) if self.use_state else None
        )
        self.state_position = (nn.Parameter(torch.zeros(config.get("memory_frames", 1), width))
                               if self.use_state else None)
        self.queries = nn.Parameter(torch.empty(self.horizon, width))
        nn.init.normal_(self.queries, std=0.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.intent_classifier = None
        if config.get("drawer_intent_supervision", False):
            self.intent_query = nn.Parameter(torch.empty(1, 1, width))
            nn.init.normal_(self.intent_query, std=0.02)
            self.intent_classifier = nn.Linear(width, 4)
            self.intent_condition = nn.Linear(4, width, bias=False)
        self.readout = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, self.per_action_dim),
        )

    def forward(self, fused_tokens, state=None, action_mask=None, embodiment_id=None,
                history_mask=None, fused_mask=None, return_intent=False):
        context = self.visual(fused_tokens.to(self.visual.weight.dtype))
        batch = context.shape[0]
        valid = (torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
                 if fused_mask is None else fused_mask.to(device=context.device, dtype=torch.bool))
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        intent_logits = None
        if self.intent_classifier is not None:
            # This query reads only native Pi-MEM visual/language tokens,
            # before the measured robot state is appended to the context.
            visual_query = self.intent_query.expand(batch, -1, -1)
            intent, _ = self.attention(visual_query, context, context,
                                       key_padding_mask=~valid, need_weights=False)
            intent_logits = self.intent_classifier(intent[:, 0])
            query = query + self.intent_condition(intent_logits.softmax(dim=-1)).unsqueeze(1)
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("Direct action head requires robot proprioception")
            if state.ndim == 2:
                state = state.unsqueeze(1)
            count = state.shape[1]
            if not 1 <= count <= len(self.state_position):
                raise ValueError("State history exceeds the direct action head window")
            observed = self.state_encoder(state.to(self.state_encoder.weight.dtype))
            state_valid = (torch.ones(observed.shape[:2], device=context.device, dtype=torch.bool)
                           if history_mask is None else history_mask.reshape(batch, count).to(device=context.device, dtype=torch.bool))
            query = query + observed[:, -1:]
            context = torch.cat((context, observed + self.state_position[-count:]), dim=1)
            valid = torch.cat((valid, state_valid), dim=1)
        update, _ = self.attention(query, context, context, key_padding_mask=~valid, need_weights=False)
        actions = self.readout(query + update)
        mask = self._expand_action_mask(action_mask, batch, self.per_action_dim,
                                        actions.device, actions.dtype)
        actions = (actions * mask).reshape(batch, self.action_dim)
        return (actions, intent_logits) if return_intent else actions

    def get_action(self, *args, **kwargs):
        return self(*args, **kwargs).clamp(-1.0, 1.0)