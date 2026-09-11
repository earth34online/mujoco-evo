import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int):
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        return self.pe[:, :seq_len, :]

    def _extend_pe(self, new_max_len):
        old_max_len, dim = self.pe.size(1), self.pe.size(2)
        if new_max_len <= old_max_len:
            return
        extra_positions = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        extra_pe = torch.zeros(new_max_len - old_max_len, dim)
        extra_pe[:, 0::2] = torch.sin(extra_positions * div_term)
        extra_pe[:, 1::2] = torch.cos(extra_positions * div_term)
        extra_pe = extra_pe.unsqueeze(0)
        new_pe = torch.cat([self.pe, extra_pe.to(self.pe.device)], dim=1)
        self.pe = new_pe

class CategorySpecificLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_categories: int = 1):
        super().__init__()
        self.num_categories = num_categories
        if num_categories <= 1:
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            self.weight = nn.Parameter(torch.empty(num_categories, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(num_categories, out_dim))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):

        if self.num_categories <= 1:
            if x.dtype != self.linear.weight.dtype:
                x = x.to(dtype=self.linear.weight.dtype)
            return self.linear(x)

        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)

        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1]) 
        if category_id.dim() == 0:
       
            cid = category_id.item()
            out = x_flat @ self.weight[cid] + self.bias[cid]
        else:
           
            category_id = category_id.reshape(-1)
            if category_id.numel() != x_flat.size(0):
                raise ValueError(
                    f"category_id length {category_id.numel()} does not match "
                    f"flattened batch {x_flat.size(0)}"
                )
            weight_selected = self.weight[category_id]        
            bias_selected = self.bias[category_id]        
            out = torch.bmm(x_flat.unsqueeze(1), weight_selected).squeeze(1) + bias_selected
        out_shape = orig_shape[:-1] + (out.shape[-1],)
        return out.view(out_shape)

class CategorySpecificMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_categories: int = 1):
        super().__init__()
        self.fc1 = CategorySpecificLinear(input_dim, hidden_dim, num_categories)
        self.fc2 = CategorySpecificLinear(hidden_dim, output_dim, num_categories)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, category_id: torch.LongTensor):
        out = self.activation(self.fc1(x, category_id))
        out = self.fc2(out, category_id)
        return out

class MultiEmbodimentActionEncoder(nn.Module):

    def __init__(self, action_dim: int, embed_dim: int, hidden_dim: int, horizon: int, num_categories: int = 1):
        super().__init__()
        self.horizon = horizon
        self.embed_dim = embed_dim
        self.num_categories = num_categories
        
        self.W1 = CategorySpecificLinear(action_dim, hidden_dim, num_categories)
        self.W2 = CategorySpecificLinear(hidden_dim, hidden_dim, num_categories)
        self.W3 = CategorySpecificLinear(hidden_dim, embed_dim, num_categories)
   
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, max_len=horizon)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, action_seq: torch.Tensor, category_id: torch.LongTensor):

        B, H, D = action_seq.shape
        assert H == self.horizon, "Action sequence length must match horizon"
       
        x = action_seq.reshape(B * H, D) 
      
        if category_id.dim() == 0:
           
            cat_ids = category_id.expand(H * B)
        else:
            cat_ids = category_id.unsqueeze(1).expand(B, H).reshape(B * H)
        out = self.activation(self.W1(x, cat_ids))            
    
        pos_enc = self.pos_encoding(H).to(device=out.device, dtype=out.dtype)
        out = out.view(B, H, -1) + pos_enc
        out = out.view(B * H, -1)
        out = self.activation(self.W2(out, cat_ids))         
        out = self.W3(out, cat_ids)                        
        out = out.view(B, H, self.embed_dim)
        return out

class BasicTransformerBlock(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        time_emb: torch.Tensor,
        context_key_padding_mask: torch.Tensor = None,
    ):

        x = self.norm1(action_tokens)
        attn_out, _ = self.attn(
            x,
            context_tokens,
            context_tokens,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )

        x = action_tokens + attn_out

        x2 = self.norm2(x)

        if time_emb is not None:
            x2 = x2 + time_emb.unsqueeze(1).to(dtype=x2.dtype, device=x2.device)
        x2 = x2.to(dtype=self.ff[0].weight.dtype)
        ff_out = self.ff(x2)
        x = x + ff_out
        return x

class FlowmatchingActionHead(nn.Module):

    def __init__(self, config=None,
                 embed_dim: int = 896, 
                 hidden_dim: int = 1024,
                 action_dim: int = 16*7,
                 horizon: int = 16,
                 per_action_dim: int = 7,
                 num_heads: int = 8,
                 num_layers: int = 8,
                 dropout: float = 0.0,
                 num_inference_timesteps: int = 20,
                 num_categories: int = 1):
        super().__init__()

        if config is not None:
      
            embed_dim = getattr(config, "embed_dim", embed_dim)
            hidden_dim = getattr(config, "hidden_dim", hidden_dim)
            action_dim = getattr(config, "action_dim", action_dim)
            horizon = getattr(config, "horizon", horizon)
            num_heads = getattr(config, "num_heads", num_heads)
            num_layers = getattr(config, "num_layers", num_layers)
            dropout = getattr(config, "dropout", dropout)
            num_inference_timesteps = getattr(config, "num_inference_timesteps", num_inference_timesteps)
            num_categories = getattr(config, "num_categories", num_categories)
            self.config = config
        else:
            from types import SimpleNamespace
            self.config = SimpleNamespace(embed_dim=embed_dim, hidden_dim=hidden_dim,
                                          action_dim=action_dim, horizon=horizon,
                                          per_action_dim=per_action_dim,
                                          num_heads=num_heads, num_layers=num_layers,
                                          dropout=dropout, num_inference_timesteps=num_inference_timesteps,
                                          num_categories=num_categories)
        logger.info(
            "FlowmatchingActionHead num_inference_timesteps=%s",
            num_inference_timesteps,
        )
        self.embed_dim = embed_dim
        self.horizon = horizon
        self.per_action_dim = getattr(self.config, "per_action_dim", per_action_dim)
        self.action_dim = getattr(self.config, "action_dim", action_dim)


        self.time_pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=1000)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(embed_dim=embed_dim, num_heads=num_heads,
                                   hidden_dim=embed_dim*4, dropout=dropout)
            for _ in range(num_layers)
        ])
       
        self.norm_out = nn.LayerNorm(embed_dim)
        self.seq_pool_proj = nn.Linear(self.horizon * self.embed_dim, self.embed_dim)

        self.mlp_head = CategorySpecificMLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                                            output_dim=action_dim, num_categories=num_categories)

        self.use_state = bool(getattr(self.config, "use_state", False))
        self.state_encoder = None
        if hasattr(self.config, "state_dim") and self.config.state_dim is not None:
       
            state_hidden = getattr(self.config, "state_hidden_dim", embed_dim)
        
            self.state_encoder = CategorySpecificMLP(input_dim=self.config.state_dim,
                                                    hidden_dim=state_hidden,
                                                    output_dim=embed_dim,
                                                    num_categories=num_categories)
            if not self.use_state:
                for param in self.state_encoder.parameters():
                    param.requires_grad = False

        self.action_encoder = None
        if horizon > 1:
          
            per_action_dim = getattr(self.config, "per_action_dim", None)
            if per_action_dim is None:
            
                per_action_dim = action_dim // horizon if action_dim % horizon == 0 else action_dim
            self.action_encoder = MultiEmbodimentActionEncoder(action_dim=per_action_dim,
                                                               embed_dim=embed_dim,
                                                               hidden_dim=embed_dim,
                                                               horizon=horizon,
                                                               num_categories=num_categories)
            self.single_action_proj = None
        else:
            self.action_encoder = None
            self.single_action_proj = nn.Linear(
                self.per_action_dim,
                self.embed_dim,
            )

    def _project_actions(
        self,
        action_seq: torch.Tensor,
        embodiment_id: torch.LongTensor,
    ) -> torch.Tensor:
        if self.horizon > 1 and self.action_encoder is not None:
            return self.action_encoder(action_seq, embodiment_id)
        if self.single_action_proj is None:
            raise RuntimeError("single_action_proj is not initialized")
        return self.single_action_proj(action_seq)

    def _expand_action_mask(
        self,
        action_mask: torch.Tensor,
        batch_size: int,
        per_action_dim: int,
        device,
        dtype,
    ) -> torch.Tensor:
        if action_mask is None:
            raise ValueError("action_mask must be provided for flow matching inference")

        expected_flat_dim = self.horizon * per_action_dim
        if action_mask.dim() == 1:
            if action_mask.shape[0] == per_action_dim:
                expanded = action_mask.view(1, 1, per_action_dim).expand(
                    batch_size,
                    self.horizon,
                    per_action_dim,
                )
            elif action_mask.shape[0] == expected_flat_dim:
                expanded = action_mask.view(
                    1, self.horizon, per_action_dim
                ).expand(batch_size, self.horizon, per_action_dim)
            else:
                raise ValueError(
                    "Expected one-dimensional action_mask length "
                    f"{per_action_dim} or {expected_flat_dim}, got "
                    f"{action_mask.shape[0]}"
                )
        elif action_mask.dim() == 2:
            if action_mask.shape == (batch_size, expected_flat_dim):
                expanded = action_mask.reshape(
                    batch_size,
                    self.horizon,
                    per_action_dim,
                )
            elif action_mask.shape == (batch_size, per_action_dim):
                expanded = action_mask.unsqueeze(1).expand(
                    batch_size,
                    self.horizon,
                    per_action_dim,
                )
            else:
                raise ValueError(
                    "Expected action_mask shape "
                    f"{(batch_size, expected_flat_dim)} or "
                    f"{(batch_size, per_action_dim)}, got "
                    f"{tuple(action_mask.shape)}"
                )
        elif action_mask.dim() == 3:
            expected_shape = (batch_size, self.horizon, per_action_dim)
            if tuple(action_mask.shape) != expected_shape:
                raise ValueError(
                    f"Expected action_mask shape {expected_shape}, got "
                    f"{tuple(action_mask.shape)}"
                )
            expanded = action_mask
        else:
            raise ValueError(f"Unsupported action_mask rank: {action_mask.dim()}")

        return expanded.to(device=device, dtype=dtype)

    def _state_temporal_encoding(
        self,
        length: int,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        """Fixed sinusoidal encoding for positions ``[-K+1, ..., 0]``."""
        positions = torch.arange(
            -(length - 1), 1, device=device, dtype=torch.float32
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / self.embed_dim)
        )
        encoding = torch.zeros(
            length, self.embed_dim, device=device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if self.embed_dim > 1:
            odd_width = encoding[:, 1::2].shape[1]
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[:odd_width]
            ) - 1.0
        return encoding.to(dtype=dtype)

    def _encode_state_history(
        self,
        state: torch.Tensor,
        embodiment_id: torch.LongTensor,
        history_mask: torch.Tensor = None,
    ):
        """Project K proprioceptive observations to K masked context tokens."""
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"Expected state [B,D] or [B,K,D], got {tuple(state.shape)}"
            )
        batch_size, memory_length, state_dim = state.shape
        if state_dim != self.config.state_dim:
            raise ValueError(
                f"Expected state dimension {self.config.state_dim}, got {state_dim}"
            )

        if history_mask is None:
            history_mask = torch.ones(
                batch_size, memory_length, dtype=torch.bool, device=state.device
            )
        else:
            history_mask = torch.as_tensor(
                history_mask, dtype=torch.bool, device=state.device
            )
            if history_mask.ndim == 1:
                history_mask = history_mask.unsqueeze(0)
            if history_mask.shape != (batch_size, memory_length):
                raise ValueError(
                    f"Expected history_mask {(batch_size, memory_length)}, "
                    f"got {tuple(history_mask.shape)}"
                )
        if not bool(history_mask[:, -1].all()):
            raise ValueError("Every sample's current (last) state must be valid")

        if embodiment_id.dim() == 0:
            flat_embodiment_ids = embodiment_id.expand(batch_size * memory_length)
        else:
            if embodiment_id.numel() != batch_size:
                raise ValueError(
                    f"Expected {batch_size} embodiment ids, got {embodiment_id.numel()}"
                )
            flat_embodiment_ids = embodiment_id.reshape(batch_size, 1).expand(
                batch_size, memory_length
            ).reshape(-1)
        state_tokens = self.state_encoder(
            state.reshape(batch_size * memory_length, state_dim),
            flat_embodiment_ids,
        ).reshape(batch_size, memory_length, self.embed_dim)
        state_tokens = state_tokens + self._state_temporal_encoding(
            memory_length,
            device=state_tokens.device,
            dtype=state_tokens.dtype,
        ).unsqueeze(0)
        state_tokens = state_tokens * history_mask.unsqueeze(-1).to(state_tokens.dtype)
        return state_tokens, ~history_mask

    def _build_context(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.LongTensor,
        history_mask: torch.Tensor = None,
        fused_mask: torch.Tensor = None,
    ):
        context_tokens = fused_tokens
        if fused_mask is None:
            context_key_padding_mask = torch.zeros(
                fused_tokens.shape[:2],
                dtype=torch.bool,
                device=fused_tokens.device,
            )
        else:
            fused_mask = torch.as_tensor(
                fused_mask, dtype=torch.bool, device=fused_tokens.device
            )
            if fused_mask.shape != fused_tokens.shape[:2]:
                raise ValueError(
                    f"Expected fused_mask shape {tuple(fused_tokens.shape[:2])}, "
                    f"got {tuple(fused_mask.shape)}"
                )
            if not bool(fused_mask.any(dim=1).all()):
                raise ValueError("Every sample must contain at least one valid fused token")
            context_key_padding_mask = ~fused_mask
        if self.use_state and state is not None and self.state_encoder is not None:
            state_tokens, state_padding_mask = self._encode_state_history(
                state, embodiment_id, history_mask
            )
            context_tokens = torch.cat([context_tokens, state_tokens], dim=1)
            context_key_padding_mask = torch.cat(
                [
                    context_key_padding_mask,
                    state_padding_mask,
                ],
                dim=1,
            )
        return context_tokens, context_key_padding_mask

    def forward(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        actions_gt: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        state_mask: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        history_mask: torch.Tensor = None,
        fused_mask: torch.Tensor = None,
    ):

        if actions_gt is None:
            return self.get_action(
                fused_tokens,
                state=state,
                embodiment_id=embodiment_id,
                action_mask=action_mask,
                history_mask=history_mask,
                fused_mask=fused_mask,
            )
        B = fused_tokens.size(0)
        device = fused_tokens.device

        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens, context_key_padding_mask = self._build_context(
            fused_tokens, state, embodiment_id, history_mask, fused_mask
        )

        t = torch.distributions.Beta(2, 2).sample((B,)).clamp(0.02, 0.98).to(device).to(dtype=self.dtype)

        
                    
        time_index = (t * 1000).long()
        time_emb = self.time_pos_enc(1000)[:, time_index, :].squeeze(0)
        time_emb = time_emb.to(dtype=context_tokens.dtype)
    
        noise = torch.rand_like(actions_gt) * 2 - 1

        if action_mask is not None:
            action_mask = action_mask.to(dtype=noise.dtype, device=noise.device)
            assert action_mask.shape == noise.shape, f"action_mask shape {action_mask.shape} != noise shape {noise.shape}"
            noise = noise * action_mask

        actions_gt_seq = actions_gt

        if self.horizon > 1:
            noise_seq = noise.view(B, self.horizon, self.per_action_dim)
            
        else:
            noise_seq = noise.unsqueeze(1)

        if self.horizon > 1:
            t_broadcast = t.view(B, 1, 1)
        else:
            t_broadcast = t.view(B, 1)
        action_intermediate_seq = (1 - t_broadcast) * noise_seq + t_broadcast * actions_gt_seq
        if action_mask is not None:
            # Invalid embodiment dimensions must not enter action-token
            # self-attention.  Masking only the loss still lets arbitrary
            # padded targets influence valid action predictions.
            action_intermediate_seq = action_intermediate_seq * action_mask

        action_tokens = self._project_actions(
            action_intermediate_seq,
            embodiment_id,
        )
        target_dtype = self.dtype
        action_tokens = action_tokens.to(dtype=target_dtype)
        context_tokens = context_tokens.to(dtype=target_dtype)
        time_emb = time_emb.to(dtype=target_dtype)

        x = action_tokens
        for block in self.transformer_blocks:
            x = block(
                x,
                context_tokens,
                time_emb,
                context_key_padding_mask=context_key_padding_mask,
            )

        x = self.norm_out(x)  

        if self.horizon > 1:
 
            x_flat = x.reshape(B, -1)  

            x_pooled = self.seq_pool_proj(x_flat)
        else:
          
            x_pooled = x.squeeze(1) 

        pred_velocity = self.mlp_head(x_pooled, embodiment_id) 

        return pred_velocity, noise

    def get_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor = None,
        embodiment_id: torch.LongTensor = None,
        action_mask: torch.Tensor = None,
        history_mask: torch.Tensor = None,
        fused_mask: torch.Tensor = None,
    ):


        B = fused_tokens.size(0)
        device = fused_tokens.device
        if embodiment_id is None:
            embodiment_id = torch.zeros(B, dtype=torch.long, device=device)

        context_tokens, context_key_padding_mask = self._build_context(
            fused_tokens, state, embodiment_id, history_mask, fused_mask
        )

        action_dim_total = getattr(self.config, "action_dim", None)
        if action_dim_total is None:
          
            action_dim_total = self.action_dim
       
        if self.horizon > 1:
            per_action_dim = getattr(self.config, "per_action_dim", action_dim_total // self.horizon)
        else:
            per_action_dim = action_dim_total

        action = (
            torch.rand(
                B,
                action_dim_total,
                device=device,
                dtype=context_tokens.dtype,
            ) * 2 - 1
        )

        if self.horizon > 1:
            action_seq = action.view(B, self.horizon, per_action_dim)

        else:
            action_seq = action.view(B, 1, per_action_dim)

        action_mask = self._expand_action_mask(
            action_mask,
            batch_size=B,
            per_action_dim=per_action_dim,
            device=action_seq.device,
            dtype=action_seq.dtype,
        )
        action_seq = action_seq * action_mask

        N = int(getattr(self.config, "num_inference_timesteps", 32))
        if N <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {N}")
        dt = 1.0 / N
        target_dtype = self.dtype
        context_tokens = context_tokens.to(dtype=target_dtype)
        time_table = self.time_pos_enc(1000)[0].to(
            device=device, dtype=target_dtype
        )
        for i in range(N):
            t = i / N

            time_index = int(t * 1000)
            time_emb = time_table[time_index].unsqueeze(0).expand(B, -1)
            action_seq = action_seq * action_mask
            action_tokens = self._project_actions(action_seq, embodiment_id)
            action_tokens = action_tokens.to(dtype=target_dtype)
            time_emb = time_emb.to(dtype=target_dtype)

            x = action_tokens
            for block in self.transformer_blocks:
                x = block(
                    x,
                    context_tokens,
                    time_emb,
                    context_key_padding_mask=context_key_padding_mask,
                )
            x = self.norm_out(x)

            if self.horizon > 1:
                x_flat = x.reshape(B, -1)
                x_pooled = self.seq_pool_proj(x_flat)
            else:
                x_pooled = x.squeeze(1)
         
            pred = self.mlp_head(x_pooled, embodiment_id)  
  
            action = action + dt * pred
          
            if self.horizon > 1:
                action_seq = action.view(B, self.horizon, per_action_dim)
            else:
                action_seq = action.view(B, 1, per_action_dim)
      
        action_seq = action_seq * action_mask
        return action_seq.reshape(B, -1)

    @property
    def device(self):
      
        return next(self.parameters()).device

    @property
    def dtype(self):
        
        return next(self.parameters()).dtype


