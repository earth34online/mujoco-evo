"""轻量、无 PEFT 依赖的 LoRA 层。

实现保留原 ``weight``/``bias`` 的 state-dict 键，只额外增加 ``lora_*``
参数。因此旧 Evo-1 checkpoint 可以作为 LoRA 训练的初始化权重，新 LoRA
checkpoint 也可以由同一模型结构严格加载。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    """保持 ``nn.Linear`` 接口和原权重键的低秩增量线性层。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        if rank <= 0:
            raise ValueError(f"LoRA rank 必须大于 0，当前为 {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha 必须大于 0，当前为 {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout 必须位于 [0, 1)，当前为 {dropout}")

        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = nn.Dropout(float(dropout))
        # 小型 LoRA 参数保持 FP32，避免 lr=1e-5 时 BF16 更新被舍入为零。
        self.lora_A = nn.Parameter(
            torch.empty(self.lora_rank, in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, self.lora_rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_merged = False

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        result = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # 直接沿用 Parameter，避免复制大权重并保持旧 checkpoint 键完全一致。
        result.weight = linear.weight
        result.bias = linear.bias
        result.train(linear.training)
        return result

    def _delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.lora_scaling

    def merge_lora_(self) -> None:
        if self.lora_merged:
            return
        with torch.no_grad():
            self.weight.add_(self._delta_weight().to(self.weight.dtype))
        self.lora_merged = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base = F.linear(input, self.weight, self.bias)
        if self.lora_merged:
            return base
        # Keep FP32 master parameters for optimizer accuracy, but execute the
        # adapter matmuls in the model's activation dtype.  Casting a full
        # [B*K*N,C] activation and its 3C-wide QKV update to FP32 makes a true
        # physical batch need hundreds of extra MiB; gradients still flow
        # through these casts and accumulate on the FP32 master parameters.
        compute_dtype = base.dtype
        adapter_input = self.lora_dropout(input).to(compute_dtype)
        update = F.linear(
            F.linear(adapter_input, self.lora_A.to(compute_dtype)),
            self.lora_B.to(compute_dtype),
        )
        return base + update * self.lora_scaling


class LoRAResidualProjection(nn.Module):
    """Low-rank residual for one memory call path without replacing the base.

    The temporal video encoder reuses a ViT attention module for both spatial
    and temporal attention.  Replacing ``attention.qkv`` with ``LoRALinear``
    applies the same adapter to both paths and can therefore move a previously
    trained single-frame representation.  This module owns only the low-rank
    residual; callers explicitly add it during temporal matching or at the
    composed memory output while the shared base projection remains untouched.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        device=None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank 必须大于 0，当前为 {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha 必须大于 0，当前为 {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout 必须位于 [0, 1)，当前为 {dropout}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(
                self.lora_rank,
                self.in_features,
                device=device,
                dtype=torch.float32,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.lora_rank,
                device=device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRAResidualProjection":
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=linear.weight.device,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        compute_dtype = input.dtype
        adapter_input = self.lora_dropout(input).to(compute_dtype)
        hidden = F.linear(adapter_input, self.lora_A.to(compute_dtype))
        update = F.linear(hidden, self.lora_B.to(compute_dtype))
        return update * self.lora_scaling


class LoRAMultiheadAttention(nn.MultiheadAttention):
    """对 MHA 的融合 QKV 和输出投影实施标准低秩权重增量。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = False,
        rank: int = 8,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
        device=None,
        dtype=None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        if rank <= 0:
            raise ValueError(f"LoRA rank 必须大于 0，当前为 {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha 必须大于 0，当前为 {alpha}")
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(
                f"LoRA dropout 必须位于 [0, 1)，当前为 {lora_dropout}"
            )
        self.lora_rank = int(rank)
        self.lora_alpha = float(alpha)
        self.lora_scaling = self.lora_alpha / self.lora_rank
        self.lora_dropout = nn.Dropout(float(lora_dropout))
        self.lora_A_in = nn.Parameter(
            torch.empty(rank, embed_dim, device=device, dtype=torch.float32)
        )
        self.lora_B_in = nn.Parameter(
            torch.zeros(3 * embed_dim, rank, device=device, dtype=torch.float32)
        )
        self.lora_A_out = nn.Parameter(
            torch.empty(rank, embed_dim, device=device, dtype=torch.float32)
        )
        self.lora_B_out = nn.Parameter(
            torch.zeros(embed_dim, rank, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A_in, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))
        self.lora_merged = False

    @classmethod
    def from_mha(
        cls,
        module: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRAMultiheadAttention":
        if not module._qkv_same_embed_dim:
            raise ValueError("当前 LoRA MHA 仅支持 Q/K/V 使用相同 embed_dim")
        result = cls(
            module.embed_dim,
            module.num_heads,
            dropout=module.dropout,
            bias=module.in_proj_bias is not None,
            add_bias_kv=module.bias_k is not None,
            add_zero_attn=module.add_zero_attn,
            batch_first=module.batch_first,
            rank=rank,
            alpha=alpha,
            lora_dropout=dropout,
            device=module.in_proj_weight.device,
            dtype=module.in_proj_weight.dtype,
        )
        result.in_proj_weight = module.in_proj_weight
        result.in_proj_bias = module.in_proj_bias
        result.out_proj.weight = module.out_proj.weight
        result.out_proj.bias = module.out_proj.bias
        result.bias_k = module.bias_k
        result.bias_v = module.bias_v
        result.train(module.training)
        return result

    def _delta_in(self) -> torch.Tensor:
        return (self.lora_B_in @ self.lora_A_in) * self.lora_scaling

    def _delta_out(self) -> torch.Tensor:
        return (self.lora_B_out @ self.lora_A_out) * self.lora_scaling

    def merge_lora_(self) -> None:
        if self.lora_merged:
            return
        with torch.no_grad():
            self.in_proj_weight.add_(self._delta_in().to(self.in_proj_weight.dtype))
            self.out_proj.weight.add_(self._delta_out().to(self.out_proj.weight.dtype))
        self.lora_merged = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        is_batched = query.dim() == 3
        if not is_batched:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if self.batch_first and is_batched:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        if self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise RuntimeError("LoRA MHA 当前不支持 bias_k/bias_v/add_zero_attn")

        target_len, batch_size, _ = query.shape
        source_len = key.shape[0]
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)
        q_proj = F.linear(query, q_weight, q_bias)
        k_proj = F.linear(key, k_weight, k_bias)
        v_proj = F.linear(value, v_weight, v_bias)

        if not self.lora_merged:
            b_q, b_k, b_v = self.lora_B_in.chunk(3, dim=0)

            def low_rank(x, b):
                compute_dtype = q_proj.dtype
                adapter_input = self.lora_dropout(x).to(compute_dtype)
                hidden = F.linear(
                    adapter_input, self.lora_A_in.to(compute_dtype)
                )
                return F.linear(
                    hidden, b.to(compute_dtype)
                ) * self.lora_scaling

            q_proj = q_proj + low_rank(query, b_q)
            k_proj = k_proj + low_rank(key, b_k)
            v_proj = v_proj + low_rank(value, b_v)

        head_dim = self.embed_dim // self.num_heads
        q = q_proj.contiguous().view(
            target_len, batch_size, self.num_heads, head_dim
        ).permute(1, 2, 0, 3)
        k = k_proj.contiguous().view(
            source_len, batch_size, self.num_heads, head_dim
        ).permute(1, 2, 0, 3)
        v = v_proj.contiguous().view(
            source_len, batch_size, self.num_heads, head_dim
        ).permute(1, 2, 0, 3)

        additive_mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                expanded = attn_mask.view(1, 1, target_len, source_len)
            elif attn_mask.dim() == 3:
                expanded = attn_mask.view(
                    batch_size, self.num_heads, target_len, source_len
                )
            else:
                raise ValueError("attn_mask 必须是二维或三维")
            if expanded.dtype == torch.bool:
                additive_mask = torch.zeros(
                    expanded.shape, device=q.device, dtype=q.dtype
                ).masked_fill(expanded.to(q.device), float("-inf"))
            else:
                additive_mask = expanded.to(device=q.device, dtype=q.dtype)

        if key_padding_mask is not None:
            padding = key_padding_mask.view(batch_size, 1, 1, source_len)
            if padding.dtype == torch.bool:
                padding = torch.zeros(
                    padding.shape, device=q.device, dtype=q.dtype
                ).masked_fill(padding.to(q.device), float("-inf"))
            else:
                padding = padding.to(device=q.device, dtype=q.dtype)
            additive_mask = padding if additive_mask is None else additive_mask + padding

        if is_causal and additive_mask is not None:
            causal = torch.ones(
                target_len, source_len, device=q.device, dtype=torch.bool
            ).triu(diagonal=1)
            causal = torch.zeros(
                (1, 1, target_len, source_len), device=q.device, dtype=q.dtype
            ).masked_fill(causal, float("-inf"))
            additive_mask = additive_mask + causal
            is_causal = False

        dropout_p = self.dropout if self.training else 0.0
        if need_weights:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            if additive_mask is not None:
                scores = scores + additive_mask
            if is_causal:
                causal = torch.ones(
                    target_len, source_len, device=q.device, dtype=torch.bool
                ).triu(diagonal=1)
                scores = scores.masked_fill(causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=dropout_p, training=self.training)
            attended = torch.matmul(weights, v)
            if average_attn_weights:
                weights = weights.mean(dim=1)
        else:
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
            weights = None

        attended = attended.permute(2, 0, 1, 3).contiguous().view(
            target_len, batch_size, self.embed_dim
        )
        output = F.linear(attended, self.out_proj.weight, self.out_proj.bias)
        if not self.lora_merged:
            compute_dtype = output.dtype
            hidden = F.linear(
                attended.to(compute_dtype), self.lora_A_out.to(compute_dtype)
            )
            update = F.linear(hidden, self.lora_B_out.to(compute_dtype))
            output = output + update * self.lora_scaling

        if self.batch_first and is_batched:
            output = output.transpose(0, 1)
        elif not is_batched:
            output = output.squeeze(1)
            if weights is not None:
                weights = weights.squeeze(0)
        return output, weights


def inject_lora_linear_layers(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
    predicate: Optional[Callable[[str, nn.Module], bool]] = None,
    include_multihead_attention: bool = True,
) -> List[str]:
    """递归替换线性层，返回已注入模块的完整名称。"""

    replaced: List[str] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
            continue
        allowed = predicate is None or predicate(full_name, child)
        if include_multihead_attention and isinstance(child, nn.MultiheadAttention) and allowed:
            setattr(
                module,
                child_name,
                LoRAMultiheadAttention.from_mha(
                    child, rank=rank, alpha=alpha, dropout=dropout
                ),
            )
            replaced.append(full_name)
            continue
        if isinstance(child, nn.Linear) and allowed:
            setattr(
                module,
                child_name,
                LoRALinear.from_linear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            replaced.append(full_name)
            continue
        replaced.extend(
            inject_lora_linear_layers(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
                predicate=predicate,
                include_multihead_attention=include_multihead_attention,
            )
        )
    return replaced


def attach_temporal_attention_lora(
    attention: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> List[str]:
    """Attach residuals used only by the composed short-term-memory path."""

    attached: List[str] = []
    specifications = (
        ("qkv", "temporal_qkv_lora"),
        ("proj", "temporal_proj_lora"),
    )
    for base_name, adapter_name in specifications:
        base = getattr(attention, base_name, None)
        if not isinstance(base, nn.Linear):
            raise TypeError(
                f"Temporal LoRA requires attention.{base_name} to be nn.Linear, "
                f"got {type(base)!r}"
            )
        existing = getattr(attention, adapter_name, None)
        if existing is not None:
            if not isinstance(existing, LoRAResidualProjection):
                raise TypeError(
                    f"attention.{adapter_name} already exists with incompatible "
                    f"type {type(existing)!r}"
                )
        else:
            setattr(
                attention,
                adapter_name,
                LoRAResidualProjection.from_linear(
                    base,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
        attached.append(f"{prefix}.{adapter_name}" if prefix else adapter_name)
    return attached


def is_lora_parameter(name: str) -> bool:
    return any(part.startswith("lora_A") or part.startswith("lora_B") for part in name.split("."))


def merge_lora_weights(module: nn.Module) -> int:
    """把 LoRA 增量合并进基础权重，用于无额外算子开销的推理。"""

    merged = 0
    for child in module.modules():
        if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
            child.merge_lora_()
            merged += 1
    return merged
