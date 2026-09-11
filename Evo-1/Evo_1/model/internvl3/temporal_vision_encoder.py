"""π-MEM-style short-term video encoding for InternVL vision towers.

The implementation deliberately owns no trainable parameters.  On every
``temporal_layer_interval``-th active ViT layer it reuses that layer's
normalization, QKV projection, output projection, layer-scale and drop-path
modules to add causal attention across time for the same spatial patch.  Past
tokens can be discarded before the upper ViT layers, and only the current
frame is returned to the language backbone.
"""

from __future__ import annotations

import math

import torch


def fixed_relative_temporal_encoding(
    length: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return fixed sinusoidal positions whose current-time vector is zero.

    Positions run from ``-(length - 1)`` through ``0``.  Subtracting the
    ordinary sinusoid at position zero enforces the π-MEM boundary condition
    e(0)=0 and makes the K=1 initialization exactly match the image encoder.
    """
    if length < 1:
        raise ValueError(f"Temporal length must be positive, got {length}")
    positions = torch.arange(
        -(length - 1), 1, device=device, dtype=torch.float32
    ).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dim)
    )
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if dim > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]]) - 1.0
    return encoding.to(dtype=dtype)


def _causal_temporal_attention(
    attention,
    hidden_states: torch.Tensor,
    history_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply an InternAttention module over time with a causal validity mask.

    Args:
        attention: Existing InternVL attention module.  Its QKV and projection
            weights are reused; this function creates no parameters.
        hidden_states: ``[groups, time, hidden]`` where each group is one
            camera/spatial-patch pair.
        history_mask: Boolean tensor ``[time]``; false entries are left padding.
    """
    groups, time, channels = hidden_states.shape
    if time == 1:
        return torch.zeros_like(hidden_states)

    qkv = attention.qkv(hidden_states).reshape(
        groups, time, 3, attention.num_heads, attention.head_dim
    )
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

    if attention.qk_normalization:
        q = attention.q_norm(
            q.transpose(1, 2).flatten(-2, -1)
        ).view(groups, time, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = attention.k_norm(
            k.transpose(1, 2).flatten(-2, -1)
        ).view(groups, time, attention.num_heads, attention.head_dim).transpose(1, 2)

    scores = (q * attention.scale) @ k.transpose(-2, -1)
    valid = history_mask.to(device=scores.device, dtype=torch.bool)
    causal = torch.ones(time, time, device=scores.device, dtype=torch.bool).tril()
    allowed = causal & valid.unsqueeze(0)
    scores = scores.masked_fill(~allowed.view(1, 1, time, time), torch.finfo(scores.dtype).min)

    # Left-padded query rows have no semantic meaning.  Give them finite logits
    # to avoid NaNs, then explicitly zero their outputs below.
    invalid_queries = ~valid
    if invalid_queries.any():
        scores[:, :, invalid_queries, :] = 0

    weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
    weights = attention.attn_drop(weights)
    output = (weights @ v).transpose(1, 2).reshape(groups, time, channels)
    output = attention.proj(output)
    output = attention.proj_drop(output)
    output[:, invalid_queries, :] = 0
    return output


def _space_time_layer(
    layer,
    hidden_states: torch.Tensor,
    *,
    num_frames: int,
    num_views: int,
    history_mask: torch.Tensor,
    temporal_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one ViT block with additive spatial and causal temporal attention."""
    if num_frames == 1:
        # Exact checkpoint-compatible image path required by MEM.
        return layer(hidden_states)

    _, num_tokens, channels = hidden_states.shape
    if temporal_position is None:
        temporal_position = fixed_relative_temporal_encoding(
            num_frames,
            channels,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif temporal_position.shape != (num_frames, channels):
        raise ValueError(
            "Expected temporal_position shape "
            f"{(num_frames, channels)}, got {tuple(temporal_position.shape)}"
        )
    positioned = hidden_states.view(num_frames, num_views, num_tokens, channels)
    positioned = positioned + temporal_position[:, None, None, :]
    positioned_flat = positioned.reshape(num_frames * num_views, num_tokens, channels)

    normalized = layer.norm1(positioned_flat).to(positioned_flat.dtype)
    spatial_output = layer.attn(normalized)

    temporal_input = normalized.view(num_frames, num_views, num_tokens, channels)
    temporal_input = temporal_input.permute(1, 2, 0, 3).reshape(
        num_views * num_tokens, num_frames, channels
    )
    temporal_output = _causal_temporal_attention(
        layer.attn, temporal_input, history_mask
    )
    temporal_output = temporal_output.view(num_views, num_tokens, num_frames, channels)
    temporal_output = temporal_output.permute(2, 0, 1, 3).reshape_as(spatial_output)

    hidden_states = hidden_states + layer.drop_path1(
        (spatial_output + temporal_output) * layer.ls1
    )
    hidden_states = hidden_states + layer.drop_path2(
        layer.mlp(layer.norm2(hidden_states).to(hidden_states.dtype)) * layer.ls2
    )
    return hidden_states


def extract_temporal_feature(
    chat_model,
    pixel_values: torch.Tensor,
    *,
    num_frames: int,
    num_views: int,
    history_mask: torch.Tensor,
    temporal_layer_interval: int = 4,
    drop_past_after_layer: int | None = None,
) -> torch.Tensor:
    """Encode ``[time, view]`` images and return current-view image tokens.

    ``pixel_values`` must be flattened in time-major order and contain exactly
    one image tile for every time/view pair.  The result shape is the same as
    ``chat_model.extract_feature`` applied to the current views.
    """
    if temporal_layer_interval < 1:
        raise ValueError("temporal_layer_interval must be at least 1")
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected pixel_values [T*V,C,H,W], got {tuple(pixel_values.shape)}")
    if pixel_values.shape[0] != num_frames * num_views:
        raise ValueError(
            f"Expected {num_frames * num_views} images for T={num_frames}, V={num_views}; "
            f"got {pixel_values.shape[0]}"
        )
    history_mask = torch.as_tensor(history_mask, device=pixel_values.device, dtype=torch.bool)
    if history_mask.shape != (num_frames,):
        raise ValueError(f"Expected history_mask shape {(num_frames,)}, got {tuple(history_mask.shape)}")
    if not bool(history_mask[-1]):
        raise ValueError("The current (last) memory frame must be valid")

    # Dataset and online memory use left padding only.  Padded observations are
    # guaranteed not to affect valid queries, so removing them before the ViT
    # is both semantically exact and much cheaper near episode boundaries.
    if bool((history_mask[:-1] & ~history_mask[1:]).any()):
        raise ValueError("history_mask must contain only a false left-padding prefix")
    valid_frame_count = int(history_mask.sum().item())
    if valid_frame_count < num_frames:
        pixel_values = pixel_values.reshape(
            num_frames, num_views, *pixel_values.shape[1:]
        )[-valid_frame_count:].reshape(
            valid_frame_count * num_views, *pixel_values.shape[1:]
        )
        num_frames = valid_frame_count
        history_mask = torch.ones(
            num_frames, dtype=torch.bool, device=pixel_values.device
        )

    if num_frames == 1:
        return chat_model.extract_feature(pixel_values)

    vision_model = chat_model.vision_model
    layers = vision_model.encoder.layers
    num_layers = len(layers)
    select_layer = int(getattr(chat_model, "select_layer", -1))
    target_state_index = (
        select_layer if select_layer >= 0 else num_layers + 1 + select_layer
    )
    if not 0 <= target_state_index <= num_layers:
        raise ValueError(
            f"select_layer={select_layer} is invalid for {num_layers} ViT layers"
        )

    if drop_past_after_layer is not None:
        drop_past_after_layer = int(drop_past_after_layer)
        if drop_past_after_layer <= 0:
            drop_past_after_layer = None
        elif drop_past_after_layer > num_layers:
            raise ValueError(
                "drop_past_after_layer cannot exceed the ViT depth: "
                f"{drop_past_after_layer} > {num_layers}"
            )
        elif drop_past_after_layer % temporal_layer_interval != 0:
            raise ValueError(
                "drop_past_after_layer must be a multiple of "
                "temporal_layer_interval"
            )

    hidden_states = vision_model.embeddings(pixel_values)
    temporal_position = fixed_relative_temporal_encoding(
        num_frames,
        hidden_states.shape[-1],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    history_dropped = False
    for layer_index, layer in enumerate(layers[:target_state_index]):
        layer_number = layer_index + 1
        use_temporal_attention = (
            not history_dropped
            and layer_number % temporal_layer_interval == 0
        )
        if use_temporal_attention:
            if vision_model.encoder.gradient_checkpointing and vision_model.training:
                def temporal_forward(states, current_layer=layer):
                    return _space_time_layer(
                        current_layer,
                        states,
                        num_frames=num_frames,
                        num_views=num_views,
                        history_mask=history_mask,
                        temporal_position=temporal_position,
                    )

                hidden_states = torch.utils.checkpoint.checkpoint(
                    temporal_forward, hidden_states, use_reentrant=False
                )
            else:
                hidden_states = _space_time_layer(
                    layer,
                    hidden_states,
                    num_frames=num_frames,
                    num_views=num_views,
                    history_mask=history_mask,
                    temporal_position=temporal_position,
                )
        elif vision_model.encoder.gradient_checkpointing and vision_model.training:
            hidden_states = torch.utils.checkpoint.checkpoint(
                layer, hidden_states, use_reentrant=False
            )
        else:
            hidden_states = layer(hidden_states)

        # MEM Figure 4: once the last temporal layer has compressed history
        # into the current observation, upper ViT layers process current tokens
        # only instead of carrying all past-observation tokens to the top.
        if (
            not history_dropped
            and drop_past_after_layer is not None
            and layer_number == drop_past_after_layer
        ):
            hidden_states = hidden_states.reshape(
                num_frames, num_views, *hidden_states.shape[1:]
            )[-1]
            history_dropped = True

    if not history_dropped:
        hidden_states = hidden_states.reshape(
            num_frames, num_views, *hidden_states.shape[1:]
        )[-1]
    hidden_states = hidden_states[:, 1:, :]

    height = width = int(hidden_states.shape[1] ** 0.5)
    if height * width != hidden_states.shape[1]:
        raise ValueError("InternVL patch token count is not square")
    hidden_states = hidden_states.reshape(num_views, height, width, -1)
    hidden_states = chat_model.pixel_shuffle(
        hidden_states, scale_factor=chat_model.downsample_ratio
    )
    hidden_states = hidden_states.reshape(num_views, -1, hidden_states.shape[-1])
    return chat_model.mlp1(hidden_states)
