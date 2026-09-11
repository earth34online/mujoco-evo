import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from types import SimpleNamespace
from typing import List, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
from model.internvl3.internvl3_embedder import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
from model.lora import (
    LoRALinear,
    LoRAMultiheadAttention,
    inject_lora_linear_layers,
    is_lora_parameter,
)
import logging
class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.memory_frames = int(config.get("memory_frames", 1))
        self.temporal_layer_interval = int(config.get("temporal_layer_interval", 4))
        self.temporal_drop_past_after_layer = config.get(
            "temporal_drop_past_after_layer", 20
        )
        self.embedder = InternVL3Embedder(
            model_name=vlm_name,
            image_size=int(config.get("image_size", 448)),
            device=self._device,
            temporal_layer_interval=self.temporal_layer_interval,
            temporal_drop_past_after_layer=self.temporal_drop_past_after_layer,
            use_flash_attn=bool(config.get("use_flash_attn", True)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
            compact_masked_views=bool(
                config.get("compact_masked_views", self.memory_frames > 1)
            ),
        )

        action_head_type = config.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.get("action_horizon", config.get("horizon", 16))
            per_action_dim = config.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),    
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                use_state=config.get("use_state", True),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1)
            )).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        # 为保持旧 checkpoint 和外部直接构造 EVO1 的兼容性，模型类只在配置中
        # 明确存在 use_lora=true 时注入；训练 CLI 会默认写入 true。
        self.use_lora = bool(config.get("use_lora", False))
        self.lora_module_names = []
        if self.use_lora:
            self._configure_lora()

    def _configure_lora(self):
        rank = int(self.config.get("lora_rank", 8))
        alpha = float(self.config.get("lora_alpha", 16.0))
        dropout = float(self.config.get("lora_dropout", 0.0))
        raw_targets = self.config.get("lora_targets", "vision,action")
        if isinstance(raw_targets, str):
            targets = {
                item.strip().lower()
                for item in raw_targets.split(",")
                if item.strip()
            }
        else:
            targets = {str(item).strip().lower() for item in raw_targets}
        allowed = {"vision", "language", "action"}
        unknown = targets - allowed
        if unknown:
            raise ValueError(
                f"未知 LoRA 目标 {sorted(unknown)}；允许值为 {sorted(allowed)}"
            )
        if not targets:
            raise ValueError("use_lora=true 时 lora_targets 不能为空")
        self.lora_targets = targets

        if "vision" in targets:
            vision_model = self.embedder.model.vision_model
            if self.config.get("finetune_vlm", False):
                names = inject_lora_linear_layers(
                    vision_model,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    prefix="embedder.model.vision_model",
                    include_multihead_attention=False,
                )
            else:
                names = []
                drop_after = self.temporal_drop_past_after_layer
                for layer_index, layer in enumerate(vision_model.encoder.layers):
                    layer_number = layer_index + 1
                    if layer_number % self.temporal_layer_interval != 0:
                        continue
                    if (
                        drop_after is not None
                        and int(drop_after) > 0
                        and layer_number > int(drop_after)
                    ):
                        continue
                    names.extend(
                        inject_lora_linear_layers(
                            layer.attn,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            prefix=(
                                "embedder.model.vision_model.encoder.layers."
                                f"{layer_index}.attn"
                            ),
                            include_multihead_attention=False,
                        )
                    )
            self.lora_module_names.extend(names)

        if "language" in targets:
            self.lora_module_names.extend(
                inject_lora_linear_layers(
                    self.embedder.model.language_model,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    prefix="embedder.model.language_model",
                    include_multihead_attention=False,
                )
            )

        if "action" in targets:
            self.lora_module_names.extend(
                inject_lora_linear_layers(
                    self.action_head,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    prefix="action_head",
                    include_multihead_attention=True,
                )
            )

        if not self.lora_module_names:
            raise RuntimeError(
                "LoRA 已启用，但没有找到可注入的线性层；请核对 lora_targets"
            )
        # 保存归一化后的结构配置，保证训练 checkpoint 在推理时构造同一结构。
        self.config["lora_targets"] = ",".join(sorted(targets))
        self.config["lora_rank"] = rank
        self.config["lora_alpha"] = alpha
        self.config["lora_dropout"] = dropout

    @staticmethod
    def _enable_adapter_biases(module: nn.Module) -> int:
        """放开已注入 LoRA 投影的偏置，不解冻其基础权重。"""

        trainable = 0
        for child in module.modules():
            if isinstance(child, LoRALinear) and child.bias is not None:
                child.bias.requires_grad = True
                trainable += child.bias.numel()
            elif isinstance(child, LoRAMultiheadAttention):
                for bias in (
                    child.in_proj_bias,
                    child.out_proj.bias,
                    child.bias_k,
                    child.bias_v,
                ):
                    if bias is not None:
                        bias.requires_grad = True
                        trainable += bias.numel()
        return trainable

    @staticmethod
    def _enable_norm_parameters(module: nn.Module) -> int:
        """放开 LayerNorm，给低秩更新保留幅值和偏移校准能力。"""

        trainable = 0
        for child in module.modules():
            if isinstance(child, nn.LayerNorm):
                for parameter in child.parameters(recurse=False):
                    parameter.requires_grad = True
                    trainable += parameter.numel()
        return trainable

    def _enable_lora_support_parameters(self) -> int:
        """启用少量偏置/归一化参数，降低纯低秩适配的欠拟合风险。"""

        if not self.config.get("lora_train_bias_norm", True):
            return 0

        trainable = 0
        if "action" in self.lora_targets:
            trainable += self._enable_adapter_biases(self.action_head)
            trainable += self._enable_norm_parameters(self.action_head)

        if "vision" in self.lora_targets:
            vision_model = self.embedder.model.vision_model
            if self.config.get("finetune_vlm", False):
                trainable += self._enable_adapter_biases(vision_model)
                trainable += self._enable_norm_parameters(vision_model)
            else:
                drop_after = self.temporal_drop_past_after_layer
                for layer_index, layer in enumerate(vision_model.encoder.layers):
                    layer_number = layer_index + 1
                    if layer_number % self.temporal_layer_interval != 0:
                        continue
                    if (
                        drop_after is not None
                        and int(drop_after) > 0
                        and layer_number > int(drop_after)
                    ):
                        continue
                    trainable += self._enable_adapter_biases(layer.attn)
                    for parameter in layer.norm1.parameters():
                        parameter.requires_grad = True
                        trainable += parameter.numel()
                    if hasattr(layer, "ls1"):
                        layer.ls1.requires_grad = True
                        trainable += layer.ls1.numel()

        if "language" in self.lora_targets:
            language_model = self.embedder.model.language_model
            trainable += self._enable_adapter_biases(language_model)
            trainable += self._enable_norm_parameters(language_model)

        return trainable

    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None,
        history_mask: Union[torch.Tensor, None] = None,
        return_attention_mask: bool = False,
    ):

        if return_cls_only is None:
            return_cls_only = self.return_cls_only

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
            history_mask=history_mask,
            return_attention_mask=return_attention_mask,
        )

    def prepare_state(
        self,
        state_input: Union[list, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if state_input is None:
            return None

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        elif state_tensor.ndim == 2:
            if batch_size == 1:
                state_tensor = state_tensor.unsqueeze(0)
            elif state_tensor.shape[0] == batch_size:
                state_tensor = state_tensor.unsqueeze(1)
            else:
                raise ValueError(
                    f"Cannot align state shape {tuple(state_tensor.shape)} with batch {batch_size}"
                )
        elif state_tensor.ndim != 3:
            raise ValueError(
                "State must have shape [D], [T,D], [B,D], or [B,T,D]; "
                f"got {tuple(state_tensor.shape)}"
            )

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
        history_mask: torch.Tensor = None,
        fused_mask: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if actions_gt is None:
            return self.action_head.get_action(
                fused_tokens,
                state=state,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
                history_mask=history_mask,
                fused_mask=fused_mask,
            )
        else:
            return self.action_head(
                fused_tokens,
                state=state,
                actions_gt=actions_gt,
                action_mask=action_mask,
                embodiment_id=embodiment_ids,
                history_mask=history_mask,
                fused_mask=fused_mask,
            )


    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None,
        history_mask: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:

        fused_tokens, fused_mask = self.get_vl_embeddings(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            return_cls_only=return_cls_only,
            history_mask=history_mask,
            return_attention_mask=True,
        )

        state_tensor = self.prepare_state(state_input, batch_size=fused_tokens.shape[0])
        if history_mask is not None:
            history_mask = torch.as_tensor(
                history_mask, dtype=torch.bool, device=self._device
            )
            if history_mask.ndim == 1:
                history_mask = history_mask.unsqueeze(0)
        
        return self.predict_action(
            fused_tokens,
            state_tensor,
            action_mask=action_mask,
            history_mask=history_mask,
            fused_mask=fused_mask,
        )
    

    def forward(
        self,
        fused_tokens,
        state=None,
        actions_gt=None,
        action_mask=None,
        embodiment_ids=None,
        history_mask=None,
        fused_mask=None,
    ):
   

        return self.predict_action(
            fused_tokens,
            state,
            actions_gt,
            action_mask,
            embodiment_ids,
            history_mask,
            fused_mask,
        )

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        config = self.config  
        if self.use_lora:
            # LoRA 模式冻结大矩阵，只训练低秩增量和少量偏置/归一化参数。
            # 后者几乎不增加优化器显存，但比纯 LoRA 更接近原训练路径的
            # 表达能力；可用 --no-lora_train_bias_norm 做严格纯 LoRA 消融。
            self._freeze_module(self.embedder, "VLM (InternVL3) base weights")
            self._freeze_module(self.action_head, "Action Head base weights")
            adapter_trainable = 0
            for name, parameter in self.named_parameters():
                if is_lora_parameter(name):
                    parameter.requires_grad = True
                    adapter_trainable += parameter.numel()
            if adapter_trainable == 0:
                raise RuntimeError("LoRA 已启用，但没有可训练的 LoRA 参数")
            support_trainable = self._enable_lora_support_parameters()
            print(
                f"LoRA enabled: {len(self.lora_module_names)} modules, "
                f"{adapter_trainable / 1e6:.2f}M adapter parameters + "
                f"{support_trainable / 1e6:.2f}M bias/norm parameters..."
            )
            return

        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
            if (
                self.memory_frames > 1
                and config.get("finetune_temporal_vision", True)
            ):
                trainable_layer_count = 0
                for layer_index, layer in enumerate(
                    self.embedder.model.vision_model.encoder.layers
                ):
                    if (layer_index + 1) % self.temporal_layer_interval != 0:
                        continue
                    if (
                        self.temporal_drop_past_after_layer is not None
                        and int(self.temporal_drop_past_after_layer) > 0
                        and layer_index + 1
                        > int(self.temporal_drop_past_after_layer)
                    ):
                        continue
                    # MEM shares these weights between spatial and temporal
                    # attention; selectively unfreezing them learns temporal
                    # use without unfreezing the language backbone.
                    for module in (layer.norm1, layer.attn):
                        for parameter in module.parameters():
                            parameter.requires_grad = True
                    layer.ls1.requires_grad = True
                    if config.get("fp32_temporal_parameters", True):
                        # Plain AdamW has no FP32 master copy for BF16 weights;
                        # at lr=1e-5 an update can round to exactly zero forever.
                        # Keeping only the active trainable temporal blocks in
                        # FP32 is inexpensive and works with or without
                        # DeepSpeed. Autocast still performs BF16 matmuls.
                        layer.norm1.float()
                        layer.attn.float()
                        layer.ls1.data = layer.ls1.data.float()
                    trainable_layer_count += 1
                print(
                    "Finetuning reused attention weights in "
                    f"{trainable_layer_count} temporal vision layers..."
                )
                if config.get("fp32_temporal_parameters", True):
                    print("Keeping trainable temporal weights in FP32...")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")
