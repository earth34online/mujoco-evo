from __future__ import annotations
# model/internvl3/internvl3_embedder.py
from PIL import Image
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import transformers
from typing import Union, List

from model.internvl3.temporal_vision_encoder import extract_temporal_feature
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def configure_flash_attention(requested: bool = True) -> bool:
    """Return whether FlashAttention is importable, otherwise force fallback.

    Some environments contain flash-attn package metadata but an extension
    compiled against a newer C++ ABI.  Transformers checks only the metadata
    and then crashes while importing Llama.  Probe the extension itself and,
    on failure, make Transformers use its standard PyTorch attention path.
    """
    available = False
    if requested:
        try:
            import flash_attn  # noqa: F401
            available = True
        except (ImportError, OSError) as exc:
            logging.warning("FlashAttention unavailable; using PyTorch attention: %s", exc)
    if not available:
        transformers.utils.is_flash_attn_2_available = lambda: False
        transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    return available

# === Image Transformations ===
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# === Aspect Ratio Handling ===
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size**2 * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

class InternVL3Embedder(nn.Module):
    def __init__(
        self,
        model_name="OpenGVLab/InternVL3-1B",
        image_size=448,
        device="cuda",
        temporal_layer_interval=4,
        temporal_drop_past_after_layer=20,
        pi_mem_attention_mode="legacy_additive",
        use_flash_attn=True,
        gradient_checkpointing=True,
        compact_masked_views=False,
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.temporal_layer_interval = int(temporal_layer_interval)
        self.temporal_drop_past_after_layer = (
            None
            if temporal_drop_past_after_layer is None
            else int(temporal_drop_past_after_layer)
        )
        self.pi_mem_attention_mode = str(pi_mem_attention_mode)
        self.compact_masked_views = bool(compact_masked_views)
        if self.temporal_layer_interval < 1:
            raise ValueError("temporal_layer_interval must be at least 1")
        self.max_text_length = 1024  # InternVL3 supports up to 1024 tokens
        self.transform = build_transform(image_size)
        self.flash_attention_requested = bool(use_flash_attn)
        use_flash_attn = configure_flash_attention(self.flash_attention_requested)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            low_cpu_mem_usage=True,
            _fast_init=False,
        ).to(self.device) 

        vision_layers = self.model.vision_model.encoder.layers
        vision_flash = bool(vision_layers) and all(
            bool(getattr(layer.attn, "use_flash_attn", False))
            for layer in vision_layers
        )
        language_config = getattr(self.model.language_model, "config", None)
        language_backend = getattr(
            language_config, "_attn_implementation", "unknown"
        )
        language_flash = language_backend == "flash_attention_2"
        self.flash_attention_active = vision_flash and language_flash
        logging.info(
            "Attention backend: vision=%s, language=%s",
            "flash-attn" if vision_flash else "standard",
            language_backend,
        )
        if self.flash_attention_requested and use_flash_attn and not self.flash_attention_active:
            logging.warning(
                "FlashAttention imported but was not activated for both InternVL towers"
            )
        
        if hasattr(self.model.language_model, 'model'):
            layers = self.model.language_model.model.layers

        else:
            layers = self.model.language_model.layers
        layers = layers[:14]

        if hasattr(self.model.language_model, 'model'):
            self.model.language_model.model.layers = torch.nn.ModuleList(layers)
        else:
            self.model.language_model.layers = torch.nn.ModuleList(layers)
        self.model.language_model.lm_head = torch.nn.Identity()

        if hasattr(self.model, "vision_model") and hasattr(self.model.vision_model, "encoder"):
            self.model.vision_model.encoder.gradient_checkpointing = bool(
                gradient_checkpointing
            )
        if gradient_checkpointing and hasattr(
            self.model.language_model, "gradient_checkpointing_enable"
        ):
            self.model.language_model.gradient_checkpointing_enable()
        

    def _normalize_memory_images(self, image_tensors):
        """Normalize legacy/current inputs to a ``time x view`` Python grid."""
        if isinstance(image_tensors, torch.Tensor):
            if image_tensors.ndim == 4:
                image_tensors = image_tensors.unsqueeze(0)
            if image_tensors.ndim != 5:
                raise ValueError(
                    "Tensor images must have shape [V,C,H,W] or [T,V,C,H,W], "
                    f"got {tuple(image_tensors.shape)}"
                )
            return [list(frame.unbind(0)) for frame in image_tensors.unbind(0)]

        if not isinstance(image_tensors, (list, tuple)) or not image_tensors:
            raise ValueError("image_tensors must contain at least one image")
        if isinstance(image_tensors[0], (list, tuple)):
            grid = [list(frame) for frame in image_tensors]
        else:
            grid = [list(image_tensors)]
        num_views = len(grid[0])
        if num_views < 1 or any(len(frame) != num_views for frame in grid):
            raise ValueError("All memory timesteps must contain the same positive number of views")
        return grid

    def _normalize_memory_image_batch(self, image_tensors_batch):
        """Normalize inputs to a rectangular ``batch x time x view`` grid."""
        if isinstance(image_tensors_batch, torch.Tensor):
            if image_tensors_batch.ndim == 5:
                image_tensors_batch = image_tensors_batch.unsqueeze(0)
            if image_tensors_batch.ndim != 6:
                raise ValueError(
                    "Batched tensor images must have shape [B,T,V,C,H,W], "
                    f"got {tuple(image_tensors_batch.shape)}"
                )
            batch_grid = [
                self._normalize_memory_images(sample)
                for sample in image_tensors_batch.unbind(0)
            ]
        else:
            if not isinstance(image_tensors_batch, (list, tuple)) or not image_tensors_batch:
                raise ValueError("image_tensors_batch must contain at least one sample")
            batch_grid = [
                self._normalize_memory_images(sample)
                for sample in image_tensors_batch
            ]

        num_frames = len(batch_grid[0])
        num_views = len(batch_grid[0][0])
        if num_frames < 1 or num_views < 1:
            raise ValueError("Every sample must contain at least one frame and view")
        for sample in batch_grid:
            if len(sample) != num_frames or any(
                len(frame) != num_views for frame in sample
            ):
                raise ValueError(
                    "All samples in a VLM batch must have the same time/view shape"
                )
        return batch_grid, num_frames, num_views

    def _compact_left_padded_history(self, image_grid, history_mask):
        """Remove masked prefix frames before resize and ViT computation."""
        num_frames = len(image_grid)
        if history_mask is None:
            mask = torch.ones(num_frames, dtype=torch.bool, device=self.device)
        else:
            mask = torch.as_tensor(
                history_mask, dtype=torch.bool, device=self.device
            )
        if mask.shape != (num_frames,):
            raise ValueError(
                f"Expected history_mask shape {(num_frames,)}, got {tuple(mask.shape)}"
            )
        if not bool(mask[-1]):
            raise ValueError("The current (last) memory frame must be valid")
        if bool((mask[:-1] & ~mask[1:]).any()):
            raise ValueError(
                "history_mask must contain only a false left-padding prefix"
            )
        valid_frame_count = int(mask.sum().item())
        return image_grid[-valid_frame_count:], torch.ones(
            valid_frame_count, dtype=torch.bool, device=self.device
        )

    def _preprocess_images(self, image_tensors):
        image_grid = self._normalize_memory_images(image_tensors)
        flat_images = [image for frame in image_grid for image in frame]

        # Training/evaluation supply tensors.  Keep that common path entirely
        # in tensor space and resize the whole K*V batch at once.  The previous
        # Tensor -> CPU PIL -> Tensor round trip synchronized CUDA, allocated K*V
        # Python images, and quantized floating-point inputs to 8 bits.
        if all(isinstance(image, torch.Tensor) for image in flat_images):
            chw_images = []
            for image in flat_images:
                image = image.detach()
                if image.ndim != 3:
                    raise ValueError(
                        "Each tensor image must have three dimensions, got "
                        f"{tuple(image.shape)}"
                    )
                if image.shape[0] not in (1, 3, 4):
                    if image.shape[-1] in (1, 3, 4):
                        image = image.permute(2, 0, 1)
                    else:
                        raise ValueError(
                            "Tensor image must be CHW or HWC with 1, 3, or 4 channels, "
                            f"got {tuple(image.shape)}"
                        )
                if image.shape[0] == 1:
                    image = image.expand(3, -1, -1)
                elif image.shape[0] == 4:
                    image = image[:3]
                if image.dtype == torch.uint8:
                    image = image.to(torch.float32).div_(255.0)
                else:
                    image = image.to(torch.float32)
                chw_images.append(image)

            source_device = chw_images[0].device
            if any(image.device != source_device for image in chw_images):
                raise ValueError("All tensor images must be on the same device")
            pixel_values = torch.stack(chw_images, dim=0)
            # Dataset tensors are already 448x448 in the normal training path.
            # Avoid launching an identity bicubic resize over B*K*V images;
            # this is mathematically exact and removes needless CPU work.
            if pixel_values.shape[-2:] != (self.image_size, self.image_size):
                pixel_values = F.interpolate(
                    pixel_values,
                    size=(self.image_size, self.image_size),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            mean = pixel_values.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = pixel_values.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            pixel_values = (pixel_values - mean) / std
            pixel_values = pixel_values.to(dtype=torch.bfloat16, device=self.device)
        else:
            pixel_values_list = []
            for image in flat_images:
                if not isinstance(image, Image.Image):
                    raise TypeError(
                        "A memory batch must contain either all tensors or all PIL images; "
                        f"found {type(image)!r}"
                    )
                tiles = dynamic_preprocess(image, image_size=self.image_size)
                if len(tiles) != 1:
                    raise ValueError(
                        "Temporal encoding currently requires exactly one tile per view"
                    )
                pixel_values_list.append(self.transform(tiles[0]).unsqueeze(0))
            pixel_values = torch.cat(pixel_values_list, dim=0).to(
                dtype=torch.bfloat16, device=self.device
            )
        num_frames = len(image_grid)
        num_views = len(image_grid[0])
        current_num_tiles_list = [1] * num_views
        return pixel_values, current_num_tiles_list, num_frames, num_views

    def _build_multimodal_prompt(
        self,
        num_tiles_list: List[int],
        text_prompt: str
    ) -> str:

        prompt = ''
        for i in range(len(num_tiles_list)):
            prompt += f"Image-{i+1}: <image>\n"
        prompt += text_prompt.strip()

        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        for tile_count in num_tiles_list:
            token_count = self.model.num_image_token * tile_count
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * token_count + IMG_END_TOKEN
            prompt = prompt.replace("<image>", image_tokens, 1)

        return prompt
    
    def _prepare_and_fuse_embeddings(
        self,
        prompt: str,
        vit_embeds: torch.Tensor,
        image_mask: torch.Tensor,
        num_tiles_list: List[int]
    ) -> (torch.Tensor, torch.Tensor):
   
        untruncated_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        true_sequence_length = untruncated_ids.shape[1]

        if true_sequence_length > self.max_text_length:
            print("\n" + "="*80)
            print(f" WARNING: Input prompt was TRUNCATED!")
            print(f"   - Max Length Allowed    : {self.max_text_length}")
            print(f"   - Actual Length      : {true_sequence_length}")
            print(f"   - Truncated Prompt (first 100 chars): '{prompt[:100]}...'")
            print("="*80 + "\n")

        # Do not materialize 1024 tokens for every sample.  Truncation still
        # enforces the model limit, while batch-level padding is performed only
        # after individual VLM calls in train.py.
        model_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

       
        img_token_mask = input_ids == self.img_context_token_id
        img_token_locations = torch.where(img_token_mask)[1]


        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        selected = input_ids == self.img_context_token_id
        tokens_per_tile = self.model.num_image_token
        expected_image_tokens = sum(num_tiles_list) * tokens_per_tile
        actual_image_tokens = int(selected.sum().item())
        flat_vit_embeds = vit_embeds.reshape(-1, C)
        if actual_image_tokens != expected_image_tokens:
            raise ValueError(
                "Prompt image-token count does not match num_tiles_list: "
                f"{actual_image_tokens} != {expected_image_tokens}"
            )
        if flat_vit_embeds.shape[0] != actual_image_tokens:
            raise ValueError(
                "Vision embedding count does not match prompt image tokens: "
                f"{flat_vit_embeds.shape[0]} != {actual_image_tokens}"
            )
        input_embeds[selected] = flat_vit_embeds.to(
            device=input_embeds.device, dtype=input_embeds.dtype
        )

        current_token_idx = 0
        for i in range(len(image_mask)):
           
            num_tiles_for_this_image = num_tiles_list[i]
            num_tokens_for_this_image = num_tiles_for_this_image * tokens_per_tile
       
            if not image_mask[i]:
                
                start_idx = img_token_locations[current_token_idx]
                end_idx = start_idx + num_tokens_for_this_image
               
                attention_mask[0, start_idx:end_idx] = 0
    
            current_token_idx += num_tokens_for_this_image

        input_embeds = input_embeds.reshape(B, N, C)
        return input_embeds, attention_mask

    def _prepare_and_fuse_embeddings_batch(
        self,
        prompts: List[str],
        vit_embeds_by_sample: List[torch.Tensor],
        image_masks_by_sample: List[torch.Tensor],
        num_tiles_by_sample: List[List[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and inject visual tokens for the complete physical batch."""
        batch_size = len(prompts)
        if not (
            len(vit_embeds_by_sample)
            == len(image_masks_by_sample)
            == len(num_tiles_by_sample)
            == batch_size
        ):
            raise ValueError("Batched prompt, vision, mask, and tile counts must match")

        untruncated = self.tokenizer(
            prompts, padding=False, truncation=False
        )["input_ids"]
        longest = max(len(ids) for ids in untruncated)
        if longest > self.max_text_length:
            logging.warning(
                "A batched multimodal prompt exceeds max length: %d > %d",
                longest,
                self.max_text_length,
            )

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        ).to(self.device)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        image_token_mask = input_ids == self.img_context_token_id
        input_embeds = self.model.language_model.get_input_embeddings()(
            input_ids
        ).clone()
        _, _, channels = input_embeds.shape
        tokens_per_tile = self.model.num_image_token

        for batch_index in range(batch_size):
            locations = torch.where(image_token_mask[batch_index])[0]
            expected_tokens = (
                sum(num_tiles_by_sample[batch_index]) * tokens_per_tile
            )
            if int(locations.numel()) != expected_tokens:
                raise ValueError(
                    "Prompt image-token count does not match tiles for sample "
                    f"{batch_index}: {int(locations.numel())} != {expected_tokens}"
                )
            sample_vit = vit_embeds_by_sample[batch_index].reshape(-1, channels)
            if sample_vit.shape[0] != expected_tokens:
                raise ValueError(
                    "Vision embedding count does not match prompt image tokens for "
                    f"sample {batch_index}: {sample_vit.shape[0]} != {expected_tokens}"
                )
            input_embeds[batch_index, locations] = sample_vit.to(
                device=input_embeds.device, dtype=input_embeds.dtype
            )

            image_mask = image_masks_by_sample[batch_index]
            if len(image_mask) != len(num_tiles_by_sample[batch_index]):
                raise ValueError(
                    f"Image mask/tile mismatch for sample {batch_index}"
                )
            token_offset = 0
            for is_valid, tile_count in zip(
                image_mask, num_tiles_by_sample[batch_index]
            ):
                image_token_count = tile_count * tokens_per_tile
                if not bool(is_valid):
                    attention_mask[
                        batch_index,
                        locations[token_offset : token_offset + image_token_count],
                    ] = 0
                token_offset += image_token_count

        return input_embeds, attention_mask

    def get_fused_image_text_embedding_batch(
        self,
        image_tensors_batch,
        image_masks: torch.Tensor,
        text_prompts: List[str],
        return_cls_only: bool = True,
        history_masks: Union[torch.Tensor, None] = None,
        return_attention_mask: bool = False,
    ):
        """Run one batched ViT and one batched language-model forward.

        Images are laid out as ``[B,K,V,C,H,W]``.  The temporal encoder keeps
        every sample isolated while sharing the large CUDA kernels across B.
        """
        batch_grid, num_frames, total_views = self._normalize_memory_image_batch(
            image_tensors_batch
        )
        batch_size = len(batch_grid)
        if len(text_prompts) != batch_size:
            raise ValueError(
                f"Expected {batch_size} prompts, got {len(text_prompts)}"
            )

        history_masks = torch.ones(
            batch_size,
            num_frames,
            dtype=torch.bool,
            device=self.device,
        ) if history_masks is None else torch.as_tensor(
            history_masks, dtype=torch.bool, device=self.device
        )
        if history_masks.ndim == 1 and batch_size == 1:
            history_masks = history_masks.unsqueeze(0)
        if history_masks.shape != (batch_size, num_frames):
            raise ValueError(
                f"Expected history_masks shape {(batch_size, num_frames)}, "
                f"got {tuple(history_masks.shape)}"
            )
        if not bool(history_masks[:, -1].all()):
            raise ValueError("Every sample's current memory frame must be valid")
        if bool((history_masks[:, :-1] & ~history_masks[:, 1:]).any()):
            raise ValueError("history_masks must contain only false left-padding prefixes")

        image_masks = torch.as_tensor(image_masks, device=self.device)
        if image_masks.ndim == 1 and batch_size == 1:
            image_masks = image_masks.unsqueeze(0)
        elif (
            image_masks.ndim == 2
            and batch_size == 1
            and image_masks.shape != (1, total_views)
        ):
            image_masks = image_masks[-1:].contiguous()
        elif image_masks.ndim == 3:
            image_masks = image_masks[:, -1]
        if image_masks.shape != (batch_size, total_views):
            raise ValueError(
                f"Expected current image_masks shape {(batch_size, total_views)}, "
                f"got {tuple(image_masks.shape)}"
            )
        image_masks = image_masks.to(torch.bool)
        if not bool(image_masks.any(dim=1).all()):
            raise ValueError("Every sample must contain at least one valid camera view")

        # Use the union of real view slots so one rectangular ViT batch can
        # support datasets whose camera availability differs by sample.
        union_view_indices = torch.where(image_masks.any(dim=0))[0].tolist()
        selected_grid = [
            [
                [frame[view_index] for view_index in union_view_indices]
                for frame in sample
            ]
            for sample in batch_grid
        ]
        flattened_grid = [
            frame for sample in selected_grid for frame in sample
        ]
        pixel_values, _, flat_frames, union_views = self._preprocess_images(
            flattened_grid
        )
        if flat_frames != batch_size * num_frames:
            raise RuntimeError("Batched preprocessing changed the frame count")

        valid_vit_embeds = extract_temporal_feature(
            self.model,
            pixel_values,
            batch_size=batch_size,
            num_frames=num_frames,
            num_views=union_views,
            history_mask=history_masks,
            temporal_layer_interval=self.temporal_layer_interval,
            drop_past_after_layer=self.temporal_drop_past_after_layer,
            attention_mode=self.pi_mem_attention_mode,
        )
        valid_vit_embeds = valid_vit_embeds.reshape(
            batch_size,
            union_views,
            valid_vit_embeds.shape[-2],
            valid_vit_embeds.shape[-1],
        )

        vit_embeds_by_sample = []
        fusion_masks = []
        num_tiles_by_sample = []
        for batch_index in range(batch_size):
            if self.compact_masked_views:
                positions = [
                    union_position
                    for union_position, original_view in enumerate(union_view_indices)
                    if bool(image_masks[batch_index, original_view])
                ]
                sample_vit = valid_vit_embeds[batch_index, positions]
                sample_mask = torch.ones(
                    len(positions), dtype=torch.bool, device=self.device
                )
                sample_tiles = [1] * len(positions)
            else:
                sample_vit = torch.zeros(
                    total_views,
                    valid_vit_embeds.shape[-2],
                    valid_vit_embeds.shape[-1],
                    dtype=valid_vit_embeds.dtype,
                    device=valid_vit_embeds.device,
                )
                sample_vit[union_view_indices] = valid_vit_embeds[batch_index]
                sample_mask = image_masks[batch_index]
                sample_tiles = [1] * total_views
            vit_embeds_by_sample.append(sample_vit)
            fusion_masks.append(sample_mask)
            num_tiles_by_sample.append(sample_tiles)

        prompts = [
            self._build_multimodal_prompt(tiles, prompt)
            for tiles, prompt in zip(num_tiles_by_sample, text_prompts)
        ]
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings_batch(
            prompts,
            vit_embeds_by_sample,
            fusion_masks,
            num_tiles_by_sample,
        )
        language_backbone = getattr(
            self.model.language_model, "model", self.model.language_model
        )
        outputs = language_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        fused_hidden = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        ).to(torch.float32)

        if return_cls_only:
            result = fused_hidden[:, 0, :]
            result_mask = torch.ones(
                batch_size, 1, dtype=torch.bool, device=fused_hidden.device
            )
        else:
            result = fused_hidden
            result_mask = attention_mask.to(torch.bool)
        if return_attention_mask:
            return result, result_mask
        return result


    def get_fused_image_text_embedding_from_tensor_images(
        self,
        image_tensors,
        image_mask: torch.Tensor,
        text_prompt: str,
        return_cls_only: bool = True,
        history_mask: Union[torch.Tensor, None] = None,
        return_attention_mask: bool = False,
    ):
        image_grid = self._normalize_memory_images(image_tensors)
        image_grid, history_mask = self._compact_left_padded_history(
            image_grid, history_mask
        )
        num_frames = len(image_grid)
        total_views = len(image_grid[0])
        image_mask = torch.as_tensor(image_mask, device=self.device)
        if image_mask.ndim == 2:
            image_mask = image_mask[-1]
        if image_mask.shape != (total_views,):
            raise ValueError(
                f"Expected current image_mask shape {(total_views,)}, got {tuple(image_mask.shape)}"
            )
        valid_view_indices = torch.where(image_mask.to(torch.bool))[0].tolist()
        if not valid_view_indices:
            raise ValueError("At least one current camera view must be valid")

        # Masked camera slots are padding, so encoding their K zero images only
        # wastes ViT memory.  Encode real views and restore zero embeddings for
        # masked prompt slots before language fusion.
        valid_grid = [
            [frame[view_index] for view_index in valid_view_indices]
            for frame in image_grid
        ]
        pixel_values, _, processed_frames, valid_views = self._preprocess_images(valid_grid)
        if processed_frames != num_frames:
            raise RuntimeError("Temporal preprocessing changed the frame count")
        num_tiles_list = [1] * total_views

       
        if pixel_values.shape[0] == 0:
           
            print("Warning: No valid images to process after masking.")

        valid_vit_embeds = extract_temporal_feature(
            self.model,
            pixel_values,
            num_frames=num_frames,
            num_views=valid_views,
            history_mask=history_mask,
            temporal_layer_interval=self.temporal_layer_interval,
            drop_past_after_layer=self.temporal_drop_past_after_layer,
            attention_mode=self.pi_mem_attention_mode,
        )
        if self.compact_masked_views:
            # Padding cameras are not observations.  Omitting their 256-token
            # placeholders reduces a one-real/two-padding prompt from roughly
            # 800 tokens to roughly 290, rather than merely masking work after
            # the language backbone has already performed it.
            fused_embeds = valid_vit_embeds
            num_tiles_list = [1] * valid_views
            fusion_image_mask = torch.ones(
                valid_views, dtype=torch.bool, device=self.device
            )
        else:
            fused_embeds = torch.zeros(
                total_views,
                valid_vit_embeds.shape[1],
                valid_vit_embeds.shape[2],
                device=valid_vit_embeds.device,
                dtype=valid_vit_embeds.dtype,
            )
            fused_embeds[valid_view_indices] = valid_vit_embeds
            fusion_image_mask = image_mask
        prompt = self._build_multimodal_prompt(num_tiles_list, text_prompt)
        inputs_embeds, attention_mask = self._prepare_and_fuse_embeddings(
            prompt, fused_embeds, fusion_image_mask, num_tiles_list
        )

        language_backbone = getattr(
            self.model.language_model, "model", self.model.language_model
        )
        outputs = language_backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            fused_hidden = outputs.last_hidden_state.to(torch.float32)
        else:
            fused_hidden = outputs[0].to(torch.float32)

        if return_cls_only:
            result = fused_hidden[:, 0, :]
            result_mask = torch.ones(
                fused_hidden.shape[0], 1, dtype=torch.bool, device=fused_hidden.device
            )
        else:
            result = fused_hidden
            result_mask = attention_mask.to(dtype=torch.bool)
        if return_attention_mask:
            return result, result_mask
        return result
