"""Command-line configuration for the Evo-1 training entry point."""

import argparse
from datetime import datetime
import sys

from mujoco_pickplace.release import CHECKPOINT, PROJECT_ROOT


def parse_training_args(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="Train Evo-1")
    parser.add_argument("--pi-mem", action="store_true",
                        help="Use the released three-task MuJoCo training profile.")

    # Basic config
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run_name", type=str, default="evo1_mujoco_pickplace")
    parser.add_argument("--vlm_name", type=str, default="OpenGVLab/InternVL3-1B")
    parser.add_argument(
        "--use_flash_attn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FlashAttention when its compiled extension is importable; otherwise fall back safely.",
    )
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching", "direct"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--disable_swanlab", action="store_true", help="Disable SwanLab logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, default=str(PROJECT_ROOT / "Evo-1/Evo_1/dataset/config.yaml"))
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional derived-window cache directory. Use a dedicated path "
            "when the dataset combination changes so Task3 cannot replace the "
            "existing Task1 cache."
        ),
    )
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=None,
        help="Optional per-episode sample cap for startup smoke tests.",
    )
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument(
        "--memory_frames",
        type=int,
        default=6,
        help="π-MEM observation count including the current frame (default: 6).",
    )
    parser.add_argument(
        "--memory_stride_steps",
        type=int,
        default=5,
        help="Fallback history spacing in dataset/control steps (default: 5 at 5 Hz).",
    )
    parser.add_argument(
        "--memory_stride_seconds",
        type=float,
        default=1.0,
        help="Prefer timestamp-based history spacing in seconds; set <=0 to use steps.",
    )
    parser.add_argument(
        "--temporal_layer_interval",
        type=int,
        default=4,
        help="Add causal same-patch temporal attention every N ViT layers.",
    )
    parser.add_argument(
        "--temporal_drop_past_after_layer",
        type=int,
        default=20,
        help=(
            "After this temporal ViT layer, discard past-frame tokens and run "
            "upper layers on the current frame only (default: 20; <=0 disables)."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint ViT layers to make K-frame memory training fit limited VRAM.",
    )
    parser.add_argument(
        "--compact_masked_views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Omit padding-camera image tokens before the language model instead "
            "of computing and masking them later (default: enabled)."
        ),
    )
    parser.add_argument("--binarize_gripper", action="store_true", default=False, help="Whether to binarize gripper state/action (default: False).")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation on images")
    parser.add_argument(
        "--mask_terminal_action_padding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Exclude fabricated tail actions from new-run supervision. Exact resume "
              "inherits the checkpoint's setting, including legacy tail supervision."),
    )
    parser.add_argument("--overwrite_horizon_cache", action="store_true", default=False,
                        help="Force rebuilding the derived training window cache. "
                             "Normally source fingerprints invalidate it automatically.")
    parser.add_argument("--no-overwrite_horizon_cache", dest="overwrite_horizon_cache",
                        action="store_false",
                        help="Reuse the existing generated horizon cache.")
    parser.add_argument(
        "--memory_event_sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt into local inverse-square-root event weighting. By default, "
            "shuffle all eligible windows uniformly, as in MINT."
        ),
    )

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument(
        "--fused_adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA fused AdamW on the single-GPU LoRA path (default: enabled).",
    )
    parser.add_argument(
        "--allow_single_gpu_deepspeed",
        action="store_true",
        help=(
            "Keep DeepSpeed on a one-GPU LoRA run. This is intended only for "
            "diagnostics because ZeRO-2 cannot shard across one GPU and may use "
            "more memory than fused AdamW."
        ),
    )


    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "ckpt/evo1_mujoco_pickplace_stage3_retrain"))
    parser.add_argument(
        "--verbose_parameter_listing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show MINT's full parameter listing (default: enabled).",
    )

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")


    # Finetuning
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use LoRA for the selected vision/action modules (default: disabled). "
            "Pass --use_lora explicitly to enable adapter training."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="vision,action",
        help="Comma-separated LoRA targets: vision, language, action.",
    )
    parser.add_argument(
        "--lora_train_bias_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Alongside LoRA, train supported target-module biases and LayerNorm "
            "parameters (default: enabled). Separate temporal LoRA still keeps "
            "shared vision bias, norm, and layer scale frozen."
        ),
    )
    parser.add_argument(
        "--separate_temporal_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply vision LoRA only to temporal matching/value mixing and the "
            "composed memory output while leaving Stage1 spatial Q/K frozen "
            "(default: enabled). "
            "K=1 training uses spatial vision LoRA because temporal adapters "
            "are inactive; K>1 may use --no-separate_temporal_lora for a shared-LoRA ablation."
        ),
    )
    parser.add_argument("--finetune_vlm", action="store_true")
    parser.add_argument(
        "--frozen_embedder_eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Keep an entirely frozen VLM in eval mode during action alignment. "
              "Exact resume inherits the checkpoint's original behavior."),
    )
    parser.add_argument(
        "--finetune_temporal_vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the full VLM is frozen, train the reused attention/norm weights "
            "of temporal ViT layers (default: enabled)."
        ),
    )
    parser.add_argument(
        "--fp32_temporal_parameters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep trainable temporal attention/norm weights in FP32 so plain "
            "AdamW cannot round small BF16 updates to zero (default: enabled)."
        ),
    )
    parser.add_argument("--finetune_action_head", action="store_true")
    parser.add_argument("--use_state", action="store_true",
                        help="Keep the state branch active. Leave unset for the pure-vision setup.")

    # Misc
    parser.add_argument("--per_action_dim", type=int, default=24)
    parser.add_argument("--state_dim", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_num_workers", type=int, default=1,
                        help="Processes for rebuilding derived windows; 1 runs in the current process.")
    # dropout
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.set_defaults(
        disable_wandb=True,
        finetune_action_head=True,
        use_augmentation=True,
        use_state=True,
        # Internal checkpoint architecture marker.  Older checkpoints omit it
        # and retain their legacy additive attention during evaluation.
        pi_mem_attention_mode="composed",
    )
    parser.add_argument("--seed", type=int, default=None, help="Training RNG seed for independent replicates")
    parser.add_argument("--drawer_intent_supervision", action="store_true")
    parser.add_argument("--task3_binary_gripper_supervision", dest="task4_binary_gripper_supervision",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Binary drawer gripper supervision (default with direct actions and drawer intent supervision).")
    if "--pi-mem" in args:
        parser.set_defaults(
            action_head="direct", drawer_intent_supervision=True,
            task4_binary_gripper_supervision=True, memory_event_sampling=True,
            batch_size=2, max_steps=40000, warmup_steps=1000, ckpt_interval=2000,
            log_interval=20, lr=1e-5, dropout=0.1, weight_decay=1e-3,
            memory_frames=6, memory_stride_steps=5, memory_stride_seconds=1.0,
            temporal_layer_interval=4, temporal_drop_past_after_layer=20,
            gradient_checkpointing=True, compact_masked_views=True,
            frozen_embedder_eval=True, mask_terminal_action_padding=True,
            run_name="evo_pi_mem",
            disable_wandb=True, disable_swanlab=True,
            dataset_config_path=str(PROJECT_ROOT / "mujoco_pickplace/pi_mem_release_data.yaml"),
            resume=True, resume_pretrain=True,
            resume_path=str(CHECKPOINT),
            save_dir=str(PROJECT_ROOT / "ckpt" / datetime.now().strftime("evo_pi_mem_%Y%m%d_%H%M%S_%f")),
        )
    return parser.parse_args(args)
