import sys
import os
import math
from contextlib import nullcontext
from torch import amp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from Evo1 import EVO1
from accelerate import Accelerator 
import logging
from datetime import datetime
import argparse
from accelerate import Accelerator, DistributedType
import json
import shutil
import copy
from torch.optim import AdamW

import warnings

# 单卡 LoRA 只训练约 1.49M 参数。此时 ZeRO-2 无法跨 GPU 分片，却会保留
# DeepSpeed 包装器和通信缓冲，8 GB 显卡反而更容易 OOM。兼容用户沿用旧命令：
# 若 accelerate 仍带有 --use_deepspeed，则在构造 Accelerator 前自动回到
# 普通单卡路径。全量微调和显式要求保留 DeepSpeed 的诊断运行不受影响。
_single_gpu_deepspeed_disabled = False
if (
    os.environ.get("ACCELERATE_USE_DEEPSPEED", "").lower() == "true"
    and int(os.environ.get("WORLD_SIZE", "1")) <= 1
    and "--no-use_lora" not in sys.argv
    and "--allow_single_gpu_deepspeed" not in sys.argv
):
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ.pop("DEEPSPEED_CONFIG_FILE", None)
    _single_gpu_deepspeed_disabled = True

accelerator = Accelerator()
wandb = None
swanlab = None

def get_with_warning(config: dict, key: str, default):
    if key in config:
        return config[key]
    else:
        warnings.warn(f"'{key}' not found in config, using default: {default!r}")
        return default


def inspect_named_submodules(module_dict: dict, verbose: bool = True):

    total_all, trainable_all = 0, 0
    logging.info("\n Parameter Inspection by Module:")
    logging.info("=" * 70)
    for module_name, module in module_dict.items():
        total, trainable = 0, 0
        logging.info(f"\n Module: {module_name}")
        logging.info("-" * 70)
        for name, param in module.named_parameters():
            num_params = param.numel()
            total += num_params
            if param.requires_grad:
                trainable += num_params
                if verbose:
                    logging.info(f"Trainable {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
            elif verbose:
                logging.info(f"Frozen {name:55s} | shape: {str(tuple(param.shape)):20s} | {num_params/1e6:6.2f}M")
        logging.info("-" * 70)
        logging.info(f"Total     : {total / 1e6:.2f}M")
        logging.info(f"Trainable : {trainable / 1e6:.2f}M")
        logging.info(f"Frozen    : {(total - trainable) / 1e6:.2f}M")
        total_all += total
        trainable_all += trainable
    logging.info("=" * 70)
    logging.info(f"ALL TOTAL     : {total_all / 1e6:.2f}M")
    logging.info(f"ALL TRAINABLE : {trainable_all / 1e6:.2f}M")
    logging.info(f"ALL FROZEN    : {(total_all - trainable_all) / 1e6:.2f}M")
    logging.info("=" * 70)


def custom_collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    images = [item["images"] for item in batch]
    states = torch.stack([item["state"] for item in batch], dim=0)
    actions = torch.stack([item["action"] for item in batch], dim=0)
    action_mask = torch.stack([item["action_mask"] for item in batch], dim=0)
    image_masks = torch.stack([item["image_mask"] for item in batch], dim=0)
    state_mask = torch.stack([item["state_mask"] for item in batch], dim=0)
    history_mask = torch.stack([item["history_mask"] for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)

    return {
        "prompts": prompts,
        "images": images,
        "states": states,
        "actions": actions,
        "action_mask": action_mask,
        "state_mask": state_mask,
        "history_mask": history_mask,
        "image_masks": image_masks,
        "embodiment_ids": embodiment_ids
    }

def get_lr_lambda(warmup_steps, total_steps, resume_step=0):
    def lr_lambda(current_step):
        current_step += resume_step  
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda
    
def setup_logging(log_dir: str) -> str:
    from datetime import datetime
    import logging, os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_log_{timestamp}.log")
    if accelerator is None or accelerator.is_main_process:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging to: {log_path}")
    return log_path

def init_wandb(config: dict, accelerator: Accelerator):
    global wandb

    if accelerator.is_main_process:
        if get_with_warning(config, "disable_wandb", False):
            return
        import wandb as wandb_module

        wandb = wandb_module

        wandb.init(
            project=get_with_warning(config, "wandb_project", "default_run"),
            name=get_with_warning(config, "run_name", "default_run"),
            config=config,
            dir=get_with_warning(config, "save_dir", "checkpoints"),
            mode="offline",
        )

        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

def init_swanlab(config: dict, accelerator: Accelerator):
    global swanlab

    if accelerator is None or accelerator.is_main_process:
        if get_with_warning(config, "disable_swanlab", False):
            return
        import swanlab as swanlab_module

        swanlab = swanlab_module
        swanlab.init(
            project=config.get("wandb_project", "default_run"),
            name=config.get("run_name", "default_run"),
            config=config
        )

def prepare_dataset(config: dict) -> torch.utils.data.Dataset:
    dataset_type = get_with_warning(config, "dataset_type", "lerobot")
    image_size = get_with_warning(config, "image_size", 448)
    max_samples = get_with_warning(config, "max_samples_per_file", None)
    horizon = get_with_warning(config, "horizon", 14)
    binarize_gripper = get_with_warning(config, "binarize_gripper", False)
    use_augmentation = get_with_warning(config, "use_augmentation", False)
    overwrite_horizon_cache = get_with_warning(config, "overwrite_horizon_cache", False)
    memory_frames = get_with_warning(config, "memory_frames", 6)
    memory_stride_steps = get_with_warning(config, "memory_stride_steps", 5)
    memory_stride_seconds = get_with_warning(config, "memory_stride_seconds", 1.0)
    if dataset_type == "lerobot":
        from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset 
        import yaml
        with open(config.get("dataset_config_path"), 'r') as f:
            dataset_config = yaml.safe_load(f)

        dataset = LeRobotDataset(
            config=dataset_config,
            image_size=image_size,
            max_samples_per_file=max_samples,
            action_horizon=horizon,
            binarize_gripper=binarize_gripper,
            use_augmentation=use_augmentation,
            overwrite_horizon_cache=overwrite_horizon_cache,
            memory_frames=memory_frames,
            memory_stride_steps=memory_stride_steps,
            memory_stride_seconds=memory_stride_seconds,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Loaded {len(dataset)} samples from {config['data_paths']} ({dataset_type})")
    return dataset


def prepare_dataloader(dataset, config: dict) -> DataLoader:
    batch_size = get_with_warning(config, "batch_size", 8)
    num_workers = get_with_warning(config, "num_workers", 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Initialized dataloader with batch size {batch_size}")
    return dataloader


def check_numerical_stability(step: int, **named_tensors) -> bool:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f"[Step {step}] Non-finite value detected in {name}"
            )
    return True

def log_training_step(
    step, loss, total_norm, clipped_norm, scheduler, dataloader, accelerator, config
):
    current_epoch = step / len(dataloader)
    if accelerator is None or accelerator.is_main_process:
        logging.info(f"Estimated Epoch: {current_epoch:.2f}")
        logging.info(f"[Step {step}] Loss: {loss.item():.4f}")
        metrics = {
            "step": step,
            "loss": loss.item(),
            "current_epoch": current_epoch,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if not config.get("disable_wandb", False):
            wandb.log(metrics)
        if not config.get("disable_swanlab", False):
            swanlab.log(metrics)

def _pad_normalization_stats_for_evo1(norm_stats, target_dim=24):
    result = copy.deepcopy(norm_stats)

    for robot_key, robot_stats in result.items():
        for key in ("observation.state", "action"):
            item = robot_stats[key]

            min_values = list(item["min"])
            max_values = list(item["max"])

            if len(min_values) != len(max_values):
                raise ValueError(
                    f"{robot_key}/{key}: min/max dimension mismatch"
                )

            current_dim = len(min_values)
            if current_dim > target_dim:
                raise ValueError(
                    f"{robot_key}/{key}: dimension {current_dim} "
                    f"exceeds Evo-1 dimension {target_dim}"
                )

            padding = target_dim - current_dim

            item["min"] = min_values + [-1.0] * padding
            item["max"] = max_values + [1.0] * padding

    return result

def save_checkpoint(
    save_dir,
    step,
    model_engine,
    loss,
    accelerator,
    config=None,
    norm_stats=None,
    optimizer=None,
    scheduler=None,
    global_step=None,
):
    tag = f"step_{step}"
    checkpoint_dir = os.path.join(save_dir, tag)

    if accelerator.is_main_process and os.path.exists(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} exists. Removing before overwrite.")
        shutil.rmtree(checkpoint_dir)

    accelerator.wait_for_everyone()

    client_state = {
        # Human-readable tags such as step_best/step_final must not replace the
        # numeric progress required for an exact optimizer/scheduler resume.
        "step": step if global_step is None else int(global_step),
        "best_loss": loss if isinstance(loss, float) else loss.item(),
        "config": config,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    } if accelerator.is_main_process else {} 

    if hasattr(model_engine, "save_checkpoint"):
        model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
    elif accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        module = accelerator.unwrap_model(model_engine)
        checkpoint = {
            "module": module.state_dict(),
            "client_state": client_state,
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, "mp_rank_00_model_states.pt"),
        )
    
    if accelerator.is_main_process:
        if config is not None:
            config_path = os.path.join(checkpoint_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        if norm_stats is not None:
            norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.json")
            with open(norm_stats_path, "w") as f:
                checkpoint_norm_stats = _pad_normalization_stats_for_evo1(norm_stats, target_dim=24,)
                json.dump(checkpoint_norm_stats, f, indent=2)
                
        checkpoint_meta_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint_meta = {
            "type": "ds_model",
            "version": 0.0,
            "checkpoints": "mp_rank_00_model_states.pt"
        }
        with open(checkpoint_meta_path, "w") as f:
            json.dump(checkpoint_meta, f, indent=2)
        logging.info(f"[Rank {accelerator.process_index}] Saved checkpoint to {checkpoint_dir}")


def load_checkpoint_with_deepspeed(
    model_engine,
    load_dir,
    accelerator,
    tag="step_best",
    load_optimizer_states=True,
    resume_pretrain=False,
    allow_missing_lora=False,
    optimizer=None,
):
    if not hasattr(model_engine, "load_checkpoint"):
        checkpoint_path = os.path.join(load_dir, tag, "mp_rank_00_model_states.pt")
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        module_state = checkpoint.get("module", checkpoint)
        incompatible = accelerator.unwrap_model(model_engine).load_state_dict(
            module_state, strict=not allow_missing_lora
        )
        if allow_missing_lora:
            invalid_missing = [
                key for key in incompatible.missing_keys if ".lora_" not in key
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "基础 checkpoint 与 LoRA 模型存在非适配器键差异："
                    f"missing={invalid_missing}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
        if load_optimizer_states and not resume_pretrain:
            if optimizer is None:
                raise RuntimeError(
                    "恢复普通 AdamW checkpoint 时必须提供 optimizer"
                )
            if "optimizer" not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} 不包含 optimizer state，"
                    "不能作为完整续训点；请仅把它用于 --resume_pretrain 初始化"
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
        stored_client_state = checkpoint.get("client_state", {})
        client_state = {
            "step": stored_client_state.get(
                "step", checkpoint.get("step", checkpoint.get("global_steps", 0))
            ),
            "best_loss": stored_client_state.get(
                "best_loss", checkpoint.get("best_loss", float("inf"))
            ),
            "config": stored_client_state.get(
                "config", checkpoint.get("config", {})
            ),
            "scheduler": stored_client_state.get("scheduler"),
        }
        if accelerator.is_main_process:
            optimizer_scope = (
                "including optimizer state"
                if load_optimizer_states and not resume_pretrain
                else "model weights only"
            )
            logging.info(
                "Loaded regular checkpoint from %s (%s)",
                checkpoint_path,
                optimizer_scope,
            )
        try:
            start_step = int(client_state.get("step", 0) or 0)
        except (TypeError, ValueError):
            start_step = int(checkpoint.get("global_steps", 0) or 0)
        return start_step, client_state

    try:
        load_path, client_state = model_engine.load_checkpoint(
            load_dir,
            tag=tag,
            load_module_strict=not allow_missing_lora,
            load_optimizer_states=load_optimizer_states and not resume_pretrain,
            load_lr_scheduler_states=load_optimizer_states and not resume_pretrain
        )
        if accelerator.is_main_process:
            state_scope = (
                "including optimizer and scheduler states"
                if load_optimizer_states and not resume_pretrain
                else "model weights only"
            )
            logging.info(
                "Loaded DeepSpeed checkpoint from %s/%s (%s)",
                load_dir,
                tag,
                state_scope,
            )
        return client_state.get("step", 0), client_state
        
    except Exception as e:
        if accelerator.is_main_process:
            logging.warning(f"World size mismatch detected: {str(e)}")
            logging.warning("Attempting to load only model weights (skipping optimizer states)...")
        try:
            load_path, client_state = model_engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_module_strict=not allow_missing_lora,
                load_optimizer_states=False,
                load_lr_scheduler_states=False
            )
            if accelerator.is_main_process:
                logging.info(f"Loaded DeepSpeed checkpoint from {load_dir}/{tag} (model weights only)")
            return client_state.get("step", 0), client_state
            
        except Exception as e2:
            if accelerator.is_main_process:
                logging.error(f"Failed to load checkpoint even without optimizer states: {str(e2)}")
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {load_dir} with tag {tag}: {str(e2)}")

    

def get_and_clip_grad_norm(accelerator, model, loss, max_norm: float = 1.0):

    if hasattr(accelerator, "get_global_grad_norm") and hasattr(accelerator, "clip_grad_norm_"):
       
        total_norm = accelerator.get_global_grad_norm()
        accelerator.clip_grad_norm_(model.parameters(), max_norm)
        clipped_norm = accelerator.get_global_grad_norm()
    else:
 
        grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(grad_norms) == 0:
            total_norm = torch.tensor(0.0, device=loss.device)
        else:
            total_norm = torch.norm(torch.stack(grad_norms), 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        clipped_grad_norms = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if len(clipped_grad_norms) == 0:
            clipped_norm = torch.tensor(0.0, device=loss.device)
        else:
            clipped_norm = torch.norm(torch.stack(clipped_grad_norms), 2)

    return total_norm, clipped_norm

def build_param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: 
            continue
        is_bias = n.endswith("bias") or ".bias" in n
        is_norm = (p.dim() == 1) or ("norm" in n.lower())
        is_lora = ".lora_" in n
        (no_decay if is_bias or is_norm or is_lora else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]

def train(config):
    # 命令行和直接调用 train(config) 使用同一默认值；EVO1(config) 本身仍以
    # 缺省关闭保持旧推理 checkpoint 的结构兼容性。
    config.setdefault("use_lora", True)
    config.setdefault("lora_rank", 8)
    config.setdefault("lora_alpha", 16.0)
    config.setdefault("lora_dropout", 0.0)
    config.setdefault("lora_targets", "vision,action")
    config.setdefault("lora_train_bias_norm", True)


    # === Set logging ===
    save_dir = get_with_warning(config, "save_dir", "checkpoints")
    log_path = setup_logging(save_dir)
    if _single_gpu_deepspeed_disabled and accelerator.is_main_process:
        logging.warning(
            "Detected single-GPU LoRA training launched with DeepSpeed; "
            "automatically using the lower-memory Accelerate + fused AdamW path. "
            "Pass --allow_single_gpu_deepspeed only for an intentional diagnostic."
        )
    
    # === WandB and Swanlab ===
    init_wandb(config, accelerator)
    init_swanlab(config, accelerator)

    # === Debug mode ===
    if get_with_warning(config, "debug", False):
        torch.autograd.set_detect_anomaly(True)

    # === Dataset ===
    dataset = prepare_dataset(config)

    # === DataLoader ===
    dataloader = prepare_dataloader(dataset, config)

    # ``resume_pretrain`` 是跨结构初始化（例如单帧基础模型 -> π-MEM
    # LoRA），必须在 DeepSpeed 包装模型之前载入。否则 ZeRO 会依据新模型的
    # 冻结参数表解释旧 checkpoint，并可能在原 MHA 权重上触发 KeyError。
    resume = get_with_warning(config, "resume", False)
    resume_path = get_with_warning(config, "resume_path", None)
    resume_pretrain = get_with_warning(config, "resume_pretrain", False)
    if resume != bool(resume_path):
        raise ValueError(
            "Inconsistent resume configuration: --resume and --resume_path "
            "must be set together."
        )
    if resume_pretrain and not resume:
        raise ValueError("--resume_pretrain requires --resume and --resume_path.")

    resume_dir = resume_tag = None
    if resume:
        resume_path = resume_path.rstrip("/")
        resume_dir, resume_tag = os.path.split(resume_path)
        if not resume_dir or not resume_tag:
            raise ValueError(f"Invalid --resume_path: {resume_path!r}")
    if (
        resume_pretrain
        and os.path.realpath(save_dir) == os.path.realpath(resume_dir)
    ):
        raise ValueError(
            "--resume_pretrain 的 --save_dir 不能与基础 checkpoint 目录相同；"
            "请把 LoRA 结果写入单独目录，以免覆盖原模型。"
        )

    # === Model ===
    model = EVO1(config)
    model.train()
    model.set_finetune_flags()

    pretrain_client_state = None
    if resume_pretrain:
        _, pretrain_client_state = load_checkpoint_with_deepspeed(
            model,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=False,
            resume_pretrain=True,
            allow_missing_lora=get_with_warning(config, "use_lora", True),
        )
        if accelerator.is_main_process:
            logging.info(
                "Loaded initialization checkpoint before DeepSpeed wrapping: %s/%s",
                resume_dir,
                resume_tag,
            )

    lr = get_with_warning(config, "lr", 1e-5)
    wd = get_with_warning(config, "weight_decay", 1e-5)
    fused_adamw = bool(get_with_warning(config, "fused_adamw", True))
    fused_adamw = fused_adamw and torch.cuda.is_available()
    optimizer = AdamW(
        build_param_groups(model, wd),
        lr=lr,
        fused=fused_adamw,
    )
    if accelerator.is_main_process:
        logging.info(
            "Optimizer=AdamW, fused=%s, lr=%s, weight_decay=%s",
            fused_adamw,
            lr,
            wd,
        )


    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model_engine = model  
  
    if accelerator.is_main_process:
        logging.info("Initialized with Accelerate")
        prepared_optimizer = getattr(model_engine, "optimizer", optimizer)
        base_optimizer = getattr(prepared_optimizer, "optimizer", prepared_optimizer)
        zero_stage = getattr(model_engine, "zero_optimization_stage", None)
        zero_stage = zero_stage() if callable(zero_stage) else None
        logging.info(
            "Prepared optimizer=%s; base optimizer=%s; DeepSpeed ZeRO stage=%s",
            type(prepared_optimizer).__name__,
            type(base_optimizer).__name__,
            zero_stage,
        )
    
    
    # === Warmup + Cosine Scheduler ===
    max_steps = get_with_warning(config, "max_steps", 1000)
    warmup_steps = get_with_warning(config, "warmup_steps", 300)
    best_start_step = max(1000, warmup_steps)
    
    # === Checkpoint and save path setup ===
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    
    # === Logging and interval settings ===
    log_interval = get_with_warning(config, "log_interval", 100)
    ckpt_interval = get_with_warning(config, "ckpt_interval", 1000)
    max_norm = get_with_warning(config, "grad_clip_norm", 1.0)

    # === Resume training from checkpoint ===
    if resume and not resume_pretrain:
        step, client_state = load_checkpoint_with_deepspeed(
            model_engine,
            load_dir=resume_dir,
            accelerator=accelerator,
            tag=resume_tag,
            load_optimizer_states=True,  
            resume_pretrain=resume_pretrain,
            allow_missing_lora=(
                resume_pretrain and get_with_warning(config, "use_lora", True)
            ),
            optimizer=optimizer,
        )
        best_loss = client_state.get("best_loss", float("inf"))
        if accelerator.is_main_process:
            logging.info(f"Resuming from {resume_dir}/{resume_tag}, step {step}")
    elif resume_pretrain:
        client_state = pretrain_client_state or {}
        # This starts a new π-MEM/LoRA optimization run.  The source
        # checkpoint's best loss belongs to a different model structure and
        # must not prevent the new run from establishing its own step_best.
        best_loss = float("inf")
        step = 0
        if accelerator.is_main_process:
            logging.info(
                "Initialized new π-MEM training from %s/%s",
                resume_dir,
                resume_tag,
            )
    else:
        step = 0
        if accelerator.is_main_process:
            logging.info("Starting fresh training")

    scheduler_state = (
        client_state.get("scheduler")
        if resume and not resume_pretrain
        else None
    )
    scheduler = LambdaLR(
        optimizer,
        get_lr_lambda(
            warmup_steps,
            max_steps,
            resume_step=0 if scheduler_state is not None else step,
        ),
    )
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        for parameter_group, current_lr in zip(
            optimizer.param_groups, scheduler.get_last_lr()
        ):
            parameter_group["lr"] = current_lr
        if accelerator.is_main_process:
            logging.info("Restored scheduler state at step %s", step)

    if accelerator.is_main_process:
        
        inspect_named_submodules({
            "vision_model": model.embedder.model.vision_model,
            "language_model": model.embedder.model.language_model,
            "action_head": model.action_head
        }, verbose=get_with_warning(config, "verbose_parameter_listing", False))

    # === Training Loop ===
    if accelerator.is_main_process:
        logging.info(
            "VLM forward mode=batched: one vision/language forward per physical batch"
        )
    while step < max_steps:
        for batch in tqdm(
            dataloader,
            desc="Training",
            disable=not accelerator.is_main_process,
            miniters=max(1, log_interval),
            mininterval=0.0,
            maxinterval=float("inf"),
        ):
            if step >= max_steps:
                break
            prompts = batch["prompts"]
            images_batch = batch["images"]
            image_masks = batch["image_masks"]
            states = batch["states"].to(dtype=torch.bfloat16)
            actions_gt = batch["actions"].to(dtype=torch.bfloat16)
            action_mask = batch["action_mask"]
            state_mask = batch["state_mask"]
            history_masks = batch["history_mask"]
            embodiment_ids = batch["embodiment_ids"]
            # Encode the complete physical batch in one vision-language pass.
            # The π-MEM encoder keeps batch and time as independent axes, so
            # causal temporal attention never mixes different samples.
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                fused_tokens, fused_mask = model.get_vl_embeddings_batch(
                    images=images_batch,
                    image_mask=image_masks,
                    prompts=prompts,
                    return_cls_only=False,
                    history_mask=history_masks,
                    return_attention_mask=True,
                )
            fused_tokens = fused_tokens.to(dtype=torch.bfloat16)

            forward_context = (
                nullcontext()
                if accelerator.distributed_type == DistributedType.DEEPSPEED
                else torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            )
            with forward_context:

                pred_velocity, noise = model(
                    fused_tokens,
                    state=states,
                    actions_gt=actions_gt,
                    action_mask=action_mask,
                    embodiment_ids=embodiment_ids,
                    history_mask=history_masks,
                    fused_mask=fused_mask,
                )
                
            target_velocity = (actions_gt - noise).view(actions_gt.shape[0], -1)
            
            assert pred_velocity.shape == target_velocity.shape

            valid_actions_per_sample = action_mask.reshape(action_mask.shape[0], -1).sum(dim=1)
            if bool((valid_actions_per_sample == 0).any()):
                raise ValueError(f"[Step {step}] At least one sample has no valid actions. "
                            f"This indicates a problem with the data or mask generation. "
                            f"action_mask shape: {action_mask.shape}, "
                            f"valid counts: {valid_actions_per_sample.tolist()}")
            

            action_mask = action_mask.view(action_mask.shape[0], -1).to(dtype=pred_velocity.dtype)
            squared_error = (pred_velocity - target_velocity).square()
            loss = (squared_error * action_mask).sum() / action_mask.sum().clamp_min(1)

            # === NaN/Inf check ===
            check_numerical_stability(
                step,
                states=states,
                actions_gt=actions_gt,
                fused_tokens=fused_tokens,
                pred_velocity=pred_velocity,
                loss=loss
            )

            # === Backward and optimizer step ===
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            # === Clip grad norm ===
            total_norm, clipped_norm = get_and_clip_grad_norm(accelerator, model, loss, max_norm)
            if not bool(torch.isfinite(total_norm)):
                raise FloatingPointError(
                    f"[Step {step}] Non-finite gradient norm before clipping"
                )
            if not bool(torch.isfinite(clipped_norm)):
                raise FloatingPointError(
                    f"[Step {step}] Non-finite gradient norm after clipping"
                )

            optimizer.step()
            scheduler.step()

            # === Logging ===
            if step % log_interval == 0:
                log_training_step(
                    step,
                    loss,
                    total_norm,
                    clipped_norm,
                    scheduler,
                    dataloader,
                    accelerator,
                    config,
                )
                if accelerator.is_main_process and torch.cuda.is_available():
                    mib = 1024 ** 2
                    logging.info(
                        "[Step %s] CUDA allocated=%.1f MiB, reserved=%.1f MiB, "
                        "peak_allocated=%.1f MiB, peak_reserved=%.1f MiB",
                        step,
                        torch.cuda.memory_allocated() / mib,
                        torch.cuda.memory_reserved() / mib,
                        torch.cuda.max_memory_allocated() / mib,
                        torch.cuda.max_memory_reserved() / mib,
                    )
   
            # === Save best checkpoint ===
            loss_value = loss.item()
            if accelerator.is_main_process:
                # Follow the original MINT logic: warmup steps establish the
                # global best threshold, but do not write step_best yet.
                is_best = loss_value < best_loss
                if is_best:
                    best_loss = loss_value
                is_best_tensor = torch.tensor(int(is_best), device=accelerator.device)
            else:
                is_best_tensor = torch.tensor(0, device=accelerator.device)
            
            if accelerator.distributed_type != DistributedType.NO:
                torch.distributed.broadcast(is_best_tensor, src=0)
            
            if is_best_tensor.item() == 1 and step > best_start_step:
                accelerator.print("start to save best checkpoint")
                save_checkpoint(
                    save_dir,
                    step="best",
                    model_engine=model_engine,
                    loss=loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=step + 1,
                )
                accelerator.print("end to save best checkpoint")
                if accelerator.is_main_process:
                    logging.info(f"Saved best checkpoint at step {step} with loss {loss_value:.6f}")

            step += 1

            # === Save periodic checkpoint ===
            if step % ckpt_interval == 0 and step > 0:
                save_checkpoint(
                    save_dir,
                    step=step,
                    model_engine=model_engine,
                    loss=best_loss,
                    accelerator=accelerator,
                    config=config,
                    norm_stats=dataset.arm2stats_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
         
    # === Save final model ===
    save_checkpoint(
        save_dir,
        step="final",
        model_engine=model_engine,
        loss=best_loss,
        accelerator=accelerator,
        config=config,
        norm_stats=dataset.arm2stats_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=step,
    )
    logging.info(f"Final model saved to step_final/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Evo-1")

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
    parser.add_argument("--action_head", type=str, default="flowmatching", choices=["flowmatching"])
    parser.add_argument("--return_cls_only", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--disable_swanlab", action="store_true", help="Disable SwanLab logging.")

    # Dataset
    parser.add_argument("--dataset_type", type=str, default="lerobot")
    parser.add_argument("--data_paths", type=str, required=False)
    parser.add_argument("--dataset_config_path", type=str, default="/home/user/mujoco+evo/Evo-1/Evo_1/dataset/config.yaml")
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
    parser.add_argument("--overwrite_horizon_cache", action="store_true", default=False,
                        help="Force rebuilding the derived training window cache. "
                             "Normally source fingerprints invalidate it automatically.")
    parser.add_argument("--no-overwrite_horizon_cache", dest="overwrite_horizon_cache",
                        action="store_false",
                        help="Reuse the existing generated horizon cache.")

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
    parser.add_argument("--save_dir", type=str, default="/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1")
    parser.add_argument(
        "--verbose_parameter_listing",
        action="store_true",
        help="Log every individual model parameter instead of module totals only.",
    )

    # Resume
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--resume_pretrain", action="store_true")
   

    # Finetuning
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use LoRA for the selected vision/action modules (default: enabled). "
            "Pass --no-use_lora to recover the previous full/partial-finetuning path."
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
            "Alongside LoRA, train target-module biases, LayerNorm parameters, "
            "and temporal ViT layer scales to reduce underfitting risk "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--finetune_vlm", action="store_true")
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
    # dropout
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.set_defaults(
        disable_wandb=True,
        finetune_action_head=True,
        use_augmentation=True,
        use_state=True,
    )
    args = parser.parse_args()
    config = vars(args)

    try:
        train(config)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logging.info("KeyboardInterrupt received. Cleaning up...")
        sys.exit(0)
