# evo1_server_json.py

import argparse
import sys
import os
import asyncio
import base64
import io
import websockets
import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1
from model.lora import merge_lora_weights


class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path
            
        self.target_dim = 24

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < self.target_dim:
                pad = torch.zeros(self.target_dim - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > self.target_dim:
                raise ValueError(
                    f"Input length {x.shape[0]} exceeds expected {self.target_dim}"
                )
            return x

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        for key in ("observation.state", "action"):
            if (
                len(robot_stats[key]["min"]) != self.target_dim
                or len(robot_stats[key]["max"]) != self.target_dim
            ):
                raise ValueError(
                    f"{key} checkpoint statistics must be 24-D. "
                    "This checkpoint predates neutral-bound padding."
                )

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        current_dim = state.shape[-1]
        if current_dim > self.target_dim:
            raise ValueError(
                f"State length {current_dim} exceeds expected {self.target_dim}"
            )
        state_min = self.state_min[:current_dim].to(
            state.device, dtype=state.dtype
        )
        state_max = self.state_max[:current_dim].to(
            state.device, dtype=state.dtype
        )
        normalized = torch.clamp(
            2 * (state - state_min) / (state_max - state_min + 1e-8) - 1,
            -1.0,
            1.0,
        )
        if current_dim < self.target_dim:
            normalized = torch.cat(
                [
                    normalized,
                    torch.zeros(
                        (*normalized.shape[:-1], self.target_dim - current_dim),
                        device=normalized.device,
                        dtype=normalized.dtype,
                    ),
                ],
                dim=-1,
            )
        return normalized

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def load_model_and_normalizer(ckpt_dir):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
    use_state = bool(config.get("use_state", True))
    if not use_state:
        raise ValueError(
            "This MuJoCo evaluation requires image + robot proprioception; "
            "use a checkpoint trained with --use_state."
        )

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    # 保持 Evo-1 原始评估精度设置；π-MEM 只改变观测记忆，不减少流匹配求解步数。
    config["num_inference_timesteps"] = 50
    config["device"] = "cuda"

    print("Building EVO_1 module...", flush=True)
    model = EVO1(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    try:
        checkpoint = torch.load(
            ckpt_path, map_location="cpu", mmap=True, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Applying checkpoint weights...", flush=True)
    model.load_state_dict(checkpoint["module"], strict=True)
    if bool(config.get("use_lora", False)):
        merged_count = merge_lora_weights(model)
        print(
            f"Merged LoRA weights into {merged_count} modules for inference.",
            flush=True,
        )
    print("Model is ready on CUDA.", flush=True)

    print("Loading normalizer...", flush=True)
    normalizer = Normalizer(stats)
    return model, normalizer, use_state


def decode_image_from_list(img_list, image_size=448):
    img_array = np.asarray(img_list, dtype=np.uint8, )
    if img_array.ndim != 3 or img_array.shape[2] != 3:
        raise ValueError(
            f"Legacy list image must have shape [H,W,3], got {img_array.shape}"
        )
    # MuJoCo Renderer and the client both expose RGB.  The old BGR conversion
    # silently swapped red and blue for legacy requests.
    pil = Image.fromarray(img_array, mode="RGB")

    return pil.resize((image_size, image_size), Image.Resampling.BICUBIC)


def decode_jpeg_base64(encoded: str, image_size=448):
    try:
        payload = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
    except Exception as exc:
        raise ValueError("Invalid base64 JPEG observation") from exc


def decode_memory_images(data: dict, image_size: int):
    """Decode the compressed temporal schema or the legacy one-frame schema."""
    if "memory_images" in data:
        if data.get("image_encoding") != "jpeg_base64":
            raise ValueError(
                "memory_images currently requires image_encoding='jpeg_base64'"
            )
        encoded_grid = data["memory_images"]
        if not encoded_grid or not encoded_grid[0]:
            raise ValueError("memory_images must have shape [T][V]")
        num_views = len(encoded_grid[0])
        if any(len(frame) != num_views for frame in encoded_grid):
            raise ValueError("Every memory timestep must have the same view count")
        return [
            [decode_jpeg_base64(image, image_size) for image in frame]
            for frame in encoded_grid
        ]

    if "image" not in data:
        raise ValueError("Request must contain memory_images or legacy image")
    return [[decode_image_from_list(image, image_size) for image in data["image"]]]


def infer_from_json_dict(data: dict, model, normalizer, use_state: bool):
    device = "cuda"
    image_size = int(model.config.get("image_size", 448))

    images = decode_memory_images(data, image_size=image_size)
    num_frames = len(images)
    num_views = len(images[0])
    max_views = int(model.config.get("max_views", 3))
    if num_views < 1 or num_views > max_views:
        raise ValueError(
            f"Expected 1..{max_views} physical camera views, got {num_views}"
        )
    history_mask = torch.as_tensor(
        data.get("history_mask", [1] * num_frames),
        dtype=torch.bool,
        device=device,
    )
    if history_mask.shape != (num_frames,):
        raise ValueError(
            f"Expected history_mask shape {(num_frames,)}, got {tuple(history_mask.shape)}"
        )
    if not bool(history_mask[-1]):
        raise ValueError("Current memory frame must be valid")

    norm_state = None
    if use_state:
        state = torch.tensor(data["state"], dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (num_frames, 8):
            raise ValueError(
                "MuJoCo Panda state history must be [T,8] robot proprioception "
                "(eef xyz + axis-angle + two finger qpos); got "
                f"{tuple(state.shape)}"
            )
        norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.bool, device=device)
    if image_mask.ndim == 2:
        image_mask = image_mask[-1]
    if image_mask.shape != (num_views,):
        raise ValueError(
            f"Expected image_mask shape {(num_views,)}, got {tuple(image_mask.shape)}"
        )
    action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device=device)

    flow_seed = data.get("flow_seed", None, )
    if flow_seed is not None:
        flow_seed = int(flow_seed)
        
        if flow_seed < 0:
            raise ValueError(f"flow_seed must be >= 0, got {flow_seed}")
        torch.manual_seed(flow_seed)
        torch.cuda.manual_seed_all(flow_seed)
    
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
            history_mask=history_mask,
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()


async def handle_request(websocket, model, normalizer, use_state):
    print("Client connected", flush=True)
    try:
        async for message in websocket:
            json_data = json.loads(message)
            print(f"Received JSON observation")
            actions = infer_from_json_dict(json_data, model, normalizer, use_state)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")
            
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve an Evo-1 checkpoint over websocket.")
    parser.add_argument(
        "--ckpt-dir",
        default="/home/user/mujoco+evo/ckpt/evo1_mujoco_pickplace_stage1/step_best",
    )
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    ckpt_dir = args.ckpt_dir
    port = args.port
    
    print("Loading EVO_1 model...")
    model, normalizer, use_state = load_model_and_normalizer(ckpt_dir)
    
    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer, use_state),
            "0.0.0.0", port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
