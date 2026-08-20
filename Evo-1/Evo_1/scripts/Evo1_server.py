# evo1_server_json.py

import argparse
import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1


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
    config["num_inference_timesteps"] = 50

    print("Building EVO_1 module...", flush=True)
    model = EVO1(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Applying checkpoint weights...", flush=True)
    model.load_state_dict(checkpoint["module"], strict=True)
    print("Moving model to CUDA...", flush=True)
    model = model.to("cuda")

    print("Loading normalizer...", flush=True)
    normalizer = Normalizer(stats)
    return model, normalizer, use_state


def decode_image_from_list(img_list, image_size=448):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")


def infer_from_json_dict(data: dict, model, normalizer, use_state: bool):
    device = "cuda"
    image_size = int(model.config.get("image_size", 448))

    images = [decode_image_from_list(img, image_size=image_size) for img in data["image"]]
    assert len(images) == 3, "Must provide exactly 3 images."
    for img in images:
        assert img.shape == (3, image_size, image_size), f"image_size must be (3,{image_size},{image_size})"

    norm_state = None
    if use_state:
        state = torch.tensor(data["state"], dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape != (1, 8):
            raise ValueError(
                "MuJoCo Panda state must be exactly 8-D robot proprioception "
                "(eef xyz + axis-angle + two finger qpos); got "
                f"{tuple(state.shape)}"
            )
        norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
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
