"""Explicit, frozen observation readout for composite MuJoCo policies.

The backbone and Task1 flow expert remain unchanged. This backend is loaded
only by an explicit checkpoint manifest; it never reads simulator annotations.
"""
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _child(root, name):
    root = Path(root).resolve()
    relative = Path(name)
    path = (root / relative).resolve()
    _require(not relative.is_absolute() and path.is_relative_to(root),
             "Observation decoder artifact must stay inside its checkpoint")
    _require(path.is_file(), f"Missing observation decoder artifact: {path}")
    return path


def _verified_bytes(root, declaration):
    _require(isinstance(declaration, dict), "Invalid decoder artifact declaration")
    path = _child(root, declaration.get("file", ""))
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == declaration.get("sha256"),
             f"Observation decoder artifact changed: {path.name}")
    return payload


def backbone_environment_contracts(config):
    """Do not relabel copied backbone weights with the adapted head's runtime."""
    if config.get("observation_action_decoder") is None:
        return config.get("task_environment_contracts")
    source = config.get("initialization_source_task_environment_contracts")
    _require(isinstance(source, dict) and bool(source),
             "Composite policy is missing its original backbone environment provenance")
    _require(config.get("inference_only_policy_bundle") is True,
             "Observation decoder bundle must explicitly declare inference-only use")
    return source


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.linear = nn.Linear(128, 128)

    def forward(self, value):
        return value + F.gelu(self.linear(self.norm(value)))


class _SharedReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(53))
        self.register_buffer("scale", torch.ones(53))
        self.input = nn.Linear(53, 128)
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.norm = nn.LayerNorm(128)
        self.numeric = nn.Linear(128, 84)
        self.gripper = nn.Linear(128, 14)

    def forward(self, value):
        hidden = F.gelu(self.input((value - self.center) / self.scale))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return self.numeric(hidden).reshape(-1, 14, 6), self.gripper(hidden)


def _tensor(value, shape, dtype, name):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
             and value.dtype == dtype and bool(torch.isfinite(value).all()),
             f"Invalid observation decoder tensor: {name}")
    return value.detach().cpu()


class ObservationActionDecoder:
    """Serving component, deliberately separate from the backbone state_dict.

    All numerical readout parameters are frozen FP32. The CPU projection is
    FP32, and the single-frame visual classification uses the original FP64
    readout. Its cache contains scores only for the current valid window.
    """
    def __init__(self, manifest, decoder, readout, projection, stats, device):
        self.manifest = manifest
        self.stats = stats
        self.tasks = manifest["tasks"]
        self.by_prompt = {row["instruction"]: key for key, row in self.tasks.items()}
        _require(len(self.by_prompt) == 3, "Composite tasks must have distinct public instructions")
        _require(set(self.tasks) == {"mujoco_pickplace", "mujoco_chopstick", "mujoco_drawer_memory"},
                 "Unsupported composite task registry")
        for key, number in (("mujoco_pickplace", 1), ("mujoco_chopstick", 3), ("mujoco_drawer_memory", 4)):
            row = self.tasks[key]
            expected = [1, 1, 1, int(number == 4), int(number == 4), int(number == 4), 1] + [0] * 17
            _require(row.get("number") == number and row.get("action_mask") == expected,
                     "Invalid public composite task/action contract")
            _require(row.get("backend") == ("original_flow" if number == 1 else "shared_observation_readout"),
                     "Task1 must keep the original flow action path")
            _require(key in stats, f"Missing source normalization for {key}")
        _require(decoder.get("input_width") == 53 and decoder.get("hidden_width") == 128
                 and decoder.get("horizon") == 14 and decoder.get("numeric_dim") == 6,
                 "Unsupported observation readout architecture")
        state = decoder.get("module")
        _require(isinstance(state, dict), "Missing observation readout parameters")
        with torch.random.fork_rng(devices=[]):
            self.decoder = _SharedReadout()
        expected = self.decoder.state_dict()
        _require(state.keys() == expected.keys(), "Observation readout parameter names differ")
        for name, value in state.items():
            _tensor(value, expected[name].shape, torch.float32, name)
        _require(bool((state["scale"] > 0).all()), "Invalid observation readout scale")
        self.decoder.load_state_dict(state, strict=True)
        self.decoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        center = readout.get("center")
        _require(isinstance(center, torch.Tensor) and center.ndim == 1, "Invalid visual readout center")
        self.readout = {
            "center": _tensor(center, center.shape, torch.float64, "readout.center"),
            "weight": _tensor(readout.get("weight"), (len(center), 5), torch.float64, "readout.weight"),
            "bias": _tensor(readout.get("bias"), (5,), torch.float64, "readout.bias"),
        }
        center = projection.get("center")
        _require(isinstance(center, torch.Tensor) and center.ndim == 1, "Invalid visual projection center")
        self.projection = {
            "center": _tensor(center, center.shape, torch.float32, "projection.center"),
            "scale": _tensor(projection.get("scale"), center.shape, torch.float32, "projection.scale"),
            "projection": _tensor(projection.get("projection"), (len(center), 32), torch.float32, "projection.matrix"),
        }
        _require(bool((self.projection["scale"] > 0).all()), "Invalid visual projection scale")
        self._cache = {}
        self.last_diagnostics = None

    def validate_request(self, task_key, prompt, action_mask):
        _require(task_key in self.tasks and self.tasks[task_key]["instruction"] == prompt,
                 "Composite request task_key and public instruction disagree")
        value = torch.as_tensor(action_mask).detach().cpu()
        if value.shape == (1, 24):
            value = value[0]
        _require(value.shape == (24,) and torch.equal(value, torch.tensor(self.tasks[task_key]["action_mask"], dtype=value.dtype)),
                 "Composite request action mask differs from its public task contract")
        return task_key

    @torch.inference_mode()
    def predict(self, model, *, images, image_mask, prompt, state, action_mask,
                history_mask, fused_tokens, fused_mask):
        _require(prompt in self.by_prompt, "Unknown instruction for this composite policy")
        key = self.validate_request(self.by_prompt[prompt], prompt, action_mask)
        if self.tasks[key]["number"] == 1:
            self._cache.clear()
            self.last_diagnostics = {"backend": "original_flow", "task_key": key}
            return None
        _require(state is not None and tuple(state.shape) == (1, 6, 24) and state.dtype == torch.float32,
                 "Observation readout requires the original FP32 normalized K6 state")
        _require(fused_tokens.shape[0] == 1 and fused_mask is not None,
                 "Observation readout requires one masked observation window")
        history = torch.as_tensor(history_mask).detach().cpu().bool().reshape(-1)
        _require(history.shape == (6,) and bool(history[-1]), "Invalid composite history mask")
        _require(len(images) == 6 and all(len(frame) == 1 for frame in images)
                 and torch.as_tensor(image_mask).reshape(-1).tolist() == [True],
                 "Observation readout expects its single real front-camera view")
        valid = fused_mask.detach().cpu().bool().unsqueeze(-1)
        _require(bool(valid.any()), "No valid fused observation tokens")
        tokens = fused_tokens.detach().float().cpu()
        pooled = tokens.masked_fill(~valid, 0.).sum(1)[0] / valid.float().sum(1).clamp_min(1.)[0]
        p = self.projection
        visual = ((pooled - p["center"]) / p["scale"]) @ p["projection"]
        category = torch.zeros(4)
        scores = None
        if self.tasks[key]["number"] == 4:
            frames = []
            for frame, okay in zip(images, history.tolist()):
                if okay:
                    rgb = frame[0].convert("RGB")
                    _require(rgb.size == (448, 448), "Composite visual input must stay at 448px")
                    digest = hashlib.sha256(prompt.encode() + rgb.tobytes()).hexdigest()
                    frames.append((digest, rgb))
            current = {digest for digest, _ in frames}
            self._cache = {digest: score for digest, score in self._cache.items() if digest in current}
            for digest, rgb in frames:
                if digest in self._cache:
                    continue
                fused, mask = model.get_vl_embeddings(
                    images=[[rgb]], image_mask=torch.ones(1, dtype=torch.bool, device=fused_tokens.device),
                    prompt=prompt, history_mask=torch.ones(1, dtype=torch.bool, device=fused_tokens.device),
                    return_attention_mask=True)
                vector = fused.detach().float().cpu().masked_fill(~mask.detach().cpu().bool().unsqueeze(-1), 0.).flatten().double()
                _require(vector.shape == self.readout["center"].shape, "Visual readout token shape changed")
                self._cache[digest] = ((vector - self.readout["center"]) @ self.readout["weight"] + self.readout["bias"])
            scores = torch.stack([self._cache[digest][:4] - self._cache[digest][4] for digest, _ in frames]).max(0).values
            if float(scores.max()) > 0.:
                category[int(scores.argmax())] = 1.
        else:
            self._cache.clear()
        normalized = state.detach().float().cpu()[0]
        current = normalized[-1, :8]
        recent = current - normalized[-2, :8] if bool(history[-2]) else torch.zeros(8)
        public_rotation = float(torch.as_tensor(action_mask)[0, 3:6].bool().any())
        condition = torch.cat([category, visual, current, recent, torch.tensor([public_rotation])])
        _require(bool(torch.isfinite(condition).all()), "Nonfinite observation readout condition")
        with torch.autocast(device_type=fused_tokens.device.type, enabled=False):
            numeric, logits = self.decoder(condition.unsqueeze(0).to(fused_tokens.device).float())
            grip = logits.sigmoid()
        action = torch.zeros((1, 14, 24), dtype=torch.float32, device=fused_tokens.device)
        action[:, :, :6] = numeric
        low = self.stats[key]["action"]["min"][6]
        high = self.stats[key]["action"]["max"][6]
        action[:, :, 6] = 2 * (grip - low) / (high - low + 1e-8) - 1
        public = model.action_head._expand_action_mask(action_mask, batch_size=1, per_action_dim=24,
            device=action.device, dtype=torch.float32)
        _require(bool(torch.isfinite(action).all()), "Nonfinite observation readout action")
        self.last_diagnostics = dict(backend="shared_observation_readout", task_key=key,
            memory_predicted_category=(int(category.argmax()) if bool(category.any()) else None),
            memory_scores=(scores.tolist() if scores is not None else None),
            memory_category_is_learned_not_true_target=True, memory_cache_only_current_K6=True,
            normalized_current_state=current.tolist(), predicted_gripper=float(grip[0, 0]),
            decoder_visible_condition=condition.tolist(), decoder_state_precision="raw normalized float32",
            decoder_has_no_flow_or_baseline_action_input=True)
        return (action * public).reshape(1, -1)


def attach_observation_decoder(model, checkpoint_dir, config, stats, identity):
    declaration = config.get("observation_action_decoder")
    if declaration is None:
        return None
    _require(isinstance(declaration, dict) and declaration.get("version") == 1,
             "Unsupported observation decoder declaration")
    root = Path(checkpoint_dir).resolve()
    manifest_bytes = _verified_bytes(root, declaration)
    manifest = json.loads(manifest_bytes)
    _require(manifest.get("schema") == "evo-observation-action-decoder" and manifest.get("version") == 1,
             "Unsupported observation decoder manifest")
    _require(config.get("inference_only_policy_bundle") is True and config.get("use_lora") is False,
             "Composite observation readout requires its frozen, non-LoRA backbone")
    for field, expected in (("image_size", 448), ("memory_frames", 6), ("memory_stride_seconds", 1.0),
                            ("pi_mem_attention_mode", "composed")):
        _require(config.get(field) == expected and manifest.get(field) == expected,
                 f"Observation readout feature contract differs: {field}")
    _require(not config.get("return_cls_only", False) and not model.legacy_inference_contract,
             "Observation readout requires masked full-token composed inference")
    _require(config.get("action_horizon", config.get("horizon", 14)) == 14
             and config.get("state_dim", 24) == 24 and config.get("per_action_dim", 24) == 24,
             "Observation readout requires its 14 by 24 action and 24-dimensional state contract")
    source = manifest["source_backbone_identity"]
    _require(identity["weights_sha256"] == source["weights_sha256"]
             and identity["norm_stats_sha256"] == source["norm_stats_sha256"],
             "Composite source weights or normalization changed")
    source_config_bytes = _verified_bytes(root, manifest["source_config"])
    _require(hashlib.sha256(source_config_bytes).hexdigest() == source["config_sha256"],
             "Composite original configuration identity changed")
    source_config = json.loads(source_config_bytes)
    _require(source_config.get("observation_action_decoder") is None,
             "Nested composite backbones are unsupported")
    _require(backbone_environment_contracts(config) == source_config.get("task_environment_contracts")
             and config.get("task_environment_contracts") == manifest.get("runtime_task_environment_contracts"),
             "Composite source and runtime environments must be declared separately")
    # Freeze all fields that affect the visual feature function, including
    # temporal-layer precision and masks. Runtime provenance may differ only
    # because the independently learned readout was adapted in that runtime.
    runtime_fields = {"observation_action_decoder", "inference_only_policy_bundle",
        "initialization_source_task_environment_contracts", "task_environment_contracts",
        "device", "vlm_name", "legacy_inference_contract", "num_inference_timesteps"}
    for field in (config.keys() | source_config.keys()) - runtime_fields:
        _require(config.get(field) == source_config.get(field), f"Composite backbone field changed: {field}")
    _require(config.get("legacy_inference_contract") is False
             and config.get("num_inference_timesteps") == 50,
             "Composite Task1 flow must retain the same nonlegacy 50-step solver in every loader")
    artifacts = {}
    for name in ("decoder", "readout", "projection"):
        payload = _verified_bytes(root, manifest["artifacts"][name])
        artifacts[name] = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    component = ObservationActionDecoder(manifest, artifacts["decoder"], artifacts["readout"],
        artifacts["projection"], stats, next(model.action_head.parameters()).device)
    model.observation_action_decoder = component
    details = dict(backend="shared_observation_readout_v1", manifest_sha256=declaration["sha256"],
        artifacts_sha256={key: value["sha256"] for key, value in manifest["artifacts"].items()},
        backbone_step_role="frozen_initialization_source", training=manifest.get("decoder_training"),
        validation=manifest.get("validation"), inference_only=True)
    identity["observation_action_decoder"] = details
    return details
