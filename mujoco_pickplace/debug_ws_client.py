# debug_ws_client.py — minimal websocket client to locate the eval hang.
import asyncio
import json
import time
import numpy as np
import websockets

from pick_place_env import PickPlaceEnv


async def main():
    t0 = time.time()
    print("[1] creating env...", flush=True)
    env = PickPlaceEnv()
    print(f"[1] env created in {time.time()-t0:.1f}s", flush=True)

    print("[2] reset...", flush=True)
    obs = env.reset(seed=1000)
    print(f"[2] reset done in {time.time()-t0:.1f}s", flush=True)

    print("[3] building payload...", flush=True)
    images = [obs["image_front"], obs["image_overhead"], obs["image_wrist"]]
    payload = {
        "image": [img[..., ::-1].astype(np.uint8).tolist() for img in images],
        "image_mask": [1, 1, 1],
        "state": obs["state"].astype(float).tolist(),
        "action_mask": [1, 1, 1, 1] + [0] * 20,
        "prompt": "pick up the blue cube and place it on the green target",
    }
    print(f"[3] payload built in {time.time()-t0:.1f}s", flush=True)

    print("[4] connecting...", flush=True)
    async with websockets.connect("ws://127.0.0.1:9000", max_size=100_000_000) as ws:
        print(f"[4] connected in {time.time()-t0:.1f}s", flush=True)
        print("[5] sending obs...", flush=True)
        msg = json.dumps(payload)
        print(f"[5] json.dumps done in {time.time()-t0:.1f}s (len={len(msg)})", flush=True)
        await ws.send(msg)
        print(f"[5] sent in {time.time()-t0:.1f}s", flush=True)
        print("[6] waiting for action...", flush=True)
        result = await ws.recv()
        print(f"[6] received action in {time.time()-t0:.1f}s, shape={np.asarray(json.loads(result)).shape}", flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
