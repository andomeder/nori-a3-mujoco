"""ACT-shaped torch policy through run_policy, on whatever device devices.torch_device() picks.

The network is UNTRAINED: a small CNN + MLP that outputs a 20-step action chunk as a bounded
residual (+-2 units) around the current joint state, so the robot only twitches in place.
The point is the plumbing and the timing on your device: camera frames -> tensor on
cuda/xpu/cpu -> chunk -> actions over the wire, with inference off the event loop.

    python examples/torch_policy.py                    # sim, auto device
    NORI_DEVICE=xpu python examples/torch_policy.py    # force Intel Arc (falls back to CPU)
    python examples/torch_policy.py --bench 200        # inference only, no sim
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import numpy as np
import torch

from nori_a3_mujoco import devices
from nori_a3_mujoco.policy import Observation, run_policy
from nori_a3_mujoco.torch_policy import ChunkedTorchPolicy

SIDES = ("left", "right")
DOFS = ("shoulder_pitch", "shoulder_roll", "bicep_yaw", "elbow_pitch",
        "forearm_yaw", "wrist_pitch", "wrist_roll", "gripper")
JOINTS = [f"{s}_arm_{d}.pos" for s in SIDES for d in DOFS]
CAMERAS = ["front", "left_wrist", "right_wrist"]
CHUNK = 20


class TinyACT(torch.nn.Module):
    """state [B,S] + images [B,N,3,H,W] -> chunk [B,T,A] (absolute targets)."""

    def __init__(self, n_state: int, n_cams: int, chunk: int = CHUNK, width: int = 256):
        super().__init__()
        self.chunk, self.n_act = chunk, n_state
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 * n_cams + n_state, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, chunk * n_state))

    def forward(self, state, images):
        b, n = images.shape[:2]
        feats = self.cnn(images.flatten(0, 1)).view(b, -1)
        delta = torch.tanh(self.head(torch.cat([feats, state], 1))) * 2.0
        return state.unsqueeze(1) + delta.view(b, self.chunk, self.n_act)


def make_policy(device=None) -> ChunkedTorchPolicy:
    torch.manual_seed(0)
    return ChunkedTorchPolicy(TinyACT(len(JOINTS), len(CAMERAS)), JOINTS, cameras=CAMERAS,
                              image_size=(224, 224), query_every=10, device=device)


def bench(n: int) -> None:
    pol = make_policy()
    obs = Observation(step=0, t=0.0, state={k: 0.0 for k in JOINTS}, safety="ok",
                      watchdog="ok", telemetry=None,
                      images={c: np.zeros((240, 320, 3), np.uint8) for c in CAMERAS})
    for _ in range(10):
        pol.infer(obs)  # warm-up (XPU/CUDA kernel compilation happens here)
    pol.latencies.clear()
    for _ in range(n):
        pol.infer(obs)
    lat = sorted(pol.latencies)
    print(f"device={pol.device}  forward+copy: median {1e3 * statistics.median(lat):.2f} ms, "
          f"p95 {1e3 * lat[int(0.95 * (len(lat) - 1))]:.2f} ms over {n} calls")


async def run(seconds: float, hz: float) -> None:
    from nori_a3_mujoco import MujocoRobot, mujoco_session

    pol = make_policy()
    print(f"policy device: {pol.device}")
    bot = MujocoRobot()
    async with mujoco_session(bot, realtime=True) as robot:
        report = await run_policy(robot, pol, hz=hz, cameras=CAMERAS, duration=seconds,
                                  clock=lambda: bot.sim_time)
    lat = pol.latencies
    print(report)
    if lat:
        print(f"{len(lat)} inferences, median {1e3 * statistics.median(lat):.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", type=int, default=0, help="inference-only benchmark, N calls")
    ap.add_argument("--seconds", type=float, default=5.0, help="sim seconds to run")
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()
    print(devices.describe())
    t0 = time.perf_counter()
    if args.bench:
        bench(args.bench)
    else:
        asyncio.run(run(args.seconds, args.hz))
    print(f"wall {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
