"""Run a torch policy (e.g. an ACT-style action-chunking model) inside run_policy().

    from nori_a3_mujoco.torch_policy import ChunkedTorchPolicy
    policy = ChunkedTorchPolicy(model, joint_keys=info.descriptor.joints, cameras=["front"])
    await run_policy(robot, policy, hz=30, cameras=["front"])

Model contract (the shape ACT/LeRobot-style models use; adapt with a small nn.Module wrapper
if yours differs):

    model(state: FloatTensor[B, S], images: FloatTensor[B, N, 3, H, W]) -> FloatTensor[B, T, A]

  * state  = telemetry values for `state_keys` (default: joint_keys), in that order
  * images = the `cameras` roles in that order, RGB in [0, 1], resized to `image_size`
  * output = an ACTION CHUNK of T future targets for `joint_keys` (A == len(joint_keys)),
             in the robot's wire units unless you pass `action_mean`/`action_std`
             (or `state_mean`/`state_std` for the input), which are applied for you.

Every `query_every` ticks the model is queried; between queries the next row of the last
chunk is sent (plain action chunking). `temporal_ensemble=m` instead averages all chunks that
cover the current tick with exp(-m * age) weights, as in the ACT paper.

Inference runs in a worker thread (asyncio.to_thread) so the event loop -- and with it the
watchdog keep-alive and, in sim, the physics pump -- keeps running during a forward pass.
If inference outlives run_policy's `stale_after`, the keep-alive goes silent ON PURPOSE.

Device: devices.torch_device() -> CUDA, Intel XPU (Arc), or CPU; override with NORI_DEVICE.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import devices


class ChunkedTorchPolicy:
    def __init__(
        self,
        model: Any,
        joint_keys: Sequence[str],
        *,
        state_keys: Sequence[str] | None = None,
        cameras: Sequence[str] = (),
        image_size: tuple[int, int] = (224, 224),
        query_every: int = 1,
        temporal_ensemble: float | None = None,
        device: Any = None,
        state_mean: Any = None, state_std: Any = None,
        action_mean: Any = None, action_std: Any = None,
        dtype: Any = None,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device if device is not None else devices.torch_device()
        self.dtype = dtype or torch.float32
        self.model = devices.maybe_ipex_optimize(model.to(self.device).eval(), self.device)
        self.joint_keys = list(joint_keys)
        self.state_keys = list(state_keys or joint_keys)
        self.cameras = list(cameras)
        self.image_size = image_size
        self.query_every = max(1, int(query_every))
        self.temporal_ensemble = temporal_ensemble

        def t(x):
            return None if x is None else torch.as_tensor(np.asarray(x), dtype=self.dtype,
                                                          device=self.device)
        self.state_mean, self.state_std = t(state_mean), t(state_std)
        self.action_mean, self.action_std = t(action_mean), t(action_std)
        self._chunk: np.ndarray | None = None
        self._chunk_step = 0
        self._history: deque[tuple[int, np.ndarray]] = deque()
        self._tick = 0
        self.latencies: list[float] = []

    # --- tensor plumbing ---------------------------------------------------------------------

    def _state(self, obs: Any):
        torch = self.torch
        values = [float(obs.state.get(k, 0.0)) for k in self.state_keys]
        x = torch.tensor([values], dtype=self.dtype, device=self.device)
        if self.state_mean is not None:
            x = (x - self.state_mean) / self.state_std
        return x

    def _images(self, obs: Any):
        torch = self.torch
        if not self.cameras:
            return None
        h, w = self.image_size
        frames = []
        for role in self.cameras:
            img = obs.images.get(role)
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)  # missing camera -> black tile
            frames.append(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1))
        x = torch.stack(frames).to(self.device, self.dtype) / 255.0      # [N, 3, H0, W0]
        if x.shape[-2:] != (h, w):
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear",
                                                align_corners=False)
        return x.unsqueeze(0)                                            # [1, N, 3, H, W]

    def infer(self, obs: Any) -> np.ndarray:
        """One forward pass -> action chunk [T, A] in wire units (blocking)."""
        torch = self.torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            state = self._state(obs)
            images = self._images(obs)
            out = self.model(state, images) if images is not None else self.model(state, None)
            if out.dim() == 2:
                out = out.unsqueeze(1)                                   # [B, A] -> [B, 1, A]
            if self.action_mean is not None:
                out = out * self.action_std + self.action_mean
            chunk = out[0].float().cpu().numpy()                         # implies a sync
        self.latencies.append(time.perf_counter() - t0)
        if chunk.shape[-1] != len(self.joint_keys):
            raise ValueError(f"model returned {chunk.shape[-1]} action dims, "
                             f"expected {len(self.joint_keys)} (joint_keys)")
        return chunk

    # --- the run_policy callable -------------------------------------------------------------

    async def __call__(self, obs: Any):
        tick = self._tick
        self._tick += 1
        need = (self._chunk is None or tick % self.query_every == 0
                or self._chunk_step >= len(self._chunk) or self.temporal_ensemble is not None)
        if need:
            chunk = await asyncio.to_thread(self.infer, obs)
            self._chunk, self._chunk_step = chunk, 0
            if self.temporal_ensemble is not None:
                self._history.append((tick, chunk))
        if self.temporal_ensemble is not None:
            rows, weights = [], []
            for start, chunk in list(self._history):
                age = tick - start
                if age >= len(chunk):
                    continue
                rows.append(chunk[age])
                weights.append(np.exp(-self.temporal_ensemble * age))
            while self._history and tick - self._history[0][0] >= len(self._history[0][1]):
                self._history.popleft()
            w = np.asarray(weights)
            target = (np.asarray(rows) * w[:, None]).sum(0) / w.sum()
        else:
            target = self._chunk[min(self._chunk_step, len(self._chunk) - 1)]
            self._chunk_step += 1
        return {"action": {k: float(v) for k, v in zip(self.joint_keys, target)}}
