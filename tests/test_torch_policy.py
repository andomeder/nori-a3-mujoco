import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nori_a3_mujoco import devices  # noqa: E402
from nori_a3_mujoco.policy import Observation  # noqa: E402
from nori_a3_mujoco.torch_policy import ChunkedTorchPolicy  # noqa: E402

KEYS = ["left_arm_elbow_pitch.pos", "left_arm_gripper.pos"]


class TinyACT(torch.nn.Module):
    """[B,S] + [B,N,3,H,W] -> [B,T,A]; row t of the chunk is state + t."""

    def __init__(self, chunk=4):
        super().__init__()
        self.chunk = chunk
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, state, images):
        steps = torch.arange(self.chunk, dtype=state.dtype, device=state.device)
        out = state.unsqueeze(1) + steps.view(1, -1, 1)
        if images is not None:
            out = out + 0.0 * images.mean()
        return out * self.scale


def obs(step, value=0.0):
    return Observation(step=step, t=step * 0.1, state={k: value for k in KEYS}, safety="ok",
                       watchdog="ok", images={"front": np.zeros((240, 320, 3), np.uint8)})


def test_device_selection_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("NORI_DEVICE", "xpu")
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        assert devices.torch_device().type == "cpu"
    monkeypatch.delenv("NORI_DEVICE")
    assert devices.torch_device().type in ("cpu", "cuda", "xpu")


@pytest.mark.asyncio
async def test_action_chunking():
    pol = ChunkedTorchPolicy(TinyACT(), KEYS, cameras=["front"], image_size=(64, 64),
                             query_every=4, device=torch.device("cpu"))
    outs = [await pol(obs(i)) for i in range(5)]
    assert [o["action"][KEYS[0]] for o in outs] == [0.0, 1.0, 2.0, 3.0, 0.0]
    assert len(pol.latencies) == 2  # queried at tick 0 and tick 4


@pytest.mark.asyncio
async def test_temporal_ensemble():
    pol = ChunkedTorchPolicy(TinyACT(), KEYS, temporal_ensemble=0.0, device=torch.device("cpu"))
    a0 = (await pol(obs(0)))["action"][KEYS[0]]
    a1 = (await pol(obs(1)))["action"][KEYS[0]]
    assert a0 == 0.0 and a1 == pytest.approx(0.5)  # mean of chunk0[1]=1 and chunk1[0]=0
