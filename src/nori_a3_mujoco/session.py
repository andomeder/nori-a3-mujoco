"""`mujoco_session()`: a connected RemoteTeleop backed by a MujocoRobot.

The same shape as `nori_sdk.mock.mock_session`, deliberately, so the ONE line a script swaps
is the context manager:

    async with mock_session(MockRobot()) as robot:     # SDK protocol double
    async with mujoco_session(MujocoRobot()) as robot: # this package: physics + cameras
    async with RemoteTeleop(SupabaseSignaling(...)) as robot:  # hardware

Differences from mock_session, all on the robot side of the wire:
  * physics is stepped at `sim_hz` (default 100 Hz) while telemetry still goes out at
    `telemetry_hz` (15 Hz, what the real gateway streams at);
  * a video track renders the four camera roles, so frames()/snapshot() work;
  * `realtime=True` (default) paces sim time to the wall clock. With realtime=False the sim
    still advances one `1/sim_hz` step per pump tick, but as fast as the loop allows.

Like mock_session, this assembles RemoteTeleop's post-negotiation state directly (there is
no WebRTC). It reaches the same private names mock_session does; pin nori-sdk accordingly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from nori_sdk.mock import loopback_pair
from nori_sdk.teleop import RemoteTeleop

from .robot import MujocoRobot
from .video import SimVideoTrack

TELEMETRY_HZ = 15.0
SIM_HZ = 100.0


class _SimChannel:
    """Stands in for the RTCDataChannel. Replies are scheduled, never delivered inline,
    exactly like nori_sdk.mock.session._MockChannel (a real channel is never reentrant)."""

    readyState = "open"

    def __init__(self, robot: MujocoRobot, teleop: RemoteTeleop) -> None:
        self._robot = robot
        self._teleop = teleop
        self.sent: list[dict[str, Any]] = []

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        loop = asyncio.get_running_loop()
        for reply in self._robot.handle(raw):
            loop.call_soon(self._teleop._handle_frame, reply)


@asynccontextmanager
async def mujoco_session(
    robot: MujocoRobot | None = None,
    *,
    telemetry_hz: float = TELEMETRY_HZ,
    sim_hz: float = SIM_HZ,
    realtime: bool = True,
    video: bool = True,
    video_fps: float = 15.0,
    strict: bool = False,
    viewer: bool | None = None,
) -> AsyncIterator[RemoteTeleop]:
    """`viewer=True` (or env NORI_VIEWER=1) opens MuJoCo's interactive passive viewer
    (GLFW window; needs a display) and syncs it from the pump at ~60 Hz wall time."""
    bot = MujocoRobot() if robot is None else robot
    if viewer is None:
        viewer = os.environ.get("NORI_VIEWER", "") not in ("", "0")
    handle = None
    if viewer:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(bot.model, bot.data)
    operator, _robot_side = loopback_pair()
    teleop = RemoteTeleop(operator, strict=strict)
    teleop._loop = asyncio.get_running_loop()
    teleop._control = _SimChannel(bot, teleop)
    teleop._set_phase("connected")
    teleop._connected.set()

    for frame in bot.on_channel_open():
        teleop._handle_frame(frame)

    track = SimVideoTrack(bot, fps=video_fps) if video and bot.cameras else None
    if track is not None:
        teleop._video_track = track

    async def pump() -> None:
        dt = 1.0 / sim_hz
        tel_every = max(1, int(round(sim_hz / telemetry_hz))) if telemetry_hz > 0 else 0
        tick = 0
        start = time.monotonic()
        last_sync = [0.0]
        while True:
            if realtime:
                lag = (start + tick * dt) - time.monotonic()
                await asyncio.sleep(max(0.0, lag))
            else:
                await asyncio.sleep(0)
            bot.step(dt)
            tick += 1
            if handle is not None and handle.is_running():
                now = time.monotonic()
                if now - last_sync[0] >= 1 / 60:
                    handle.sync()
                    last_sync[0] = now
            for event in bot.drain_events():
                teleop._handle_frame(event)
            if tel_every and tick % tel_every == 0:
                teleop._handle_frame(bot.telemetry())
                layout = bot.repeat_layout()
                if layout is not None:
                    teleop._handle_frame(layout)

    task = asyncio.create_task(pump())
    try:
        yield teleop
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if track is not None:
            track.stop()
        if handle is not None:
            handle.close()
        await teleop.stop()


__all__ = ["SIM_HZ", "TELEMETRY_HZ", "mujoco_session"]
