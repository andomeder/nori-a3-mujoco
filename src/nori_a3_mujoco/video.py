"""A stand-in for the robot's composite H.264 video track, rendered from MuJoCo cameras.

The real A3 sends ONE video track with every camera tiled into a grid, and the client crops
with `camera_layout.rect(role)`. This track renders the same grid (row-major, in the order
of the robot's announced tiles) so `frames()`, `snapshot(role=...)` and `snapshot_png()` on
RemoteTeleop work unchanged. No encoder, no network, no exposure: pixels are exact.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

try:  # match the real return type when PyAV is available
    import av  # type: ignore
except Exception:  # pragma: no cover - optional
    av = None


class SimFrame:
    """Minimal av.VideoFrame look-alike: `.width`, `.height`, `.to_ndarray(format="rgb24")`."""

    def __init__(self, rgb: np.ndarray, pts: int = 0) -> None:
        self._rgb = rgb
        self.height, self.width = rgb.shape[:2]
        self.pts = pts

    def to_ndarray(self, format: str = "rgb24") -> np.ndarray:  # noqa: A002 - av's name
        if format not in ("rgb24", "rgb"):
            raise ValueError(f"SimFrame only provides rgb24, not {format!r}")
        return self._rgb


class CameraRenderer:
    """Renders named MuJoCo cameras. Create and use it on ONE thread (GL context affinity)."""

    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240,
                 shadows: bool = False) -> None:
        self.model = model
        self.width, self.height = width, height
        self.shadows = shadows  # off by default: halves software-GL render time
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData, camera: str | int | mujoco.MjvCamera) -> np.ndarray:
        self._renderer.update_scene(data, camera=camera)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(self.shadows)
        return self._renderer.render().copy()

    def composite(self, data: mujoco.MjData, tiles: list[str], cols: int, rows: int) -> np.ndarray:
        out = np.zeros((rows * self.height, cols * self.width, 3), dtype=np.uint8)
        for i, role in enumerate(tiles):
            r, c = divmod(i, cols)
            try:
                img = self.render(data, role)
            except Exception:
                continue  # a role with no camera in the model stays black, like a dead tile
            out[r * self.height:(r + 1) * self.height, c * self.width:(c + 1) * self.width] = img
        return out

    def close(self) -> None:
        try:
            self._renderer.close()
        except Exception:
            pass


class SimVideoTrack:
    """`await track.recv()` -> the next composite frame, paced at `fps` wall-clock."""

    kind = "video"

    def __init__(self, robot: Any, fps: float = 15.0, tile_width: int = 320,
                 tile_height: int = 240) -> None:
        self.robot = robot
        self.fps = fps
        self._tw, self._th = tile_width, tile_height
        self._renderer: CameraRenderer | None = None
        self._next = 0.0
        self._pts = 0
        self.readyState = "live"

    async def recv(self) -> Any:
        if self.readyState != "live":
            raise RuntimeError("track ended")
        # ALWAYS suspend, even when late: a recv() that never awaits would starve the event
        # loop (and with it the physics pump and the control channel) whenever rendering is
        # slower than 1/fps, e.g. under software GL.
        await asyncio.sleep(max(0.0, self._next - time.monotonic()))
        self._next = max(self._next, time.monotonic()) + 1.0 / self.fps
        if self._renderer is None:
            self._renderer = CameraRenderer(self.robot.model, self._tw, self._th)
        layout = self.robot._layout()
        rgb = self._renderer.composite(self.robot.data, layout["tiles"], layout["cols"],
                                       layout["rows"])
        self._pts += 1
        if av is not None:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts = self._pts
            return frame
        return SimFrame(rgb, self._pts)

    def stop(self) -> None:
        self.readyState = "ended"
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
