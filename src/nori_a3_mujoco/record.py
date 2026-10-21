"""Offscreen video recording of a MujocoRobot, keyed to SIM time (not wall time).

    rec = VideoRecorder(robot, fps=30)        # attaches to robot.step_callbacks
    ... run anything that steps the robot ...
    rec.save("demo.mp4")

Each frame is a wide shot from the scene's `demo_wide` camera plus the four robot camera
roles down the right edge, with an optional caption. Needs imageio + imageio-ffmpeg + Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .video import CameraRenderer


class VideoRecorder:
    def __init__(self, robot: Any, *, fps: float = 30.0, width: int = 960, height: int = 720,
                 camera: str = "demo_wide", inset: bool = True,
                 roles: tuple[str, ...] = ("front", "overhead", "left_wrist", "right_wrist"),
                 track_base: bool = True) -> None:
        self.robot = robot
        self.fps = fps
        self.width, self.height = width, height
        self.camera = camera
        self.inset = inset
        self.roles = roles
        self.track_base = track_base
        self.caption = ""
        self.frames: list[np.ndarray] = []
        self._next_t = robot.sim_time
        self._main = CameraRenderer(robot.model, width, height, shadows=True)
        th = height // len(roles)
        self._tile = CameraRenderer(robot.model, int(th * 4 / 3), th) if inset else None
        self._free = mujoco.MjvCamera()
        self._free.distance = 2.4
        self._free.azimuth = -125.0
        self._free.elevation = -18.0
        robot.step_callbacks.append(self._on_step)

    def _on_step(self, robot: Any) -> None:
        if robot.sim_time + 1e-9 >= self._next_t:
            self._next_t += 1.0 / self.fps
            self.frames.append(self.render())

    def render(self) -> np.ndarray:
        data = self.robot.data
        if self.track_base:
            x, y, _ = self.robot.base_pose()
            self._free.lookat[:] = [x + 0.25, y, 0.7]
            main = self._main.render(data, self._free)
        else:
            main = self._main.render(data, self.camera)
        if not self.inset:
            frame = main
        else:
            tiles = [self._tile.render(data, role) for role in self.roles]
            column = np.vstack(tiles)
            frame = np.hstack([main, column])
        return self._annotate(frame)

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return frame
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=22)
            small = ImageFont.load_default(size=15)
        except TypeError:  # old Pillow
            font = small = ImageFont.load_default()
        state = self.robot.measured_state()
        lines = [
            f"Nori A3 - MuJoCo sim   t={self.robot.sim_time:5.2f}s",
            self.caption,
            f"lift.pos {state.get('lift.pos', 0):6.1f} mm   "
            f"L grip {state.get('left_arm_gripper.pos', 0):5.1f}   "
            f"x.vel {state.get('x.vel', 0):+.2f}   safety {self._safety()}",
        ]
        y = 10
        for i, text in enumerate(lines):
            if not text:
                continue
            f = font if i < 2 else small
            draw.text((12, y), text, fill=(255, 255, 255), font=f, stroke_width=2,
                      stroke_fill=(0, 0, 0))
            y += 30 if i < 2 else 22
        if self.inset and self._tile is not None:
            th = self._tile.height
            for i, role in enumerate(self.roles):
                draw.text((self.width + 6, i * th + 4), role, fill=(255, 255, 0), font=small,
                          stroke_width=2, stroke_fill=(0, 0, 0))
        return np.asarray(img)

    def _safety(self) -> str:
        if self.robot.estopped:
            return "LATCHED"
        return {"stop": "safe_hold", "warn": "warn"}.get(self.robot.watchdog, "ok")

    def save(self, path: str | Path, stills: dict[str, int] | None = None) -> Path:
        import imageio.v2 as imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(path, self.frames, fps=self.fps, codec="libx264", quality=7,
                         macro_block_size=8)
        for name, index in (stills or {}).items():
            imageio.imwrite(path.parent / name, self.frames[min(index, len(self.frames) - 1)])
        return path

    def close(self) -> None:
        if self._on_step in self.robot.step_callbacks:
            self.robot.step_callbacks.remove(self._on_step)
        self._main.close()
        if self._tile is not None:
            self._tile.close()
