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
                 track_base: bool = True, supersample: int = 2) -> None:
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
        self._main = CameraRenderer(robot.model, width, height, shadows=True,
                                    supersample=supersample)
        th = height // len(roles)
        self._tile = CameraRenderer(robot.model, int(th * 4 / 3), th,
                                    supersample=supersample) if inset else None
        self._free = mujoco.MjvCamera()
        self._free.distance = 1.9
        self._free.azimuth = -125.0
        self._free.elevation = -22.0
        # Optional close-up: set `close_on(lookat)` to blend (over `blend_s` of sim time) from
        # the base-tracking wide shot to a low shot from across the table, looking back at the
        # claw, so the grasp, the gap under the lifted cube and the placement are visible.
        self.close_view = {"distance": 0.65, "azimuth": 150.0, "elevation": -8.0}
        self.blend_s = 1.0
        self._close_lookat: np.ndarray | None = None
        self._w = 0.0
        self._target_w = 0.0
        robot.step_callbacks.append(self._on_step)

    def close_on(self, lookat: Any) -> None:
        """Blend to the close-up shot centred on world point `lookat`."""
        self._close_lookat = np.asarray(lookat, dtype=float)
        self._target_w = 1.0

    def wide(self) -> None:
        """Blend back to the wide, base-tracking shot."""
        self._target_w = 0.0

    def _on_step(self, robot: Any) -> None:
        if robot.sim_time + 1e-9 >= self._next_t:
            self._next_t += 1.0 / self.fps
            step = 1.0 / (self.fps * max(self.blend_s, 1e-6))
            self._w = float(np.clip(self._w + np.clip(self._target_w - self._w, -step, step),
                                    0.0, 1.0))
            self.frames.append(self.render())

    def render(self) -> np.ndarray:
        data = self.robot.data
        if self.track_base:
            x, y, _ = self.robot.base_pose()
            wide_look = np.array([x + 0.3, y + 0.08, 0.75])
            wide = {"distance": 1.9, "azimuth": -125.0, "elevation": -22.0}
            w = self._w if self._close_lookat is not None else 0.0
            w = w * w * (3 - 2 * w)  # smoothstep
            look = wide_look if w == 0 else (1 - w) * wide_look + w * self._close_lookat
            self._free.lookat[:] = look
            # Azimuth: -125 -> 150 is shortest via 180 (-125 == 235).
            az0, az1 = wide["azimuth"] % 360, self.close_view["azimuth"] % 360
            d_az = (az1 - az0 + 180) % 360 - 180
            self._free.azimuth = az0 + w * d_az
            for key in ("distance", "elevation"):
                setattr(self._free, key, (1 - w) * wide[key] + w * self.close_view[key])
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
        # High-quality H.264 (crf 14) in yuv420p so it plays everywhere.
        imageio.mimwrite(path, self.frames, fps=self.fps, codec="libx264", quality=None,
                         macro_block_size=8, pixelformat="yuv420p",
                         output_params=["-crf", "14", "-preset", "slow", "-tune", "animation"])
        for name, index in (stills or {}).items():
            out = path.parent / name
            out.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out, self.frames[min(index, len(self.frames) - 1)])
        return path

    def close(self) -> None:
        if self._on_step in self.robot.step_callbacks:
            self.robot.step_callbacks.remove(self._on_step)
        self._main.close()
        if self._tile is not None:
            self._tile.close()
