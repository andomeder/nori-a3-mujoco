"""Tiny colour-blob perception for the demo scene: find the red cube and the green place
marker in a robot camera image and turn the pixel into a 3-D point in base_footprint.

It uses only what a real robot would have: the camera image, the camera's intrinsics
(MJCF fovy), its extrinsics from the robot's own kinematics (the converted URDF, evaluated
at the measured lift height) and one scene prior, the table height. No sim ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

TABLE_TOP_Z = 0.76     # scene prior, metres above the floor (base_footprint z ~ floor)
CUBE_HALF = 0.02


def red_mask(img: np.ndarray) -> np.ndarray:
    i = img.astype(np.int16)
    return (i[..., 0] > 120) & (i[..., 0] > 2 * i[..., 1]) & (i[..., 0] > 2 * i[..., 2])


def green_mask(img: np.ndarray) -> np.ndarray:
    i = img.astype(np.int16)
    return (i[..., 1] > 90) & (i[..., 1] > 1.6 * i[..., 0]) & (i[..., 1] > 1.4 * i[..., 2])


@dataclass
class Blob:
    u: float          # centroid column (pixels)
    v: float          # centroid row (pixels, 0 = top)
    pixels: int
    v_min: float      # top row of the blob


def blob(mask: np.ndarray, min_pixels: int = 6) -> Blob | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    return Blob(float(xs.mean()), float(ys.mean()), int(len(xs)), float(ys.min()))


class CameraModel:
    """Camera extrinsics in base_footprint from the robot's kinematics (no world state)."""

    def __init__(self, model_xml: str | None = None) -> None:
        from .robot import ROBOT_XML

        self.model = mujoco.MjModel.from_xml_path(str(model_xml or ROBOT_XML))
        self.data = mujoco.MjData(self.model)
        self._lift = self.model.joint("lift_extension_joint").qposadr[0]
        self._middle = self.model.joint("lift_middle_joint").qposadr[0]  # mimic: 0.5 x lift
        self._base = self.model.body("base_footprint").id

    def pose(self, role: str, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray, float]:
        """(position, rotation matrix, fovy_deg) of camera `role` in base_footprint."""
        self.data.qpos[:] = self.model.qpos0
        lift = float(state.get("lift.pos", 0.0)) / 1000.0
        self.data.qpos[self._lift] = lift
        self.data.qpos[self._middle] = 0.5 * lift
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        cid = self.model.camera(role).id
        rb = self.data.xmat[self._base].reshape(3, 3)
        pb = self.data.xpos[self._base]
        pos = rb.T @ (self.data.cam_xpos[cid] - pb)
        rot = rb.T @ self.data.cam_xmat[cid].reshape(3, 3)
        return pos, rot, float(self.model.cam_fovy[cid])

    def ray(self, role: str, state: dict[str, float], u: float, v: float,
            width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        pos, rot, fovy = self.pose(role, state)
        f = (height / 2) / math.tan(math.radians(fovy) / 2)
        d_cam = np.array([(u + 0.5 - width / 2) / f, -(v + 0.5 - height / 2) / f, -1.0])
        d = rot @ d_cam
        return pos, d / np.linalg.norm(d)


def _hit_plane(origin: np.ndarray, d: np.ndarray, z: float) -> np.ndarray | None:
    if abs(d[2]) < 1e-6:
        return None
    t = (z - origin[2]) / d[2]
    return origin + t * d if t > 0 else None


class SceneLocator:
    """locate(image, state) -> cube centre / marker centre in base_footprint (or None)."""

    def __init__(self, role: str = "overhead", camera: CameraModel | None = None) -> None:
        self.role = role
        self.camera = camera or CameraModel()

    def cube(self, image: np.ndarray, state: dict[str, float]) -> np.ndarray | None:
        b = blob(red_mask(image))
        if b is None:
            return None
        h, w = image.shape[:2]
        origin, d = self.camera.ray(self.role, state, b.u, b.v, w, h)
        # From the overhead camera the blob is the top face plus the near side face; its
        # centroid projects close to the cube centre, so intersect with the plane at cube
        # mid-height. Checked against sim ground truth: <= 3 mm in x, <= 8 mm in y over the
        # reachable area (tests/test_grasp.py).
        p = _hit_plane(origin, d, TABLE_TOP_Z + CUBE_HALF)
        if p is None:
            return None
        p[2] = TABLE_TOP_Z + CUBE_HALF
        return p

    def marker(self, image: np.ndarray, state: dict[str, float]) -> np.ndarray | None:
        b = blob(green_mask(image))
        if b is None:
            return None
        h, w = image.shape[:2]
        origin, d = self.camera.ray(self.role, state, b.u, b.v, w, h)
        p = _hit_plane(origin, d, TABLE_TOP_Z)
        if p is not None:
            p[2] = TABLE_TOP_Z + CUBE_HALF
        return p


__all__ = ["CUBE_HALF", "TABLE_TOP_Z", "CameraModel", "SceneLocator", "blob", "green_mask",
           "red_mask"]
