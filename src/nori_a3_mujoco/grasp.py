"""Grasp-point calibration for the A3 claw.

The URDF's `gripper_tcp_link` sits at the closed fingertips. An object is actually held in
the jaw cavity, 24 mm further back along the approach axis (and 14 mm off-centre across the
finger plate): that is the `{side}_grasp` site added by scripts/build_model.py, measured
from the visual meshes. `pose()` keeps the SDK meaning (it targets the URDF TCP), so to put
the *grasp point* somewhere, convert first:

    pos_tcp = tcp_for_grasp("left", grasp_pos, quat_xyzw)
    await robot.pose("left", pos_tcp, quat_xyzw)

The offset is a constant of the model, so this works unchanged against the mock (no model
needed: the numbers are cached below) and, to the extent the URDF is right, the real robot.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Grasp site expressed in the TCP frame (metres). Verified against the model by the tests.
GRASP_IN_TCP = {
    "left": np.array([0.0001, -0.0139, -0.024284]),
    "right": np.array([0.0001, 0.0139, -0.024284]),
}

# TCP orientations (x, y, z, w in base_footprint) for a horizontal side grasp: approach
# along base +x, fingers closing across base y, the claw's heavy hub/camera side up.
SIDE_GRASP_XYZW = {
    "left": (0.5, -0.5, 0.5, -0.5),
    "right": (0.5, 0.5, 0.5, 0.5),
}


def _mat(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat_xyzw
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tcp_for_grasp(side: str, grasp_pos: Sequence[float],
                  quat_xyzw: Sequence[float]) -> list[float]:
    """TCP position that puts the claw's grasp point at `grasp_pos` (same frame)."""
    r = _mat(quat_xyzw)
    return [float(v) for v in np.asarray(grasp_pos, float) - r @ GRASP_IN_TCP[side]]


def grasp_in_tcp_from_model(model, data, side: str) -> np.ndarray:
    """Recompute GRASP_IN_TCP from a MuJoCo model (used by the tests)."""
    import mujoco

    mujoco.mj_kinematics(model, data)
    t = model.site(f"{side}_tcp").id
    g = model.site(f"{side}_grasp").id
    rt = data.site_xmat[t].reshape(3, 3)
    return rt.T @ (data.site_xpos[g] - data.site_xpos[t])


__all__ = ["GRASP_IN_TCP", "SIDE_GRASP_XYZW", "grasp_in_tcp_from_model", "tcp_for_grasp"]
