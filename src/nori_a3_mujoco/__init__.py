"""MuJoCo simulation of the Nori A3, wired to the Nori Python SDK (nori-sdk).

    from nori_a3_mujoco import MujocoRobot, mujoco_session   # the sim (needs mujoco)
    from nori_a3_mujoco.policy import run_policy              # any session (nori-sdk only)

Imports are lazy so `nori_a3_mujoco.policy` works without MuJoCo installed.
"""

import os as _os

# Headless EGL by default; the interactive viewer (NORI_VIEWER=1) needs GLFW.
_os.environ.setdefault(
    "MUJOCO_GL", "glfw" if _os.environ.get("NORI_VIEWER", "") not in ("", "0") else "egl")

__version__ = "0.1.0"
__all__ = ["CAMERA_ROLES", "ROBOT_XML", "SCENE_XML", "MujocoRobot", "mujoco_session",
           "run_policy"]


def __getattr__(name):  # PEP 562 lazy exports
    if name in ("CAMERA_ROLES", "ROBOT_XML", "SCENE_XML", "MujocoRobot"):
        from . import robot
        return getattr(robot, name)
    if name == "mujoco_session":
        from .session import mujoco_session
        return mujoco_session
    if name == "run_policy":
        from .policy import run_policy
        return run_policy
    raise AttributeError(name)
