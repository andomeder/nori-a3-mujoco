"""Damped-least-squares IK for one A3 arm's gripper TCP, on a scratch copy of the sim.

Used to serve `pose()` (capability pose_targets) in the sim. It is NOT the robot's on-board
solver: same inputs (TCP position in base_footprint, optional xyzw orientation), but its own
numerics, so solutions, failure cases and reason strings only approximate the hardware's.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_DOFS = ("shoulder_pitch", "shoulder_roll", "bicep_yaw", "elbow_pitch",
            "forearm_yaw", "wrist_pitch", "wrist_roll")


@dataclass
class IkResult:
    ok: bool
    q: dict[str, float]          # joint name -> rad
    pos_err_m: float
    rot_err_rad: float
    iterations: int


def solve(model: mujoco.MjModel, data: mujoco.MjData, side: str, position_m,
          orientation_xyzw=None, *, restarts: int = 12, **kw) -> IkResult:
    """Solve from the current configuration; if that fails, retry from the zero pose and
    then from a fixed set of pseudo-random seeds (deterministic: same input, same answer)."""
    best = _solve(model, data, side, position_m, orientation_xyzw, seed=None, **kw)
    if best.ok:
        return best
    rng = np.random.default_rng(7)
    seeds = [0.0] + [rng.uniform(-1.2, 1.2, size=len(ARM_DOFS)) for _ in range(restarts)]
    for seed in seeds:
        res = _solve(model, data, side, position_m, orientation_xyzw, seed=seed, **kw)
        if res.ok:
            return res
        if res.pos_err_m + res.rot_err_rad < best.pos_err_m + best.rot_err_rad:
            best = res
    return best


def _solve(model: mujoco.MjModel, data: mujoco.MjData, side: str, position_m, orientation_xyzw=None,
          *, seed: float | None = None, iters: int = 200, damping: float = 0.05, pos_tol: float = 0.005,
          rot_tol: float = 0.05) -> IkResult:
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    joints = [model.joint(f"{side}_{d}_joint") for d in ARM_DOFS]
    qadr = np.array([j.qposadr[0] for j in joints])
    dadr = np.array([j.dofadr[0] for j in joints])
    lo = np.array([j.range[0] for j in joints])
    hi = np.array([j.range[1] for j in joints])
    site = model.site(f"{side}_tcp").id
    base = model.body("base_footprint").id

    if seed is not None:
        scratch.qpos[qadr] = np.clip(seed, lo, hi)
    mujoco.mj_kinematics(model, scratch)
    bpos = scratch.xpos[base].copy()
    bmat = scratch.xmat[base].reshape(3, 3).copy()
    target = bpos + bmat @ np.asarray(position_m, dtype=float)
    target_mat = None
    if orientation_xyzw is not None:
        x, y, z, w = orientation_xyzw
        q = np.array([w, x, y, z], dtype=float)
        local = np.zeros(9)
        mujoco.mju_quat2Mat(local, q / np.linalg.norm(q))
        target_mat = bmat @ local.reshape(3, 3)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = rot_err = np.inf
    for it in range(1, iters + 1):
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        e_pos = target - scratch.site_xpos[site]
        pos_err = float(np.linalg.norm(e_pos))
        if target_mat is not None:
            cur = scratch.site_xmat[site].reshape(3, 3)
            r = target_mat @ cur.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r.flatten())
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            rot_err = float(np.linalg.norm(e_rot))
        else:
            e_rot = np.zeros(3)
            rot_err = 0.0
        if pos_err < pos_tol and rot_err < rot_tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site)
        if target_mat is not None:
            J = np.vstack([jacp[:, dadr], jacr[:, dadr]])
            e = np.concatenate([e_pos, 0.3 * e_rot])
        else:
            J = jacp[:, dadr]
            e = e_pos
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(J.shape[0]), e)
        step = np.clip(dq, -0.2, 0.2)
        scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + step, lo, hi)
    ok = pos_err < pos_tol and rot_err < rot_tol
    return IkResult(ok=ok, q={j.name: float(scratch.qpos[a]) for j, a in zip(joints, qadr)},
                    pos_err_m=pos_err, rot_err_rad=rot_err, iterations=it)
