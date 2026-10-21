"""MujocoRobot: the SDK's MockRobot protocol double, with a MuJoCo A3 underneath.

Design in one paragraph: everything PROTOCOL-shaped (ack, descriptor, camera_layout,
daemon_status, capability gating, the watchdog clock, estop latch, record/navigation/
policy_stream replies, action vocabulary checks) is inherited unchanged from
`nori_sdk.mock.MockRobot`, so a client sees exactly the frames the SDK's own double emits.
What this class changes is the PHYSICS: MockRobot's integrated `pose` becomes the COMMANDED
target, MuJoCo tracks it with position servos and a differential-drive base, and telemetry
reports the MEASURED joint state. Actions therefore finish when the joint actually gets
there (or stalls on contact -> "clamped"), not instantly.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from nori_sdk.mock import A3_DESCRIPTOR, MockRobot

from . import ik, units
from .units import JointMap

ASSETS = Path(__file__).resolve().parent / "assets"
SCENE_XML = ASSETS / "scene.xml"
ROBOT_XML = ASSETS / "a3.xml"

ARM_JOINTS = (
    "shoulder_pitch", "shoulder_roll", "bicep_yaw", "elbow_pitch",
    "forearm_yaw", "wrist_pitch", "wrist_roll", "gripper",
)
CAMERA_ROLES = ("left_wrist", "right_wrist", "overhead", "front")

# Servo slew limit applied to commanded targets (rad/s). Feetech servos run a speed/accel
# profile on-board (arm_max_speed_raw / arm_acceleration in the xacro); its rad/s value is not
# published, so this is an approximation that keeps absolute moves from snapping.
ARM_MAX_SPEED = 2.0
# An action is "reached" when every target is within this many wire units...
ACTION_TOL = 2.0          # norm units (= 1% of range); lift: mm
# ...and it is "settled somewhere else" (-> clamped) when joints stop moving for this long.
SETTLE_S = 0.4
SETTLE_SPEED = 0.02       # rad/s (or m/s)
ACTION_DEADLINE_S = 8.0   # robot-side per-action deadline -> "timeout"


def _arm_maps(model: mujoco.MjModel) -> list[JointMap]:
    maps = []
    for side in ("left", "right"):
        for dof in ARM_JOINTS:
            joint = f"{side}_{dof}_joint"
            lo, hi = model.joint(joint).range
            maps.append(JointMap(
                key=f"{side}_arm_{dof}.pos", joint=joint, actuator=f"{side}_arm_{dof}",
                lower=float(lo), upper=float(hi), kind="gripper" if dof == "gripper" else "arm",
            ))
    lo, hi = model.joint("lift_extension_joint").range
    maps.append(JointMap("lift.pos", "lift_extension_joint", "lift", float(lo), float(hi), "lift"))
    return maps


class MujocoRobot(MockRobot):
    """Drop-in for `MockRobot` that simulates an A3 in MuJoCo.

        async with mujoco_session(MujocoRobot()) as robot:
            info = await robot.wait_ready()
            ...

    Parameters beyond MockRobot's:
      scene:        MJCF path (default: assets/scene.xml -- floor, table, cube)
      lift_mm:      initial lift height (default: MockRobot's seeded mid-travel value)
      timestep:     override the model's physics timestep
    """

    def __init__(
        self,
        *,
        descriptor: dict[str, Any] | None = A3_DESCRIPTOR,
        scene: str | Path | None = None,
        lift_mm: float | None = None,
        timestep: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(descriptor=descriptor, **kwargs)
        self.model = mujoco.MjModel.from_xml_path(str(scene or SCENE_XML))
        if timestep is not None:
            self.model.opt.timestep = timestep
        self.data = mujoco.MjData(self.model)
        self.maps = {m.key: m for m in _arm_maps(self.model)}
        self._act = {k: self.model.actuator(m.actuator).id for k, m in self.maps.items()}
        self._qadr = {k: int(self.model.joint(m.joint).qposadr[0]) for k, m in self.maps.items()}
        self._dadr = {k: int(self.model.joint(m.joint).dofadr[0]) for k, m in self.maps.items()}
        self._wheel_act = (self.model.actuator("left_wheel").id, self.model.actuator("right_wheel").id)
        self._base_body = self.model.body("base_footprint").id
        self._base_cmd = (0.0, 0.0)
        self._ctrl_target: dict[str, float] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._was_estopped = False
        self._was_watchdog_stop = False
        # Called as fn(robot) after every step(); used by the video recorder.
        self.step_callbacks: list[Any] = []

        # Start the sim where MockRobot's telemetry seed says the robot is.
        if lift_mm is not None:
            self.pose["lift.pos"] = float(lift_mm)
        for key, jm in self.maps.items():
            q = jm.to_sim(self.pose.get(key, 0.0))
            self.data.qpos[self._qadr[key]] = q
            self.data.ctrl[self._act[key]] = q
            self._ctrl_target[key] = q
        mujoco.mj_forward(self.model, self.data)
        # Equality constraints (lift middle stage, idler fingers) need a few steps to settle.
        for _ in range(int(0.3 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    # --- physics -----------------------------------------------------------------------------

    @property
    def sim_time(self) -> float:
        return float(self.data.time)

    def measured(self, key: str) -> float:
        """Measured joint value in wire units."""
        return self.maps[key].to_wire(float(self.data.qpos[self._qadr[key]]))

    def measured_state(self) -> dict[str, float]:
        """What telemetry reports: every descriptor joint + lift.pos (measured), and the base
        velocity in the SAME normalized units the jog uses (measured m/s / MAX_LINEAR_MPS)."""
        state = {k: round(self.measured(k), 3) for k in self.maps if k in self.pose}
        lin, ang = self.base_velocity()
        state["x.vel"] = round(lin / units.MAX_LINEAR_MPS, 4)
        state["theta.vel"] = round(ang / units.MAX_ANGULAR_RADPS, 4)
        return state

    def base_pose(self) -> tuple[float, float, float]:
        """(x, y, yaw) of base_footprint in the world, metres / radians."""
        pos = self.data.xpos[self._base_body]
        mat = self.data.xmat[self._base_body].reshape(3, 3)
        return float(pos[0]), float(pos[1]), math.atan2(mat[1, 0], mat[0, 0])

    def base_velocity(self) -> tuple[float, float]:
        """(forward m/s, yaw rate rad/s) in the base frame."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                 self._base_body, vel, 1)  # local frame: [ang(3), lin(3)]
        return float(vel[3]), float(vel[2])

    def _freeze_targets(self) -> None:
        """Hold where the robot IS -- what a servo does when its intent is dropped."""
        for key in self.maps:
            if key in self.pose:
                self.pose[key] = self.measured(key)
            self._ctrl_target[key] = float(self.data.qpos[self._qadr[key]])

    def step(self, dt: float) -> None:
        # MockRobot owns the clock, watchdog and the jog integration of TARGETS.
        super().step(dt)

        if self.estopped and not self._was_estopped:
            self._freeze_targets()
            self._finish_all("blocked", reason="estop_latched")
        self._was_estopped = self.estopped

        stop = self.watchdog == "stop"
        if stop and not self._was_watchdog_stop:
            # Control silence: the gateway drops ALL commanded intent, including a latched
            # action, and a stale target never self-resumes (teleop.hold() docstring).
            self._freeze_targets()
            self.action.clear()
            self._finish_all("timeout", reason="watchdog")
        self._was_watchdog_stop = stop

        if self.estopped or not self.online:
            lin = ang = 0.0
        else:
            lin, ang = self.pose.get("x.vel", 0.0), self.pose.get("theta.vel", 0.0)
        self._base_cmd = (lin, ang)
        wl, wr = units.wheel_speeds(lin, ang)
        self.data.ctrl[self._wheel_act[0]] = wl
        self.data.ctrl[self._wheel_act[1]] = wr

        h = self.model.opt.timestep
        n = max(1, int(round(dt / h)))
        frozen = self.estopped or not self.online
        for _ in range(n):
            for key, jm in self.maps.items():
                if key not in self.pose:
                    continue
                if not frozen:
                    goal = jm.to_sim(self.pose[key])
                    cur = self._ctrl_target.get(key, goal)
                    if jm.kind == "lift":
                        nxt = goal  # lift rate is already limited by the jog integration
                    else:
                        lim = ARM_MAX_SPEED * h
                        nxt = cur + max(-lim, min(lim, goal - cur))
                    self._ctrl_target[key] = nxt
                self.data.ctrl[self._act[key]] = self._ctrl_target[key]
            mujoco.mj_step(self.model, self.data)

        self._check_actions()
        for callback in list(self.step_callbacks):
            callback(self)

    # --- action lifecycle ----------------------------------------------------------------------

    def handle(self, raw: str | dict[str, Any]) -> list[str]:
        message = json.loads(raw) if isinstance(raw, str) else dict(raw)
        pose = message.get("pose")
        if (message.get("type") == "control" and isinstance(pose, dict)
                and "pose_targets" in self.capabilities and self.online):
            # Serve pose() with real IK instead of MockRobot's "nudge one joint" stand-in.
            rest = {k: v for k, v in message.items() if k not in ("pose", "action_id")}
            replies = super().handle(rest)
            return replies + self._handle_pose(pose, message.get("action_id"))
        replies = super().handle(message)
        action_id = message.get("action_id")
        if (message.get("type") == "control" and isinstance(message.get("action"), dict)
                and action_id and not self.estopped):
            vocab = self._joint_vocab() or []
            keys = [k for k, v in message["action"].items()
                    if k in vocab and k in self.maps and isinstance(v, (int, float))]
            if keys:
                # Hold the terminal frame back until the sim says the move is over.
                kept = []
                for raw_reply in replies:
                    frame = json.loads(raw_reply)
                    if frame.get("action_id") == action_id and frame.get("state") in (
                            "done", "clamped", "timeout"):
                        continue
                    kept.append(raw_reply)
                requested = {k: float(message["action"][k]) for k in keys}
                self._pending[action_id] = {
                    "targets": {k: self.pose[k] for k in keys},  # post-clamp targets
                    "clamped": any(abs(requested[k] - self.pose[k]) > 1e-6 for k in keys),
                    "started": self.sim_time,
                    "still_since": None,
                }
                return kept
        return replies

    def _status(self, action_id: str, state: str, reason: str | None = None) -> str:
        frame: dict[str, Any] = {"type": "action_status", "action_id": action_id,
                                 "state": state, "ts_ns": time.time_ns()}
        if reason:
            frame["reason"] = reason
        return self._emit(frame)

    def _handle_pose(self, pose: dict[str, Any], action_id: str | None) -> list[str]:
        sides = [k[:-4] for k in pose if k.endswith("_arm") and k[:-4] in ("left", "right")]
        if not sides:
            return [self._status(action_id, "blocked", "empty_pose")] if action_id else []
        if self.estopped:
            return [self._status(action_id, "blocked", "estop_latched")] if action_id else []
        side = sides[0]
        target = pose.get(f"{side}_arm") or {}
        position = target.get("position_m")
        orientation = target.get("orientation_xyzw")
        if (target.get("frame") != "base_footprint" or not isinstance(position, list)
                or len(position) != 3
                or (orientation is not None and (not isinstance(orientation, list)
                                                 or len(orientation) != 4))):
            return [self._status(action_id, "blocked", "bad_pose")] if action_id else []
        result = ik.solve(self.model, self.data, side, position, orientation)
        if not result.ok:
            return [self._status(action_id, "blocked", "no_ik_solution")] if action_id else []
        keys = []
        for joint, q in result.q.items():
            key = f"{side}_arm_{joint[len(side) + 1:-len('_joint')]}.pos"
            self.pose[key] = self.maps[key].to_wire(q)
            keys.append(key)
        if not action_id:
            return []
        self._pending[action_id] = {"targets": {k: self.pose[k] for k in keys},
                                    "clamped": False, "started": self.sim_time,
                                    "still_since": None}
        return [self._status(action_id, "accepted"), self._status(action_id, "active")]

    def _finish(self, action_id: str, state: str, reason: str | None = None) -> None:
        self._pending.pop(action_id, None)
        frame: dict[str, Any] = {"type": "action_status", "action_id": action_id,
                                 "state": state, "ts_ns": time.time_ns()}
        if reason:
            frame["reason"] = reason
        self._navigation_events.append(self._emit(frame))  # drained by the session pump

    def _finish_all(self, state: str, reason: str | None = None) -> None:
        for action_id in list(self._pending):
            self._finish(action_id, state, reason)

    def _check_actions(self) -> None:
        now = self.sim_time
        for action_id, pend in list(self._pending.items()):
            targets = pend["targets"]
            # A newer action on the same key supersedes the target we are waiting on.
            reached = all(abs(self.measured(k) - targets[k]) <= ACTION_TOL for k in targets)
            moving = any(abs(self.data.qvel[self._dadr[k]]) > SETTLE_SPEED for k in targets)
            ramping = any(abs(self._ctrl_target[k] - self.maps[k].to_sim(targets[k])) > 1e-4
                          for k in targets)
            if reached and not moving:
                self._finish(action_id, "clamped" if pend["clamped"] else self.action_outcome)
                continue
            if not moving and not ramping:
                pend["still_since"] = pend["still_since"] or now
                if now - pend["still_since"] >= SETTLE_S:
                    # Stopped short of the target: contact, gravity, or a stall.
                    self._finish(action_id, "clamped")
                    continue
            else:
                pend["still_since"] = None
            if now - pend["started"] > ACTION_DEADLINE_S:
                self._finish(action_id, "timeout")

    # --- telemetry and sensors -----------------------------------------------------------------

    def telemetry(self, state: dict[str, float] | None = None, with_status: bool = True) -> str:
        return super().telemetry(self.measured_state() if state is None else state, with_status)

    def _mock_imu(self) -> dict[str, Any]:
        frame = super()._mock_imu()
        sd = self.data.sensordata
        def read(name: str) -> list[float]:
            s = self.model.sensor(name)
            return [float(x) for x in sd[s.adr[0]: s.adr[0] + s.dim[0]]]
        w, x, y, z = read("imu_quat")
        frame["orientation_xyzw"] = [x, y, z, w]
        frame["angular_velocity_rad_s"] = read("imu_gyro")
        frame["linear_acceleration_m_s2"] = read("imu_accel")
        return frame

    def _mock_lidar(self) -> dict[str, Any]:
        """A planar scan ray-cast from the lidar_scan frame against collision geoms. The sensor's
        own housing is excluded; the rest of the robot (lift column, wheels) can occlude it."""
        frame = super()._mock_lidar()
        n = len(frame["ranges_m"])
        site = self.model.site("lidar").id
        origin = self.data.site_xpos[site].copy()
        mat = self.data.site_xmat[site].reshape(3, 3)
        angles = frame["angle_min_rad"] + frame["angle_increment_rad"] * np.arange(n)
        local = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], axis=1)
        dirs = (local @ mat.T).astype(np.float64)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # collision geoms only
        mujoco.mj_multiRay(self.model, self.data, origin, dirs.flatten(), group, 1,
                           self.model.body("lidar_link").id, geomid, dist, None, n, frame["range_max_m"])
        rmax = frame["range_max_m"]
        # No return -> range_max_m (keeps the frame strict JSON; ROS would use +inf).
        frame["ranges_m"] = [round(float(d), 4) if 0 <= d <= rmax else rmax for d in dist]
        return frame


__all__ = ["CAMERA_ROLES", "MujocoRobot", "ROBOT_XML", "SCENE_XML"]
