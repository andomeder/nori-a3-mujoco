"""Optional Gymnasium wrapper: a synchronous, fixed-step A3 environment.

    import gymnasium as gym
    from nori_a3_mujoco.gym_env import NoriA3Env
    env = NoriA3Env(cameras=("front",))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

No asyncio and no RemoteTeleop: each step() hands ONE nori-protocol control frame to the
MujocoRobot (so descriptor vocabulary, clamping, E-STOP latch and the watchdog clock are the
same code paths the SDK session exercises) and then advances 1/control_hz of sim time.

Action (Box, 19 dims, in WIRE units):
    [0:16]  absolute targets for the 16 descriptor arm joints, norm units (grippers 0..100)
    [16]    lift target in mm (sent as a lift JOG toward it: the protocol has no absolute lift)
    [17:19] base linear / angular jog rates in [-1, 1]
Observation (Dict): "state" = the same 19 quantities as measured, plus "images/<role>".
Reward is 0 unless you pass reward_fn(env, obs) -> float. Episodes end on E-STOP
(terminated) or after max_steps (truncated).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from nori_sdk.mock import A3_DESCRIPTOR
from nori_sdk.mock.robot import CENTRAL_LIFT_MM_PER_S

from .robot import MujocoRobot
from .video import CameraRenderer

JOINTS = list(A3_DESCRIPTOR["joints"])
STATE_KEYS = JOINTS + ["lift.pos", "x.vel", "theta.vel"]


class NoriA3Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, *, control_hz: float = 10.0, sim_hz: float = 100.0,
                 cameras: tuple[str, ...] = (), image_size: tuple[int, int] = (240, 320),
                 max_steps: int = 500, reward_fn: Any = None, render_mode: str | None = None,
                 robot_kwargs: dict[str, Any] | None = None) -> None:
        self.control_hz, self.sim_hz = control_hz, sim_hz
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.max_steps = max_steps
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.robot_kwargs = dict(robot_kwargs or {})
        ranges = A3_DESCRIPTOR["ranges"]
        lo = [ranges[k][0] for k in JOINTS] + [0.0, -1.0, -1.0]
        hi = [ranges[k][1] for k in JOINTS] + [700.0, 1.0, 1.0]
        self.action_space = spaces.Box(np.array(lo, np.float32), np.array(hi, np.float32))
        obs_spaces: dict[str, spaces.Space] = {
            "state": spaces.Box(-np.inf, np.inf, shape=(len(STATE_KEYS),), dtype=np.float32)}
        h, w = image_size
        for role in self.cameras:
            obs_spaces[f"images/{role}"] = spaces.Box(0, 255, (h, w, 3), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)
        self.robot: MujocoRobot | None = None
        self._cam: CameraRenderer | None = None
        self._view: CameraRenderer | None = None
        self._seq = 0
        self._steps = 0

    # --- gym API -------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.close_renderers()
        self.robot = MujocoRobot(**self.robot_kwargs)
        self._seq = 0
        self._steps = 0
        self._send({})  # first control frame: starts the dead-man clock fed
        return self._obs(), {}

    def step(self, action):
        assert self.robot is not None, "call reset() first"
        a = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low,
                    self.action_space.high)
        dt = 1.0 / self.control_hz
        lift_now = self.robot.measured("lift.pos")
        lift_rate = float(np.clip((a[16] - lift_now) / (CENTRAL_LIFT_MM_PER_S * dt), -1, 1))
        self._send({
            "action": {k: float(v) for k, v in zip(JOINTS, a[:16])},
            "jog": {"lift": lift_rate, "base": {"linear": float(a[17]), "angular": float(a[18])}},
        })
        for _ in range(max(1, int(round(self.sim_hz / self.control_hz)))):
            self.robot.step(1.0 / self.sim_hz)
        self.robot.drain_events()
        self._steps += 1
        obs = self._obs()
        reward = float(self.reward_fn(self, obs)) if self.reward_fn else 0.0
        terminated = bool(self.robot.estopped)
        truncated = self._steps >= self.max_steps
        return obs, reward, terminated, truncated, {"sim_time": self.robot.sim_time,
                                                    "watchdog": self.robot.watchdog}

    def render(self):
        if self.robot is None:
            return None
        if self._view is None:
            self._view = CameraRenderer(self.robot.model, 640, 480, shadows=True)
        return self._view.render(self.robot.data, "demo_wide")

    def close(self):
        self.close_renderers()

    # --- helpers -------------------------------------------------------------------------------

    def estop(self) -> None:
        self.robot.handle({"type": "command", "estop": True})

    def _send(self, body: dict[str, Any]) -> None:
        self._seq += 1
        self.robot.handle({"type": "control", "seq": self._seq, **body})

    def _obs(self) -> dict[str, np.ndarray]:
        state = self.robot.measured_state()
        out = {"state": np.array([state.get(k, 0.0) for k in STATE_KEYS], dtype=np.float32)}
        if self.cameras:
            if self._cam is None:
                h, w = self.image_size
                self._cam = CameraRenderer(self.robot.model, w, h)
            for role in self.cameras:
                out[f"images/{role}"] = self._cam.render(self.robot.data, role)
        return out

    def close_renderers(self) -> None:
        for r in (self._cam, self._view):
            if r is not None:
                r.close()
        self._cam = self._view = None
