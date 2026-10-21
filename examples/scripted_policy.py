"""A scripted (no learning) policy run through run_policy(): drive to the table using the
front camera, reach next to the cube with pose() (IK in the sim), close the gripper, raise the
lift, back away.

It does NOT pick the cube up, and does not claim to: in this sim the closed fingers do not
make contact with a cube placed at the TCP (see README, "Grasping"). It exercises the
policy loop, cameras in the loop, pose/action/jog outputs and the phase/settle logic.

    python examples/scripted_policy.py                 # MuJoCo sim, front camera in the loop
    python examples/scripted_policy.py --backend mock  # SDK MockRobot (no physics, no video)
    python examples/scripted_policy.py --record ../demo.mp4

The SAME policy object runs on both backends: on the mock there are no images, so the
"drive until the red cube looks big enough" step falls back to a timer. Phases advance on
TELEMETRY (joints stopped moving), not on wall-clock time, so the script tolerates a slow
sim (software rendering) the same way it would tolerate a slow robot.

What is scripted here, honestly: the cube's position relative to the base when the robot
stops is a scene prior (~0.34 m ahead, 0.12 m left), not something the policy perceives.
Only the stopping distance is closed-loop on the camera.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np
from nori_sdk.motion import JogBuilder

from nori_a3_mujoco.policy import STOP, Observation, run_policy

SIDE_GRASP_XYZW = [0.5, 0.5, 0.5, 0.5]   # TCP approach (+z) along base +x, fingers spread in y
CUBE_REL = (0.34, 0.12, 0.80)            # scene prior, base_footprint frame (metres)
RED_STOP_FRACTION = 0.0125               # front-camera red-pixel fraction at ~0.66 m


def red_fraction(image: np.ndarray) -> float:
    img = image.astype(np.int16)
    mask = (img[..., 0] > 150) & (img[..., 1] < 80) & (img[..., 2] < 80)
    return float(mask.mean())


class ReachPolicy:
    """A small phase machine. Call it with an Observation, get a command back."""

    PHASES = ("open", "drive", "pregrasp", "grasp", "close", "lift", "back", "done")

    def __init__(self, descriptor, hz: float, caption=None) -> None:
        self.jog = JogBuilder
        self.descriptor = descriptor
        self.hz = hz
        self.phase = "open"
        self.ticks = 0          # ticks spent in the current phase
        self.still = 0          # consecutive ticks with no joint motion
        self.prev: dict[str, float] | None = None
        self.caption = caption or (lambda text: None)
        self.log: list[str] = []

    def _enter(self, phase: str) -> None:
        self.log.append(phase)
        self.caption(phase)
        self.phase, self.ticks, self.still = phase, 0, 0

    def _settled(self, state: dict[str, float], min_s: float = 0.5) -> bool:
        if self.prev is not None:
            keys = [k for k in state if k.endswith(".pos")]
            moved = max((abs(state[k] - self.prev.get(k, state[k])) for k in keys), default=0)
            self.still = self.still + 1 if moved < 0.25 else 0
        return self.ticks >= min_s * self.hz and self.still >= 3

    def __call__(self, obs: Observation):
        state = obs.state
        self.ticks += 1
        try:
            return self._step(obs, state)
        finally:
            self.prev = dict(state)

    def _step(self, obs: Observation, state: dict[str, float]):
        x, y, z = CUBE_REL
        if self.phase == "open":
            if self.ticks == 1:
                self.caption("open left gripper")
                return {"action": {"left_arm_gripper.pos": 100.0}}
            if self._settled(state):
                self._enter("drive")
            return None
        if self.phase == "drive":
            front = obs.images.get("front")
            if front is not None:
                frac = red_fraction(front)
                done = frac >= RED_STOP_FRACTION
                rate = 0.3 if frac > 0.7 * RED_STOP_FRACTION else 0.6
            else:  # mock: no camera, drive for a fixed time
                done, rate = self.ticks > 2.2 * self.hz, 0.6
            if done:
                self._enter("pregrasp")
                return {"jog": {}}  # explicit base stop
            return {"jog": self.jog(self.descriptor).base(linear=rate).build()}
        if self.phase == "pregrasp":
            if self.ticks == 1:
                return {"pose": {"left": ([x - 0.07, y, z], SIDE_GRASP_XYZW)}}
            if self._settled(state, 1.0):
                self._enter("grasp")
            return None
        if self.phase == "grasp":
            if self.ticks == 1:
                return {"pose": {"left": ([x, y, z], SIDE_GRASP_XYZW)}}
            if self._settled(state, 1.0):
                self._enter("close")
            return None
        if self.phase == "close":
            if self.ticks == 1:
                return {"action": {"left_arm_gripper.pos": 0.0}}
            if self._settled(state, 0.8):
                self._enter("lift")
            return None
        if self.phase == "lift":
            if state.get("lift.pos", 0.0) >= 480.0 or self.ticks > 6 * self.hz:
                self._enter("back")
                return {"jog": {}}
            return {"jog": self.jog(self.descriptor).central_lift(1.0).build()}
        if self.phase == "back":
            if self.ticks > 1.5 * self.hz:
                self._enter("done")
                return {"jog": {}}
            return {"jog": self.jog(self.descriptor).base(linear=-0.5).build()}
        return STOP


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sim", "mock"), default="sim")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--record", help="sim only: write an mp4 of the run here")
    ap.add_argument("--stills", action="store_true", help="with --record: also save PNG stills")
    args = ap.parse_args()

    recorder = None
    if args.backend == "sim":
        from nori_a3_mujoco import MujocoRobot, mujoco_session

        bot = MujocoRobot()
        if args.record:
            from nori_a3_mujoco.record import VideoRecorder

            recorder = VideoRecorder(bot, fps=30)
        session = mujoco_session(bot)
    else:
        from nori_sdk.mock import A3_DESCRIPTOR, MockRobot, mock_session

        session = mock_session(MockRobot(descriptor=A3_DESCRIPTOR))

    async with session as robot:
        info = await robot.wait_ready()
        caption = (lambda text: setattr(recorder, "caption", text)) if recorder else None
        policy = ReachPolicy(info.descriptor, args.hz, caption)
        # In the sim, tick on SIM time so a slow (software-GL) render can't starve the policy
        # of physics between ticks. On the mock/hardware: the wall clock.
        clock = (lambda: bot.sim_time) if args.backend == "sim" else None
        report = await run_policy(robot, policy, hz=args.hz, cameras=["front"],
                                  duration=60.0, track_timeout=1.0, clock=clock)
        tel = robot.telemetry
        print(f"phases: {' -> '.join(policy.log)}")
        print(f"run: {report.steps} steps in {report.duration_s:.1f}s policy-clock "
              f"({report.wall_s:.1f}s wall), {report.achieved_hz:.1f} Hz, "
              f"reason={report.reason}, overruns={report.overruns}, "
              f"max policy latency {report.max_latency_s * 1000:.1f} ms, "
              f"ticks without images={report.missing_image_ticks}")
        print(f"final: lift={tel.state.get('lift.pos'):.1f} mm, "
              f"left gripper={tel.state.get('left_arm_gripper.pos'):.1f}, safety={tel.safety}")
        if args.backend == "sim":
            cube_z = float(bot.data.body("cube").xpos[2])
            x, y, _ = bot.base_pose()
            print(f"base ended at x={x:.3f} y={y:.3f}; cube z={cube_z:.3f} m "
                  f"({'moved' if abs(cube_z - 0.785) > 0.005 else 'untouched'})")
    if recorder is not None:
        n = len(recorder.frames)
        stills = None
        if args.stills:
            stills = {f"still_{i}.png": int(n * f) for i, f in
                      enumerate((0.05, 0.3, 0.55, 0.75, 0.95))}
        path = recorder.save(args.record, stills)
        recorder.close()
        print(f"wrote {path} ({n} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
