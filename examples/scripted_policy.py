"""A scripted (no learning) pick-and-place policy run through run_policy():

  lift to table height -> drive until the overhead camera puts the red cube in reach ->
  locate cube + green marker -> open -> ready -> pre-grasp -> approach -> close (grasp) ->
  lift -> carry -> lower -> release -> retreat -> clear -> done

    python examples/scripted_policy.py                 # MuJoCo sim, overhead camera in the loop
    python examples/scripted_policy.py --backend mock  # SDK MockRobot (no physics, no video)
    python examples/scripted_policy.py --record ../demo_scripted_policy.mp4 --stills ../stills_policy

One policy object, ticked at a fixed rate, returning jog / pose / action outputs; phases
advance on TELEMETRY (joints stopped moving), not on wall-clock time, so a slow sim (software
rendering) is tolerated the way a slow robot would be. The cube and marker positions come
from the overhead camera image + the robot's own kinematics (nori_a3_mujoco.perception);
the only scene prior is the table height. On the mock there are no images, so it falls back
to a timed drive and the scene's nominal positions -- the mock has no physics, so nothing is
picked there; it just proves the same policy runs against the SDK's mock session.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np
from nori_sdk.motion import JogBuilder

from nori_a3_mujoco.pick_place import LIFT_MM, STANDOFF_X, command, plan
from nori_a3_mujoco.policy import STOP, Observation, run_policy

NOMINAL_CUBE = np.array([STANDOFF_X, 0.12, 0.78])     # mock fallback (scene layout)
NOMINAL_PLACE = np.array([STANDOFF_X, 0.27, 0.78])


class PickPlacePolicy:
    """A small phase machine. Call it with an Observation, get a command back."""

    def __init__(self, descriptor, hz: float, caption=None, locator=None) -> None:
        self.descriptor = descriptor
        self.hz = hz
        self.locator = locator          # None on the mock
        self.phase = "lift_to_table"
        self.ticks = 0
        self.still = 0
        self.prev: dict[str, float] | None = None
        self.caption = caption or (lambda text: None)
        self.log: list[str] = [self.phase]
        self.cube: np.ndarray | None = None
        self.place: np.ndarray | None = None
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.steps = []
        self.step_i = 0
        self.caption("lift to table height")

    def _enter(self, phase: str, caption: str | None = None) -> None:
        self.log.append(phase)
        self.phase, self.ticks, self.still = phase, 0, 0
        self.caption(caption or phase)

    def _settled(self, state: dict[str, float], min_s: float = 0.5) -> bool:
        if self.prev is not None:
            keys = [k for k in state if k.endswith(".pos")]
            moved = max((abs(state[k] - self.prev.get(k, state[k])) for k in keys), default=0)
            self.still = self.still + 1 if moved < 0.2 else 0
        return self.ticks >= min_s * self.hz and self.still >= 3

    def _see(self, obs: Observation):
        img = obs.images.get("overhead") if self.locator else None
        if img is None:
            return None, None
        return self.locator.cube(img, obs.state), self.locator.marker(img, obs.state)

    def __call__(self, obs: Observation):
        state = obs.state
        self.ticks += 1
        try:
            return self._step(obs, state)
        finally:
            self.prev = dict(state)

    def _jog(self):
        return JogBuilder(self.descriptor)

    def _step(self, obs: Observation, state: dict[str, float]):
        if self.phase == "lift_to_table":
            err = LIFT_MM - state.get("lift.pos", LIFT_MM)
            if abs(err) <= 3.0 or self.ticks > 15 * self.hz:
                self._enter("drive", "drive until the cube is in reach (overhead camera)")
                return {"jog": self._jog().central_lift(0.0).build()}
            rate = float(np.clip(err / 40.0, -1, 1))
            rate = rate if abs(rate) > 0.15 else 0.15 * np.sign(rate)
            return {"jog": self._jog().central_lift(rate).build()}

        if self.phase == "drive":
            cube, _ = self._see(obs)
            if self.locator is None:      # mock: no camera
                done, rate = self.ticks > 2.0 * self.hz, 0.5
            elif cube is None:
                done, rate = self.ticks > 30 * self.hz, 0.3
            else:
                err = float(cube[0]) - STANDOFF_X
                done, rate = err <= 0.006, float(np.clip(err * 2.0, 0.08, 0.6))
            if done:
                self._enter("locate", "stop; locate cube and marker")
                return {"jog": self._jog().base(linear=0.0).build()}
            return {"jog": self._jog().base(linear=rate).build()}

        if self.phase == "locate":
            if self.ticks < 0.6 * self.hz:          # let the base coast to rest
                return None
            cube, marker = self._see(obs)
            if cube is not None and marker is not None:
                self.samples.append((cube, marker))
            if len(self.samples) >= 3 or (self.locator is None) or self.ticks > 3 * self.hz:
                if self.samples:
                    self.cube = np.mean([c for c, _ in self.samples], axis=0)
                    self.place = np.mean([m for _, m in self.samples], axis=0)
                else:
                    self.cube, self.place = NOMINAL_CUBE.copy(), NOMINAL_PLACE.copy()
                self.steps = plan(self.cube, self.place)
                self.step_i = 0
                self._enter(self.steps[0].name, CAPTIONS.get(self.steps[0].name))
                return command(self.steps[0])
            return None

        if self.step_i < len(self.steps) and self.phase == self.steps[self.step_i].name:
            step = self.steps[self.step_i]
            min_s = 0.8 if step.kind == "gripper" else 0.5
            if self._settled(state, min_s):
                self.step_i += 1
                if self.step_i >= len(self.steps):
                    self._enter("done", "done")
                    return STOP
                nxt = self.steps[self.step_i]
                self._enter(nxt.name, CAPTIONS.get(nxt.name))
                return command(nxt)
            return None
        return STOP


CAPTIONS = {
    "open": "open the claw", "ready": "arm to ready, above the table",
    "pregrasp": "pre-grasp: behind the cube", "approach": "approach: cube into the jaw",
    "grasp": "close: grasp the cube", "lift": "lift the cube clear of the table",
    "carry": "carry to the green marker", "lower": "lower onto the table",
    "release": "release", "retreat": "back the claw off", "clear": "arm clear",
}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("sim", "mock"), default="sim")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--record", help="sim only: write an mp4 of the run here")
    ap.add_argument("--stills", help="with --record: directory for PNG stills")
    args = ap.parse_args()

    recorder = None
    marks: dict[str, int] = {}
    if args.backend == "sim":
        from nori_a3_mujoco import MujocoRobot, mujoco_session

        bot = MujocoRobot()
        if args.record:
            from nori_a3_mujoco.record import VideoRecorder

            recorder = VideoRecorder(bot, fps=30)
        heights: list[float] = []
        bot.step_callbacks.append(lambda b: heights.append(float(b.data.body("cube").xpos[2])))
        session = mujoco_session(bot)
    else:
        from nori_sdk.mock import A3_DESCRIPTOR, MockRobot, mock_session

        session = mock_session(MockRobot(descriptor=A3_DESCRIPTOR))

    async with session as robot:
        info = await robot.wait_ready()
        marks: dict[str, int] = {}
        holder: list = []

        def caption(text: str) -> None:
            if recorder is not None:
                recorder.caption = text
                if holder:
                    marks.setdefault(holder[0].phase, len(recorder.frames))

        locator = None
        if args.backend == "sim":
            from nori_a3_mujoco.perception import SceneLocator

            locator = SceneLocator("overhead")
        policy = PickPlacePolicy(info.descriptor, args.hz, caption, locator)
        holder.append(policy)
        # In the sim, tick on SIM time so a slow (software-GL) render can't starve the policy
        # of physics between ticks. On the mock/hardware: the wall clock.
        clock = (lambda: bot.sim_time) if args.backend == "sim" else None
        report = await run_policy(robot, policy, hz=args.hz,
                                  cameras=["overhead"] if args.backend == "sim" else False,
                                  duration=90.0, track_timeout=1.0, clock=clock)
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
            cube = bot.data.body("cube").xpos
            mk = bot.data.geom("place_marker").xpos
            off = float(np.hypot(*(cube[:2] - mk[:2])))
            print(f"sim check: cube max z {max(heights):.3f} m (table contact z 0.780), "
                  f"final z {cube[2]:.3f} m, {off * 1000:.0f} mm from the marker centre")
    if recorder is not None:
        n = len(recorder.frames)
        stills = None
        if args.stills:
            from pathlib import Path

            order = sorted(marks.items(), key=lambda kv: kv[1])
            bounds = [i for _, i in order] + [n]
            wanted = {"drive": "01_drive", "approach": "02_approach", "grasp": "03_grasp",
                      "lift": "04_lifted", "carry": "05_carry", "release": "06_placed",
                      "done": "07_done"}
            stills = {}
            for k, (name, start) in enumerate(order):
                if name in wanted:
                    end = bounds[k + 1]
                    idx = (start + end) // 2 if name != "done" else n - 1
                    stills[str(Path(args.stills) / f"{wanted[name]}.png")] = idx
        path = recorder.save(args.record, stills)
        recorder.close()
        print(f"wrote {path} ({n} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
