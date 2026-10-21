"""Render the demo video: drive, turn, raise/lower the lift, move both arms and grippers.

    python examples/render_demo.py --out ../demo.mp4 --stills ../stills

Everything goes through the SDK session (mujoco_session -> RemoteTeleop) and run_policy();
the "policy" is a timeline keyed on SIM time, so the result is the same on a fast GPU box and
on a slow software-GL one (it just takes longer to render).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nori_sdk.motion import JogBuilder

from nori_a3_mujoco import MujocoRobot, mujoco_session
from nori_a3_mujoco.policy import STOP, run_policy
from nori_a3_mujoco.record import VideoRecorder

# (start_s, caption, command-builder)
def timeline(desc):
    jb = lambda: JogBuilder(desc)  # noqa: E731
    return [
        (0.0, "open both grippers (action)", lambda: {"action": {"left_arm_gripper.pos": 100.0,
                                                                 "right_arm_gripper.pos": 100.0}}),
        (1.2, "drive forward (jog base linear=0.6)", lambda: {"jog": jb().base(linear=0.6).build()}),
        (3.6, "turn left (angular=+0.5)", lambda: {"jog": jb().base(angular=0.5).build()}),
        (5.2, "turn right (angular=-0.5)", lambda: {"jog": jb().base(angular=-0.5).build()}),
        (6.8, "raise the lift (jog lift=+1)", lambda: {"jog": jb().central_lift(1.0).build()}),
        (11.0, "arms forward (pose -> sim IK)", lambda: {"jog": {}, "pose": {
            "left": [0.30, 0.18, 1.0], "right": [0.30, -0.18, 1.0]}}),
        (13.5, "close grippers (action)", lambda: {"action": {"left_arm_gripper.pos": 0.0,
                                                              "right_arm_gripper.pos": 0.0}}),
        (15.0, "wrist roll (action)", lambda: {"action": {"left_arm_wrist_roll.pos": 80.0,
                                                          "right_arm_wrist_roll.pos": -80.0}}),
        (16.5, "lower the lift (jog lift=-1)", lambda: {"jog": jb().central_lift(-1.0).build()}),
        (20.0, "estop() - latched", None),
        (21.0, "", "stop"),
    ]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--stills", default=None, help="directory for PNG stills")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    bot = MujocoRobot(lift_mm=250.0)
    rec = VideoRecorder(bot, fps=args.fps)
    async with mujoco_session(bot, video=False) as robot:
        info = await robot.wait_ready()
        steps = timeline(info.descriptor)
        fired: set[int] = set()

        def policy(obs):
            due = [i for i, (t, _, _) in enumerate(steps) if obs.t >= t and i not in fired]
            if not due:
                return None
            i = due[-1]
            fired.update(due)
            t, caption, make = steps[i]
            rec.caption = caption
            if make == "stop":
                return STOP
            if make is None:
                robot.estop()
                return None
            return make()

        report = await run_policy(robot, policy, hz=20.0, clock=lambda: bot.sim_time,
                                  stop_on_estop=False, duration=25.0)
        print(f"policy: {report.steps} steps, {report.duration_s:.1f}s sim, "
              f"{report.wall_s:.1f}s wall, reason={report.reason}")
        print("final telemetry:", {k: round(v, 1) for k, v in robot.telemetry.state.items()
                                   if k in ("lift.pos", "left_arm_gripper.pos", "x.vel")},
              "safety:", robot.telemetry.safety)

    n = len(rec.frames)
    stills = None
    if args.stills:
        Path(args.stills).mkdir(parents=True, exist_ok=True)
        moments = {"01_start.png": 0.5, "02_drive.png": 3.0, "03_turn.png": 4.5,
                   "04_lift_up.png": 10.8, "05_arms_forward.png": 13.3,
                   "06_grippers_closed.png": 14.8, "07_lift_down.png": 19.8}
        stills = {str(Path(args.stills).resolve() / k): int(v * args.fps) for k, v in moments.items()}
    out = rec.save(args.out)
    if stills:
        import imageio.v2 as imageio
        for path, idx in stills.items():
            imageio.imwrite(path, rec.frames[min(idx, n - 1)])
    rec.close()
    print(f"wrote {out} ({n} frames, {n / args.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
