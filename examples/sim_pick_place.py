"""Pick-and-place in the MuJoCo sim, through the Nori SDK session API.

    python examples/sim_pick_place.py
    python examples/sim_pick_place.py --record ../demo.mp4 --stills ../stills

Same shape as the SDK's mock_pick_place.py (discover the robot, drive, use a gripper,
record an episode, stop safely), but the task is real here: find the red cube with the
overhead camera, drive until it is in reach, pick it up with the left claw, carry it to the
green marker, put it down, back off, estop. Everything goes over SDK calls -- jog, snapshot,
pose (on-board IK), action -- so the logic is hardware-shaped; only calibration (grasp point,
table height, camera extrinsics) is sim-specific.

(The SDK example itself, with only the session line swapped, is sdk_mock_pick_place.py.)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from nori_a3_mujoco import MujocoRobot, mujoco_session
from nori_a3_mujoco.perception import SceneLocator
from nori_a3_mujoco.pick_place import (LIFT_MM, drive_to_standoff, locate, move_lift,
                                       pick_and_place)

CAPTIONS = {
    "open": "open the claw", "ready": "arm to ready, above the table",
    "pregrasp": "pre-grasp: behind the cube", "approach": "approach: cube into the jaw",
    "grasp": "close: grasp the cube", "lift": "lift the cube clear of the table",
    "carry": "carry to the green marker", "lower": "lower onto the table",
    "release": "release", "retreat": "back the claw off", "clear": "arm clear",
}


async def run(robot, bot: MujocoRobot | None = None, say=print) -> bool:
    info = await robot.wait_ready()
    say(f"connected: {info.descriptor.joints[:2]}... (+{len(info.descriptor.joints) - 2} more)")
    locator = SceneLocator("overhead")
    # With a recorder attached the sim runs 10-15x slower than wall time, so time the guards
    # on the sim clock and give pose()/action() waits (wall-clock in the SDK) plenty of slack.
    clock = (lambda: bot.sim_time) if bot is not None else None
    wait_s = 600.0 if bot is not None else 60.0

    say("lift to table height")
    lift = await move_lift(robot, info.descriptor, LIFT_MM, clock=clock)
    say(f"lift at {lift:.0f} mm")

    say("look for the cube and drive up to it")
    cube = await drive_to_standoff(robot, info.descriptor, locator, clock=clock)
    if cube is None:
        say("no cube in view")
        return False
    place = await locate(robot, locator, what="marker")
    if place is None:
        say("no place marker in view")
        return False
    say(f"cube at {cube.round(3).tolist()}, marker at {place.round(3).tolist()} (base_footprint)")

    await robot.record("session_start")
    await robot.record("episode_start", task="pick the red cube, place it on the marker")
    report = await pick_and_place(robot, cube, place,
                                  on_step=lambda s: say(CAPTIONS.get(s.name, s.name)),
                                  timeout=wait_s)
    await robot.record("episode_stop")
    await robot.record("session_end")
    say(f"pick-and-place: {report}")

    if bot is not None:
        z = float(bot.data.body("cube").xpos[2])
        mk = bot.data.geom("place_marker").xpos
        off = float(((bot.data.body("cube").xpos[:2] - mk[:2]) ** 2).sum() ** 0.5)
        say(f"sim check: cube z={z:.3f} m, {off * 1000:.0f} mm from the marker centre")

    say("estop")
    robot.estop()
    await asyncio.sleep(0.3)
    print(f"estopped -- latched={robot.telemetry.safety}")
    return report.ok


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", help="write an mp4 of the run here")
    ap.add_argument("--stills", help="with --record: directory for PNG stills")
    args = ap.parse_args()

    bot = MujocoRobot()
    recorder = None
    if args.record:
        from nori_a3_mujoco.record import VideoRecorder

        recorder = VideoRecorder(bot, fps=30)

    marks: dict[str, int] = {}

    def say(text: str) -> None:
        print(text)
        if recorder is not None:
            recorder.caption = text
            marks.setdefault(text, len(recorder.frames))

    async with mujoco_session(bot) as robot:
        ok = await run(robot, bot, say)

    if recorder is not None:
        stills = None
        if args.stills:
            from pathlib import Path

            names = [("lift to table height", "01_start"),
                     ("look for the cube and drive up to it", "02_drive"),
                     (CAPTIONS["approach"], "03_approach"), (CAPTIONS["grasp"], "04_grasp"),
                     (CAPTIONS["lift"], "05_lifted"), (CAPTIONS["carry"], "06_carry"),
                     (CAPTIONS["release"], "07_placed"), (CAPTIONS["clear"], "08_clear")]
            order = sorted(marks.values()) + [len(recorder.frames)]
            stills = {}
            for caption, name in names:
                if caption in marks:
                    start = marks[caption]
                    nxt = min([i for i in order if i > start] or [start + 1])
                    stills[str(Path(args.stills) / f"{name}.png")] = (start + nxt) // 2 \
                        if name not in ("01_start",) else start + 2
        path = recorder.save(args.record, stills)
        print(f"wrote {path} ({len(recorder.frames)} frames)")
        recorder.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
