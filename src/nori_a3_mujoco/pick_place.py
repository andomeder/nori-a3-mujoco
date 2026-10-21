"""Pick-and-place for the demo scene, written against the PUBLIC SDK session API only
(`wait_ready`, `jog`, `snapshot`, `pose`, `action`), so the same code drives the MuJoCo sim
and -- modulo calibration -- a real A3.

    cube   = await locate(robot, locator)            # overhead camera -> base_footprint
    await drive_to_standoff(robot, locator)          # base forward until cube is in reach
    report = await pick_and_place(robot, cube, place)

Grasp: a horizontal side grasp with the LEFT claw (approach along base +x, fingers closing
across base y), targeted at the claw's calibrated grasp point (`grasp.py`), not the URDF TCP.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from nori_sdk.motion import JogBuilder

from .grasp import SIDE_GRASP_XYZW, tcp_for_grasp

LIFT_MM = 300.0          # column height for table work (shoulders ~0.87 m above the floor)
STANDOFF_X = 0.34        # where the base stops: cube this far ahead of base_footprint (m)
GRASP_DZ = 0.02          # grasp point at the cube's top-face height: the wrist housing reaches 37 mm
                         # below the grasp point, so this keeps it 3 mm clear of the table
PREGRASP_BACK = 0.05     # pre-grasp: this far behind the cube along the approach
READY_UP = 0.06          # ready: above the pre-grasp, clear of the table edge
CARRY_UP = 0.08          # transport height above the table contact height
OPEN, CLOSED = 50.0, 0.0  # gripper units (0 closed .. 100 open, 50 ~ 45 deg per finger)
RELEASE = 70.0           # open wider to let go, so the tip hooks clear the cube
RELEASE_DZ = 0.002


@dataclass
class Step:
    name: str
    kind: str              # "pose" | "gripper"
    target: Any


@dataclass
class PickPlaceReport:
    ok: bool
    steps: list[tuple[str, str, str | None]] = field(default_factory=list)  # name, state, reason

    def __str__(self) -> str:
        return " -> ".join(f"{n}:{s}" for n, s, _ in self.steps) + f"  ok={self.ok}"


def plan(cube: Any, place: Any, side: str = "left") -> list[Step]:
    """The waypoint sequence (grasp-point positions in base_footprint)."""
    cx, cy, cz = (float(v) for v in cube)
    px, py, pz = (float(v) for v in place)
    gz, tz = cz + GRASP_DZ, pz + GRASP_DZ
    return [
        Step("open", "gripper", OPEN),
        Step("ready", "pose", (cx - PREGRASP_BACK, cy, gz + READY_UP)),
        Step("pregrasp", "pose", (cx - PREGRASP_BACK, cy, gz)),
        Step("approach", "pose", (cx, cy, gz)),
        Step("grasp", "gripper", CLOSED),
        Step("lift", "pose", (cx, cy, gz + CARRY_UP)),
        Step("carry", "pose", (px, py, tz + CARRY_UP)),
        Step("lower", "pose", (px, py, tz + RELEASE_DZ)),
        Step("release", "gripper", RELEASE),
        Step("retreat", "pose", (px - PREGRASP_BACK, py, tz + RELEASE_DZ + 0.01)),
        Step("clear", "pose", (px - PREGRASP_BACK, py, tz + READY_UP)),
    ]


def command(step: Step, side: str = "left") -> dict[str, Any]:
    """A plan step as a run_policy output (pose / action dict)."""
    if step.kind == "gripper":
        return {"action": {f"{side}_arm_gripper.pos": float(step.target)}}
    quat = list(SIDE_GRASP_XYZW[side])
    return {"pose": {side: (tcp_for_grasp(side, step.target, quat), quat)}}


async def pick_and_place(robot: Any, cube: Any, place: Any, side: str = "left", *,
                         on_step: Callable[[Step], Any] | None = None,
                         timeout: float = 60.0) -> PickPlaceReport:
    report = PickPlaceReport(ok=True)
    quat = list(SIDE_GRASP_XYZW[side])
    for step in plan(cube, place, side):
        if on_step is not None:
            on_step(step)
        if step.kind == "gripper":
            st = await robot.action({f"{side}_arm_gripper.pos": float(step.target)},
                                    wait=True, timeout=timeout)
        else:
            st = await robot.pose(side, tcp_for_grasp(side, step.target, quat), quat,
                                  wait=True, timeout=timeout)
        state = getattr(st, "state", None)
        report.steps.append((step.name, str(state), getattr(st, "reason", None)))
        # Closing on an object ends "clamped" (stalled on the cube) -- that is the grasp.
        if state in (None, "blocked", "timeout"):
            report.ok = False
            break
    return report


async def wait_telemetry(robot: Any, timeout: float = 10.0) -> dict[str, float]:
    loop = asyncio.get_running_loop()
    t_end = loop.time() + timeout
    while robot.telemetry is None:
        if loop.time() > t_end:
            raise TimeoutError("no telemetry from the robot")
        await asyncio.sleep(0.05)
    return robot.telemetry.state


async def locate(robot: Any, locator: Any, *, samples: int = 3,
                 what: str = "cube") -> np.ndarray | None:
    """Average a few overhead-camera estimates (base_footprint, metres)."""
    found = []
    for _ in range(samples):
        await wait_telemetry(robot)
        image = await robot.snapshot(role=locator.role, settle=0.0)
        state = robot.telemetry.state
        p = locator.cube(image, state) if what == "cube" else locator.marker(image, state)
        if p is not None:
            found.append(p)
    return np.mean(found, axis=0) if found else None


async def move_lift(robot: Any, descriptor: Any, target_mm: float, tol: float = 3.0,
                    max_s: float = 20.0, clock: Any = None) -> float:
    """Proportional lift jog. `clock` (default: the event-loop clock) times the `max_s` guard;
    pass the sim clock when rendering makes the sim run slower than wall time."""
    await wait_telemetry(robot)
    clock = clock or asyncio.get_running_loop().time
    t_end = clock() + max_s
    while clock() < t_end:
        err = target_mm - float(robot.telemetry.state.get("lift.pos", target_mm))
        if abs(err) <= tol:
            break
        rate = float(np.clip(err / 40.0, -1.0, 1.0))
        rate = rate if abs(rate) > 0.15 else 0.15 * np.sign(rate)
        await robot.jog(JogBuilder(descriptor).central_lift(rate).build(), duration=0.15)
    await robot.jog(JogBuilder(descriptor).central_lift(0.0).build())
    return float(robot.telemetry.state.get("lift.pos", 0.0))


async def drive_to_standoff(robot: Any, descriptor: Any, locator: Any,
                            standoff: float = STANDOFF_X, tol: float = 0.006,
                            max_s: float = 30.0, clock: Any = None) -> np.ndarray | None:
    """Creep forward until the cube is `standoff` ahead; returns the last cube estimate.
    `clock` as for move_lift."""
    clock = clock or asyncio.get_running_loop().time
    t_end = clock() + max_s
    cube = None
    while clock() < t_end:
        cube = await locate(robot, locator, samples=1)
        if cube is None:
            await robot.jog(JogBuilder(descriptor).base(linear=0.3).build(), duration=0.2)
            continue
        err = float(cube[0]) - standoff
        if err <= tol:
            break
        rate = float(np.clip(err * 2.0, 0.08, 0.6))
        await robot.jog(JogBuilder(descriptor).base(linear=rate).build(),
                        duration=0.15 if err < 0.1 else 0.3)
    await robot.jog(JogBuilder(descriptor).base(linear=0.0).build())
    await asyncio.sleep(0.5)  # let the base coast to rest before the final estimate
    return await locate(robot, locator) if cube is not None else None


__all__ = ["CARRY_UP", "LIFT_MM", "STANDOFF_X", "PickPlaceReport", "Step", "command",
           "drive_to_standoff", "locate", "move_lift", "pick_and_place", "plan", "wait_telemetry"]
