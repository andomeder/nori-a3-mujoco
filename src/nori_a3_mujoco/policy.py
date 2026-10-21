"""run_policy(): call a policy at a fixed rate on (telemetry + camera) observations and
send what it returns, without starving or fooling the robot's watchdog.

Works against ANY RemoteTeleop: the SDK's mock_session, this package's mujoco_session, or
hardware. It uses only public RemoteTeleop API (telemetry, camera_layout, frames(), jog(),
action(), estop(), info) -- no MuJoCo import, so it runs with just nori-sdk installed.

The policy is a plain function (sync or async):

    def policy(obs: Observation) -> PolicyOutput | dict | None: ...

and returns any of
    None                         hold: send nothing new (keep-alive only)
    {"action": {...}}            absolute joint targets, "<motor>.pos" -> norm units
    {"jog": {...}}               a jog payload (JogBuilder(...).build()), rates in [-1, 1]
    {"pose": {"left": [x, y, z]}}  TCP target(s) in base_footprint (needs pose_targets);
                                 a value may also be (position, orientation_xyzw)
    {"action": ..., "jog": ...}  several at once, in one control tick
    STOP                         end the run cleanly
    PolicyOutput(...)            the same, typed

Watchdog contract (the reason this is more than a `while True: sleep`):
  * The robot drops ALL commanded intent after t_stop_ms of control silence (500 ms LAN,
    1000 ms WAN). A keep-alive task therefore re-sends the policy's LAST jog (or an empty
    jog, which cancels nothing latched) at 20 Hz between ticks -- so a 2 Hz policy keeps a
    smooth base and does not have its action targets dropped mid-move.
  * But the keep-alive only speaks for a policy that is ALIVE: if a policy call has not
    returned within `stale_after` seconds (default: 80% of the robot's t_warn), the
    keep-alive goes silent and the robot's own watchdog stops it. A hung model must look
    like an absent operator, never like a healthy one.
  * On exit -- normal, STOP, exception, or Ctrl-C -- a zero jog is sent. `estop_on_error`
    (default True) latches E-STOP if the policy raises.
  * A latched E-STOP ends the run (reason "estop"); it is never reset from here.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from nori_sdk.teleop import JOG_HZ

STOP = object()  # sentinel a policy returns to end the run


@dataclass
class Observation:
    step: int
    t: float                                  # seconds since run start (wall clock)
    state: dict[str, float]                   # telemetry.state, LeRobot naming
    safety: str | None                        # "ok" | "safe_hold" | "latched" | None
    watchdog: str | None
    images: dict[str, Any] = field(default_factory=dict)  # role -> HxWx3 uint8 (if cameras)
    telemetry: Any = None                     # the full merged Telemetry object


@dataclass
class PolicyOutput:
    action: dict[str, float] | None = None
    jog: dict[str, Any] | None = None
    pose: dict[str, Any] | None = None       # side -> [x,y,z] or ([x,y,z], [x,y,z,w])
    stop: bool = False


@dataclass
class RunReport:
    steps: int = 0
    duration_s: float = 0.0
    reason: str = ""                          # "max_steps" | "duration" | "stop" | "estop" | "error" | "cancelled"
    overruns: int = 0                         # ticks where the policy took longer than 1/hz
    max_latency_s: float = 0.0
    mean_latency_s: float = 0.0
    safe_hold_ticks: int = 0
    wall_s: float = 0.0                       # wall-clock duration (== duration_s by default)
    missing_image_ticks: int = 0
    error: BaseException | None = None

    @property
    def achieved_hz(self) -> float:
        return self.steps / self.duration_s if self.duration_s > 0 else 0.0


Policy = Callable[[Observation], Any] | Callable[[Observation], Awaitable[Any]]


def _normalize(out: Any) -> PolicyOutput:
    if out is None:
        return PolicyOutput()
    if out is STOP:
        return PolicyOutput(stop=True)
    if isinstance(out, PolicyOutput):
        return out
    if isinstance(out, dict):
        unknown = set(out) - {"action", "jog", "pose", "stop"}
        if unknown:
            raise ValueError(f"policy returned unknown keys {sorted(unknown)}; "
                             "expected 'action', 'jog' and/or 'pose'")
        return PolicyOutput(action=out.get("action"), jog=out.get("jog"),
                            pose=out.get("pose"), stop=bool(out.get("stop", False)))
    raise TypeError(f"policy returned {type(out).__name__}; expected dict, PolicyOutput, "
                    "None or STOP")


class _FrameGrabber:
    """Keeps the latest composite video frame and crops camera roles out of it."""

    def __init__(self, robot: Any, roles: Sequence[str], track_timeout: float) -> None:
        self.robot = robot
        self.roles = list(roles)
        self.track_timeout = track_timeout
        self.latest: Any = None
        self.error: BaseException | None = None
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async for frame in self.robot.frames(track_timeout=self.track_timeout):
                self.latest = frame
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # no track: run continues, images stay empty
            self.error = exc

    def crops(self) -> dict[str, Any]:
        frame = self.latest
        if frame is None:
            return {}
        image = frame if hasattr(frame, "shape") else frame.to_ndarray(format="rgb24")
        layout = self.robot.camera_layout
        if layout is None:
            return {"composite": image}
        height, width = image.shape[:2]
        out = {}
        for role in self.roles:
            rect = layout.rect(role)
            if rect is None:
                continue
            x, y, w, h = rect
            out[role] = image[int(y * height):int((y + h) * height),
                              int(x * width):int((x + w) * width)]
        return out

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task


def _watchdog_warn_s(robot: Any) -> float:
    info = getattr(robot, "info", None)
    profile = getattr(info, "watchdog_profile", None)
    t_warn = getattr(profile, "t_warn_ms", None)
    return (float(t_warn) / 1000.0) if t_warn else 0.15  # LAN warn as the strict default


async def run_policy(
    robot: Any,
    policy: Policy,
    hz: float = 10.0,
    *,
    cameras: Sequence[str] | bool = False,
    max_steps: int | None = None,
    duration: float | None = None,
    stale_after: float | None = None,
    estop_on_error: bool = True,
    stop_on_estop: bool = True,
    wait_for_telemetry: float = 5.0,
    track_timeout: float = 5.0,
    on_step: Callable[[Observation, PolicyOutput], Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> RunReport:
    """Run `policy` at `hz` until max_steps / duration / STOP / E-STOP. See module docs.

    `cameras`: False for no images, True for every role in camera_layout, or a list of roles.
    `clock`: where "fixed rate" is measured. Default: the wall clock (time.monotonic), which
    is right for hardware. In simulation pass the SIM clock (`clock=lambda: bot.sim_time`) so
    the policy ticks every 1/hz of simulated time even when the sim runs slower than real
    time (software rendering) -- otherwise a slow sim looks like a robot that barely moves.
    The watchdog keep-alive always runs on the wall clock, as the robot's watchdog would.
    Call `await robot.wait_ready()` first.
    """
    now_fn = clock or time.monotonic
    if hz <= 0:
        raise ValueError("hz must be > 0")
    period = 1.0 / hz
    stale = stale_after if stale_after is not None else 0.8 * _watchdog_warn_s(robot)
    stale = max(stale, period * 1.5)  # never stricter than the policy's own tick
    report = RunReport()
    latencies: list[float] = []

    # telemetry: the policy needs a state to act on
    deadline = time.monotonic() + wait_for_telemetry
    while robot.telemetry is None:
        if time.monotonic() > deadline:
            raise TimeoutError("no telemetry within wait_for_telemetry")
        await asyncio.sleep(0.02)

    grabber: _FrameGrabber | None = None
    if cameras:
        layout = robot.camera_layout
        roles = list(layout.tiles) if cameras is True and layout else list(cameras or [])
        grabber = _FrameGrabber(robot, roles, track_timeout)
        grabber.start()

    state = {"jog": {}, "alive_at": time.monotonic(), "seq_sent": 0}

    async def send_jog(payload: dict[str, Any]) -> None:
        await robot.jog(payload)  # duration=0: exactly ONE control frame, strict checks kept

    async def keepalive() -> None:
        interval = 1.0 / JOG_HZ
        while True:
            await asyncio.sleep(interval)
            if time.monotonic() - state["alive_at"] <= stale:
                await send_jog(state["jog"])
            # else: SILENCE on purpose -- the robot's watchdog takes it from here.

    feeder = asyncio.create_task(keepalive())
    start = now_fn()
    wall_start = time.monotonic()
    next_tick = start
    step = 0
    try:
        while True:
            now = now_fn()
            if max_steps is not None and step >= max_steps:
                report.reason = "max_steps"
                break
            if duration is not None and now - start >= duration:
                report.reason = "duration"
                break
            tel = robot.telemetry
            if stop_on_estop and tel is not None and tel.safety == "latched":
                report.reason = "estop"
                break
            if tel is not None and tel.safety == "safe_hold":
                report.safe_hold_ticks += 1
            images = grabber.crops() if grabber else {}
            if grabber and not images:
                report.missing_image_ticks += 1
            obs = Observation(step=step, t=now - start, state=dict(tel.state if tel else {}),
                              safety=tel.safety if tel else None,
                              watchdog=tel.watchdog if tel else None,
                              images=images, telemetry=tel)

            t0 = time.monotonic()
            out = policy(obs)
            if inspect.isawaitable(out):
                out = await out
            latency = time.monotonic() - t0
            latencies.append(latency)
            if latency > period:
                report.overruns += 1
            cmd = _normalize(out)
            if on_step is not None:
                maybe = on_step(obs, cmd)
                if inspect.isawaitable(maybe):
                    await maybe
            if cmd.stop:
                report.reason = "stop"
                break

            state["alive_at"] = time.monotonic()
            if cmd.jog is not None:
                state["jog"] = dict(cmd.jog)
            if cmd.action:
                await robot.action(dict(cmd.action))  # no wait: latched target
            for side, target in (cmd.pose or {}).items():
                if (isinstance(target, (tuple, list)) and len(target) == 2
                        and isinstance(target[0], (tuple, list))):
                    position, orientation = target
                else:
                    position, orientation = target, None
                await robot.pose(side, list(position),
                                 list(orientation) if orientation is not None else None)
            await send_jog(state["jog"])  # every tick carries a control frame
            step += 1
            report.steps = step

            next_tick += period
            if clock is None:
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    next_tick = time.monotonic()  # fell behind: don't burst to catch up
            else:
                if now_fn() > next_tick:
                    next_tick = now_fn()
                while now_fn() < next_tick:
                    await asyncio.sleep(0)  # yield until the injected clock catches up
                    state["alive_at"] = time.monotonic()  # waiting on the clock, not hung
    except asyncio.CancelledError:
        report.reason = "cancelled"
        raise
    except KeyboardInterrupt:
        report.reason = "cancelled"
        with contextlib.suppress(Exception):
            robot.estop()
        raise
    except Exception as exc:
        report.reason = "error"
        report.error = exc
        if estop_on_error:
            with contextlib.suppress(Exception):
                robot.estop()
        raise
    finally:
        feeder.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await feeder
        with contextlib.suppress(Exception):
            await send_jog({k: ({kk: 0.0 for kk in v} if isinstance(v, dict) else 0.0)
                            for k, v in state["jog"].items()})
        if grabber:
            await grabber.stop()
        report.duration_s = now_fn() - start
        report.wall_s = time.monotonic() - wall_start
        if latencies:
            report.max_latency_s = max(latencies)
            report.mean_latency_s = sum(latencies) / len(latencies)
    return report


__all__ = ["STOP", "Observation", "PolicyOutput", "RunReport", "run_policy"]
