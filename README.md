# nori-a3-mujoco

A MuJoCo simulation of the **Nori A3** two-arm mobile manipulator that sits behind the
[Nori Python SDK](https://github.com/Nori-Robotics/nori-sdk-py) (`nori-sdk`). It uses the same
session interface as the SDK's `mock_session()`/`MockRobot`, so SDK code runs against physics
with only the session constructor changed. The repo also includes a fixed-rate **policy runner**
(`run_policy`) that works against this sim and against the SDK's mock and real sessions.

```python
from nori_a3_mujoco import MujocoRobot, mujoco_session

async with mujoco_session(MujocoRobot()) as robot:     # was: async with mock_session() as robot:
    ...                                                 # everything else unchanged
```

> Status: weekend-scale demo, unofficial. Nori Robotics has not reviewed or endorsed it. The
> kinematics come straight from Nori's description. The dynamics, servo strength and
> calibration are approximations (see [Faithful vs approximate](#faithful-vs-approximate)).

## What's in here

| piece | file |
|---|---|
| Converted model (URDF/xacro → MJCF), floor/table/cube scene | `src/nori_a3_mujoco/assets/{a3,scene}.xml`, `scripts/build_model.py` |
| Sim backend, a `MockRobot` subclass driven by physics | `src/nori_a3_mujoco/robot.py`, `session.py`, `ik.py`, `video.py`, `units.py` |
| Policy runner `run_policy(robot, fn, hz)` | `src/nori_a3_mujoco/policy.py` |
| Torch action-chunk (ACT-style) policy adapter, device selection (CUDA/XPU/CPU) | `torch_policy.py`, `devices.py` |
| Gymnasium wrapper (optional) | `src/nori_a3_mujoco/gym_env.py` |
| Video recorder (main view and 4 camera tiles, captions) | `src/nori_a3_mujoco/record.py` |
| Examples | `examples/sim_pick_place.py` (the SDK's `mock_pick_place.py`, 2 lines changed), `scripted_policy.py`, `torch_policy.py`, `render_demo.py` |

## Setup (Arch Linux, terminal)

Requires Python 3.11–3.13 (pinned to 3.12 via `.python-version`) and
[uv](https://docs.astral.sh/uv/) (`sudo pacman -S uv`).

```bash
# rendering: EGL comes with mesa (Intel Arc: mesa provides the iris/xe EGL driver)
sudo pacman -S --needed mesa ffmpeg
# Intel Arc GPU compute for torch XPU (only needed for TORCH=xpu)
sudo pacman -S --needed intel-compute-runtime level-zero-loader

make setup            # .venv from uv.lock: mujoco, nori-sdk, video, gym, xacro, CPU torch
make setup-xpu        # same, but torch 2.9.1+xpu from download.pytorch.org/whl/xpu
make test             # 33 tests, headless
make help             # all targets
```

`make setup` also copies `.env.example` to `.env`. The Makefile exports `.env` (`MUJOCO_GL`,
`NORI_DEVICE`, `NORI_IPEX`, `NORI_VIEWER`).

**pip instead of uv.** `requirements/*.txt` are exported from `uv.lock`:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/xpu.txt -r requirements/dev.txt   # or cpu.txt / base.txt (no torch)
pip install -e . --no-deps
```

`nori-sdk==1.1.0` comes from PyPI. It was developed against
`Nori-Robotics/nori-sdk-py@a8db25b`.

## Running

| | command | display? |
|---|---|---|
| SDK example against the sim | `make example` | no (EGL) |
| Scripted policy through `run_policy` | `make policy` | no |
| Same, recorded to `videos/scripted_policy.mp4` | `make policy-demo` | no |
| Showcase video and stills → `videos/` | `make demo` | no |
| Torch ACT-style policy on auto device | `uv run python examples/torch_policy.py` (`--bench 200` for inference only) | no |
| Interactive viewer, static scene (drag joints, inspect) | `make view` | yes |
| Interactive viewer while a policy runs | `make view-policy` | yes |
| Device report | `make gpu-info` | no |

**Headless vs interactive.** Offscreen rendering (camera frames, videos) uses `MUJOCO_GL=egl`,
the package default. `MUJOCO_GL=osmesa` is a pure-software fallback.
For a live window, set `NORI_VIEWER=1`. Any `mujoco_session()` then opens MuJoCo's passive
viewer (GLFW) and syncs it at about 60 Hz. This covers the unchanged SDK example too:
`NORI_VIEWER=1 MUJOCO_GL=glfw uv run python examples/sim_pick_place.py`.
You can also pass `mujoco_session(..., viewer=True)`.

## Using the sim from SDK code

```python
import asyncio
from nori_sdk.motion import JogBuilder
from nori_a3_mujoco import MujocoRobot, mujoco_session

async def main():
    async with mujoco_session(MujocoRobot()) as robot:      # realtime=True, video=True
        info = await robot.wait_ready()
        await robot.jog(JogBuilder(info.descriptor).base(linear=0.4).build(), duration=1.0)
        print(robot.telemetry.state.get("x.vel"), robot.telemetry.state.get("lift.pos"))
        result = await robot.action({"left_arm_gripper.pos": 30.0}, wait=True)
        frames = await robot.snapshot()                      # per-role camera images
        robot.estop()
asyncio.run(main())
```

What the session does, the same way the mock does:
- **Actions** (`robot.action({...})`, SDK units) go through the accepted → active →
  done/clamped/timeout/blocked lifecycle. The terminal state comes from *physics*: joints
  within 2 units and settled, otherwise `clamped` or `timeout`.
- **Jog** integrates the SDK's jog rates into targets. The base is driven by real wheel
  velocity actuators and casters.
- **pose()** (`pose_targets` capability, frame `base_footprint`) is solved with damped
  least-squares IK on the model. It returns `no_ik_solution` when the target is unreachable.
- **estop** latches: the robot freezes at its measured state, and pending actions finish as
  `blocked/estop_latched` until `reset_latch`.
- **Watchdog** (SDK profiles, WAN default 300/1000 ms): when commands go silent the robot goes
  to `safe_hold`, freezes, and pending actions finish as `timeout/watchdog`.
- **Telemetry** is measured state with LeRobot names (`left_arm_elbow_pitch.pos`, `lift.pos`
  in mm, `x.vel`, `theta.vel`, ...). IMU comes from sim sensors. Lidar is ray-cast.
- **Video**: a 2×2 composite of the 4 SDK camera roles (`left_wrist`, `right_wrist`,
  `overhead`, `front`) with the SDK's `CameraLayout`, so `snapshot()`/`frames()` crop per role.

## Policy runner

```python
from nori_a3_mujoco.policy import run_policy, STOP

def policy(obs):                       # obs.state, obs.images[role], obs.safety, obs.watchdog
    if obs.t > 5: return STOP          # desc = robot.info.descriptor
    return {"action": {"left_arm_gripper.pos": 100.0},
            "jog": JogBuilder(desc).base(angular=0.2).build()}  # or "pose": {"left": [x, y, z]}

report = await run_policy(robot, policy, hz=10, cameras=["front"], duration=10)
print(report)   # steps, achieved_hz, overruns, latency, safe_hold_ticks, reason, ...
```

- It works with any SDK session: `mujoco_session()`, the SDK's `mock_session()`, or a real
  `RemoteTeleop`. It only uses public SDK calls (`action`, `jog`, `pose`, `estop`, `frames`).
- **Fixed rate** with overrun accounting. `clock=` lets ticks follow sim time instead of wall
  time, which is useful when the sim runs slower than realtime (software GL).
- **Watchdog-honest keep-alive.** A slow policy is kept alive at 20 Hz with its last jog, but
  only while its last answer is fresh (`stale_after`, default 0.8× the watchdog warn time). A
  hung policy is *deliberately* allowed to trip the robot's watchdog (`safe_hold`). The runner
  never masks it.
- An exception in the policy triggers an **estop** (`estop_on_error`). A latched estop ends the
  run (`stop_on_estop`). The runner sends a zero jog on exit.
- Policies can be sync or `async`. A sync policy runs on the event loop, which blocks the
  sim pump. Put heavy compute in an `async` policy that uses `asyncio.to_thread`, as
  `ChunkedTorchPolicy` does.

**Scripted example** (`examples/scripted_policy.py`): drive until the red cube fills enough of
the *front camera*, then reach beside it with `pose()`, close the gripper, raise the lift, back
off. The same policy object runs with `--backend mock`, where a timer replaces the camera.
It does **not** grasp the cube (see below).

**Gym** (`nori_a3_mujoco.gym_env.NoriA3Env`): a 19-D action (16 joints, lift mm, base
linear/angular) and a Dict observation (`state` plus `images/<role>`). It steps the sim
synchronously through the same protocol handler, so the watchdog/estop rules still apply.
`reward_fn` is a hook; there's no task reward built in.

## Intel GPU (Arc B580) for policy inference

Same pattern as `decision-robot-control` / `act-mujoco-manipulation`:

- **Install.** `make setup-xpu` pulls `torch==2.9.1+xpu` plus `pytorch-triton-xpu==3.5.0`
  from `https://download.pytorch.org/whl/xpu`. The `cpu`/`xpu` extras conflict in uv, so the
  lock holds both. The oneAPI runtime pieces (SYCL, compiler runtime, oneMKL) arrive as pip
  wheels. On Arch you only need the GPU driver stack:
  `intel-compute-runtime level-zero-loader`. A full oneAPI Base Toolkit is not required.
- **Device selection** (`nori_a3_mujoco.devices.torch_device()`): `NORI_DEVICE` (`auto` |
  `xpu` | `xpu:1` | `cuda` | `cpu`). Otherwise it picks cuda → xpu → cpu. If you force `xpu`
  or `cuda` and it isn't available, it logs a warning and falls back to CPU.
  `ChunkedTorchPolicy(device=None)` uses this.
- **IPEX** is optional (`NORI_IPEX=1` runs `ipex.optimize()` on XPU models when
  `intel-extension-for-pytorch` is importable). Native `torch.xpu` is the main path. IPEX is
  not in the lock because its XPU builds must match the torch version exactly. Install the
  matching build from Intel's index yourself if you want it.
- **ONNX Runtime** has no XPU provider. `devices.onnx_providers()` returns CUDA → OpenVINO
  (`device_type: GPU`) → CPU, for whatever is installed.
- If the iGPU is also visible, pin the Arc card with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
- Check with `make gpu-info` and `NORI_DEVICE=xpu uv run python examples/torch_policy.py --bench 200`.
  The bench warms up first (XPU kernel JIT) and times forward plus device→host copy.

Caveat: the XPU path was **not run on an Intel GPU** while building this. The dev box had no
GPU, so only the CPU fallback and the lock resolution were tested. Physics (MuJoCo) is CPU-only.
Rendering uses the GPU through EGL/mesa.

## Faithful vs approximate

**Taken from Nori's description (v0.1.1) or the SDK, as-is**
- Kinematic tree, joint axes, limits, visual meshes and camera mount frames, from the URDF
  via xacro.
- The 3 mimic joints (lift middle stage, 2 gripper idler fingers) become MuJoCo joint
  equalities. Tests check that the lift middle stage moves at 0.5× the lift
  (measured 0.499) and that the idler finger counter-rotates.
- Actuator names, joint names and units, camera roles, telemetry keys, action lifecycle, estop
  latch and watchdog profiles follow the SDK's `A3_DESCRIPTOR` and `MockRobot`.
- `examples/sim_pick_place.py` is the SDK's `mock_pick_place.py` byte-for-byte except two
  lines. A test enforces this.

**Approximate or assumed (treat as placeholders)**
- **Mass/inertia.** Upstream masses are measured but sum to about 18.3 kg against a 20.5 kg
  real robot (about 9% light, per upstream NOTICE). They are not rescaled here. Inertias are
  upstream approximations, with `balanceinertia` enabled.
- **Servo strength and gains.** These are position actuators with kp=100 and kv=2. Per-joint
  torque limits (9.3/9.3/4.9/9.3/4.9/2.9/2.9/2.9 N·m from shoulder to gripper) are a
  guessed servo-grade allocation. Arm targets are slew-limited to 2 rad/s. The lift is a
  stiff position servo (kp 4e4, 400 N). Lift jog runs at 50 mm/s, matching the SDK mock;
  the docs say 30 mm/s.
- **Unit calibration.** SDK `range_m100_100` units (and gripper 0–100) map *linearly* onto the
  URDF joint limits. Gripper 0 = closed and 100 = open. The real robot's calibration offsets
  are unknown.
- **Base.** Wheel radius 0.0762 m and track 0.30 m. Max speeds 0.5 m/s and 1.5 rad/s are
  assumptions. Wheels are velocity servos (6 N·m). The URDF's fixed casters are modelled as
  frictionless contacts. Forward speed reaches about 0.39 of a commanded 0.4 m/s.
  `x.vel`/`theta.vel` telemetry is measured and normalised by those max speeds.
- **Lift range.** The descriptor allows 0–720 mm. The model clamps at the 700 mm operating
  ceiling from the upstream NOTICE.
- **Cameras.** Mount poses are upstream (and marked unverified upstream). FOV (58°) and
  resolution are guesses. Using the URDF optical frames as given, the `left_wrist` image comes
  out upside-down relative to `right_wrist`. This was kept as-is rather than "fixed".
- **Contacts.** Collision uses upstream primitive shapes. Robot self-collision is off.
- **Not simulated** (mock behaviour inherited from `MockRobot`): navigation goals, recording /
  episodes, `policy_stream`.

### Grasping
The sim does **not** pick objects up. With the upstream collision primitives, the closed
fingers don't make contact with a cube placed at the TCP site, and the TCP site may not be
the real grasp point. The demos show reach plus gripper open/close, not a pick. Fixing this
needs better finger collision geometry (and probably a calibrated TCP). That's the first
thing to improve before any grasp-learning work.

### Performance
On a GPU with EGL, rendering is cheap. On the software-GL dev box it was about 30 ms per
320×240 camera tile, and video recording ran about 6× slower than realtime. Physics at 500 Hz
(dt 0.002) is not the bottleneck.

## Regenerating the model

```bash
git clone https://github.com/Nori-Robotics/nori_description ../nori_description  # (tag v0.1.1)
make build-model DESCRIPTION=../nori_description
```
`scripts/build_model.py` expands the xacro (substituting `$(find nori_description)`, no ROS
needed), loads the URDF with `mujoco.MjSpec`, and edits the spec: free joint, actuators,
cameras on the optical frames, TCP/IMU/lidar sites and sensors, contact filtering, damping and
armature. It then writes `assets/a3.xml` and the meshes. Don't edit `a3.xml` by hand.

## License

- **Code** (`src/`, `scripts/`, `examples/`, `tests/`): Apache-2.0, see `LICENSE`.
  `examples/sim_pick_place.py` is derived from the Apache-2.0 Nori SDK example
  (© Nori Robotics).
- **Robot model** (`src/nori_a3_mujoco/assets/a3.xml`, `nori.urdf`, `meshes/`): a converted
  derivative of [Nori-Robotics/nori_description](https://github.com/Nori-Robotics/nori_description)
  v0.1.1, © Nori Robotics Inc., licensed **CC BY-NC-SA 4.0** (`assets/UPSTREAM_LICENSE`,
  `assets/UPSTREAM_NOTICE`). The converted model carries the same licence: attribution,
  **non-commercial**, share-alike. Nori describes the description as "source available", not
  open source, and so does this repo.
