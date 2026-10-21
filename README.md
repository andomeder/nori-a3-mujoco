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
| Grasp calibration, camera-based cube/marker localisation, pick-and-place routine | `grasp.py`, `perception.py`, `pick_place.py` |
| Examples | `examples/sim_pick_place.py` (real pick-and-place over the SDK API), `sdk_mock_pick_place.py` (the SDK's `mock_pick_place.py`, 2 lines changed), `scripted_policy.py`, `torch_policy.py` |

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
make test             # 40 tests, headless (~1.5 min)
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
| Pick-and-place over the SDK API | `make example` | no (EGL) |
| The SDK's own example, session line swapped | `make sdk-example` | no |
| Scripted policy through `run_policy` | `make policy` | no |
| Same, recorded to `videos/scripted_policy.mp4` | `make policy-demo` | no |
| Pick-and-place video and stills → `videos/` | `make demo` | no |
| Torch ACT-style policy on auto device | `uv run python examples/torch_policy.py` (`--bench 200` for inference only) | no |
| Interactive viewer, static scene (drag joints, inspect) | `make view` | yes |
| Interactive viewer while a policy runs | `make view-policy` | yes |
| Device report | `make gpu-info` | no |

**Headless vs interactive.** Offscreen rendering (camera frames, videos) uses `MUJOCO_GL=egl`,
the package default. `MUJOCO_GL=osmesa` is a pure-software fallback.
For a live window, set `NORI_VIEWER=1`. Any `mujoco_session()` then opens MuJoCo's passive
viewer (GLFW) and syncs it at about 60 Hz. This covers the unchanged SDK example too:
`NORI_VIEWER=1 MUJOCO_GL=glfw uv run python examples/sdk_mock_pick_place.py`.
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

**Scripted example** (`examples/scripted_policy.py`): a phase machine ticked by `run_policy`
at 10 Hz that does the full pick-and-place: lift to table height, drive until the *overhead
camera* puts the cube in reach, locate cube and marker, then pre-grasp → approach → close →
lift → carry → lower → release → retreat, all through `pose()`/`action`/jog outputs. The
same policy object runs with `--backend mock` (a timer and the nominal layout replace the
camera; the mock has no physics, so nothing is actually picked there).

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
- `examples/sdk_mock_pick_place.py` is the SDK's `mock_pick_place.py` byte-for-byte except
  two lines. A test enforces this.

**Approximate or assumed (treat as placeholders)**
- **Mass/inertia.** Upstream masses are measured but sum to about 18.3 kg against a 20.5 kg
  real robot (about 9% light, per upstream NOTICE). They are not rescaled here. Inertias are
  upstream approximations, with `balanceinertia` enabled.
- **Servo strength and gains.** These are position actuators with kp=300 and kv=4 (a few mm
  of gravity sag at full reach). Per-joint torque limits (9.3/9.3/4.9/9.3/4.9/2.9/2.9/2.9
  N·m from shoulder to gripper) are a guessed servo-grade allocation. Arm targets move
  *coordinated* (all 7 joints of an arm start and stop together, slowest at 2 rad/s), which
  keeps short `pose()` moves close to straight lines. `pose()` reports `done` only within
  0.5 units of the IK solution, `action` within 2 units. The lift is a
  stiff position servo (kp 4e4, 400 N). Lift jog runs at 50 mm/s, matching the SDK mock;
  the docs say 30 mm/s.
- **Unit calibration.** SDK `range_m100_100` units (and gripper 0–100) map *linearly* onto the
  URDF joint limits. Gripper 0 = closed and 100 = open. The real robot's calibration offsets
  are unknown.
- **Base.** Wheel radius 0.0762 m and track 0.30 m. Max speeds 0.5 m/s and 1.5 rad/s are
  assumptions. Wheels are velocity servos (6 N·m). The URDF's fixed casters are modelled as
  frictionless contacts, so when the base is idle the drive holds wheel position (a brake);
  without it, arm reaction torques yawed the base by up to 0.1 rad during a reach. Forward
  speed reaches about 0.39 of a commanded 0.4 m/s.
  `x.vel`/`theta.vel` telemetry is measured and normalised by those max speeds.
- **Lift range.** The descriptor allows 0–720 mm. The model clamps at the 700 mm operating
  ceiling from the upstream NOTICE.
- **Cameras.** Mount poses are upstream (and marked unverified upstream). FOV (58°) and
  resolution are guesses. Using the URDF optical frames as given, the `left_wrist` image comes
  out upside-down relative to `right_wrist`. This was kept as-is rather than "fixed".
- **Contacts.** Collision uses upstream primitive shapes (hidden, group 3), except the claw
  fingers (below). Robot self-collision is off.
- **Not simulated** (mock behaviour inherited from `MockRobot`): navigation goals, recording /
  episodes, `policy_stream`.

### Grasping
The sim picks the cube up for real: it is lifted ~8 cm off the table, carried 15 cm, and set
down on the marker by friction alone (no welds, no attachment constraints). To get there:

- **Finger collision.** Upstream gives each claw finger one solid box that fills the whole
  jaw, so nothing could sit between the fingers. `build_model.py` replaces it with three
  boxes per finger that trace the C-shaped jaw of the visual mesh (measured by ray-casting
  the STL): an outer arm, a flat inner pad, and an inward tip hook. Pads use condim 4
  (torsional friction), friction 1.5, solref 0.004, a 1 mm margin, and priority over the cube.
  Closed, the pads leave a 37 mm gap.
- **Cube.** 40 mm, 50 g (a wooden block), condim 4. It is squeezed at about 0.06 rad of finger
  opening, where the 2.9 N·m gripper servo saturates (~50 N per pad).
- **Grasp point / TCP.** The URDF TCP sits at the closed fingertips. Objects are held in the
  jaw cavity, 24 mm further back along the approach (14 mm off-centre across the finger
  plate). That is the `{side}_grasp` site, with the constant in `grasp.GRASP_IN_TCP`.
  `pose()` keeps the SDK meaning (URDF TCP); `grasp.tcp_for_grasp()` converts.
- **Grasp.** A horizontal side grasp with the left claw (approach along base +x, the claw's
  hub and camera on top), at the cube's top-face height. The wrist housing reaches 37 mm
  below the grasp point and would otherwise drag on the table.
- **Perception.** The cube and the green marker come from the *overhead camera*: colour blob
  → pixel ray (intrinsics from the MJCF fovy, extrinsics from the robot's own kinematics at
  the measured lift height) → plane at the known table height. Error against ground truth
  is ≤ 3 mm in x and ≤ 8 mm in y. The only scene prior is the table height.

Verified from sim state (`tests/test_grasp.py`, 3 start poses including ±15–20° cube yaw, and
6 more by hand):
- Peak height is 0.859–0.866 m, against 0.780 resting.
- During carry the cube stays within ~1.8 cm of the grasp point, constant to <1 mm.
- It ends upright at z = 0.780, 6–8 mm from the marker centre. The hooks pull the cube about
  8 mm into the jaw when closing, which shows up as that placement offset.

Limits: one grasp (left claw, side approach) in a reach window of roughly x 0.32–0.36 m,
y 0.08–0.20 m ahead of the base at lift 300 mm. The pad geometry, friction and gripper force
are ours, not measured from the hardware.

### Rendering
`scene.xml` asks for 8× MSAA (`offsamples`) and a 4096 shadow map clipped to the work area.
It uses a tight near/far range (`znear 0.02`, `zfar 15`, relative to the model extent) and a
straight-down key light: tilted directional shadows showed acne rings on the floor. Table legs
overlap the top by 2 mm and stop 1 mm above the floor, so no faces are coplanar (no z-fighting).
MSAA is driver-dependent; software Mesa ignores it. So `VideoRecorder` also supersamples 2×
(renders 4× the pixels and box-filters down), which works everywhere. Videos are H.264 crf 14
yuv420p. The main view tracks the base in a wide shot. When manipulation starts,
`VideoRecorder.close_on(point)` blends over 1 s of sim time to a low close-up from across
the table, so the cube in the claw, the gap and its shadow below it, and the placement are all
visible.

### Performance
On a GPU with EGL, rendering is cheap. On the software-GL dev box it was about 0.6 s per
supersampled 1200×720 video frame, so recording ran roughly 10–15× slower than realtime. Physics at 500 Hz
(dt 0.002) is not the bottleneck. For this reason the example guards (`move_lift`,
`drive_to_standoff`) and the policy runner tick on the sim clock; the SDK's wall-clock
`pose()`/`action()` waits get a generous timeout in `sim_pick_place.py`.

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
