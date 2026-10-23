# nori-a3-mujoco

Unofficial MuJoCo model of the Nori A3. It uses the session API from the [Nori Python SDK](https://github.com/Nori-Robotics/nori-sdk-py). Replace `mock_session()` with `mujoco_session(MujocoRobot())`. The rest of the SDK code stays.

Nori Robotics has not reviewed this. Kinematics come from [nori_description](https://github.com/Nori-Robotics/nori_description) v0.1.1. Mass, servo strength, and calibration are guesses. Read "What is approximate" before you trust a number.

```python
from nori_a3_mujoco import MujocoRobot, mujoco_session

async with mujoco_session(MujocoRobot()) as robot:
    ...
```

`mujoco_session` runs in realtime and records a video track. Pass `realtime=False` or `video=False` to turn either one off.

## Install on Arch

You need Python 3.11, 3.12, or 3.13. `.python-version` pins 3.12. Install [uv](https://docs.astral.sh/uv/).

```bash
sudo pacman -S --needed uv mesa ffmpeg
sudo pacman -S --needed intel-compute-runtime level-zero-loader
make setup-xpu
```

`make setup-xpu` installs `torch==2.9.1+xpu` from the PyTorch XPU index. The wheels include the oneAPI runtime. `make setup` installs the CPU torch build instead.

`make setup` copies `.env.example` to `.env`. The Makefile exports that file.

The pin is `nori-sdk` 1.1.0, from `nori-sdk-py` at `a8db25b`.

If you do not use uv, install `requirements/xpu.txt` and `requirements/dev.txt`, then run `pip install -e . --no-deps`.

## Run a demo

```bash
make test
make example
make demo
make view
make gpu-info
```

`make test` runs 40 tests with no window. The test that diffs `sdk_mock_pick_place.py` against the SDK skips unless `../nori-sdk-py` is checked out beside this repo.

`make example` runs the pick-and-place with no window. `make demo` writes `videos/demo.mp4` and stills. `make view` opens a GLFW window. Drag a joint with the mouse.

Offscreen frames use `MUJOCO_GL=egl`. Set `NORI_VIEWER=1` to attach the GLFW viewer to every session.

With Mesa 26.2.2 on an Arc B580, offscreen EGL used the Arc. `make demo` wrote 410 frames, 13.7 s of video, in 55 s of wall time. That is about 0.13 s per frame. Physics stays on the CPU. The examples time their waits on the sim clock, because a recording runs slower than the video.

## Drive it from SDK code

```python
from nori_sdk.motion import JogBuilder
from nori_a3_mujoco import MujocoRobot, mujoco_session

async with mujoco_session(MujocoRobot()) as robot:
    info = await robot.wait_ready()
    await robot.jog(JogBuilder(info.descriptor).base(linear=0.4).build(), duration=1.0)
    await robot.action({"left_arm_gripper.pos": 30.0}, wait=True)
    frames = await robot.snapshot()
    robot.estop()
```

`action` and `pose` finish from the physics. A joint that settles inside the tolerance ends as `done`. A joint that hits a limit ends as `clamped`. `estop` latches until `reset_latch`.

`run_policy(robot, fn, hz)` sends `action`, `jog`, and `pose` at a fixed rate. The same function accepts this sim, `mock_session()`, or a real `RemoteTeleop`. If `fn` stops answering, the runner lets the SDK watchdog trip. Return `STOP` to end the run.

`NoriA3Env` in `gym_env.py` wraps the same handler for Gymnasium. It has no task reward.

## Run a torch policy on an Arc

`NORI_DEVICE=auto` picks CUDA, then XPU, then CPU. Set `NORI_DEVICE` to `xpu`, `xpu:0`, `cuda`, or `cpu`. If that device is missing, the code warns and uses the CPU.

```bash
NORI_DEVICE=xpu uv run python examples/torch_policy.py --bench 200
```

The bench warms up, then times the forward pass and the copy back to the host. On an Arc B580 with torch 2.9.1+xpu, 200 calls took a median of 1.60 ms, 95th percentile 2.10 ms. The same wheel on CPU took a median of 5.87 ms, 95th percentile 7.04 ms.

If an integrated GPU is also visible, set `ONEAPI_DEVICE_SELECTOR=level_zero:0`.

## What is approximate

From the URDF and the SDK, unchanged:

- Kinematic tree, joint axes, limits, visual meshes, and camera mounts.
- Actuator names, units, camera roles, the estop latch, and the watchdog profiles.
- `examples/sdk_mock_pick_place.py` matches the SDK file except the import and the session constructor.

Guessed, and not measured on a robot:

- Link masses sum to about 18.3 kg. The robot is about 20.5 kg. This model does not rescale them.
- Arm joints are position servos with kp 300 and kv 4. The torque cap on each joint is a guessed split.
- SDK units map in a straight line onto the URDF limits. Gripper 0 is closed and 100 is open. The real calibration offsets are unknown.
- Wheel radius is 0.0762 m. Track is 0.30 m.
- The descriptor allows a 720 mm lift. The model stops at 700 mm.
- The left claw can pick the cube up by friction in one window. Put the cube about 32 to 36 cm forward of the base and 8 to 20 cm to the side. Set the lift to 300 mm. It lands 6 to 8 mm from the marker. The pad shape and the grip force are ours.
- The `left_wrist` picture is upside down beside `right_wrist`. It stays that way.

Navigation goals, episode recording, and `policy_stream` still take the mock path. This sim does not run them.

## Rebuild the model

```bash
git clone --branch v0.1.1 https://github.com/Nori-Robotics/nori_description ../nori_description
make build-model DESCRIPTION=../nori_description
```

`scripts/build_model.py` writes `assets/a3.xml` and the meshes. Do not edit `a3.xml` by hand.

## License

Code in `src/`, `scripts/`, `examples/`, and `tests/` is Apache-2.0. See `LICENSE`. `examples/sim_pick_place.py` comes from the Nori SDK example.

`assets/a3.xml` and the meshes come from nori_description v0.1.1, copyright Nori Robotics Inc., under CC BY-NC-SA 4.0. See `assets/UPSTREAM_LICENSE`. Non-commercial, and share-alike. Nori calls that description source-available, not open source.
