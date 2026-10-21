"""run_policy against the SDK's own MockRobot (no MuJoCo) and the sim."""

import asyncio

import pytest
from nori_sdk.mock import A3_DESCRIPTOR, MockRobot, mock_session
from nori_sdk.motion import JogBuilder

from nori_a3_mujoco.policy import STOP, run_policy

pytestmark = pytest.mark.asyncio


async def test_fixed_rate_and_actions_on_mock():
    async with mock_session(MockRobot(descriptor=A3_DESCRIPTOR)) as robot:
        info = await robot.wait_ready()
        seen = []

        def policy(obs):
            seen.append(obs.step)
            if obs.step == 0:
                return {"action": {"left_arm_gripper.pos": 40.0}}
            return {"jog": JogBuilder(info.descriptor).base(linear=0.3).build()}

        report = await run_policy(robot, policy, hz=20, max_steps=20)
        assert report.reason == "max_steps" and report.steps == 20
        assert 0.8 < report.duration_s < 1.6
        assert robot.telemetry.state["left_arm_gripper.pos"] == pytest.approx(40.0)
        await robot.hold(100)
        assert robot.telemetry.state["x.vel"] == 0.0  # zero jog sent on exit


async def test_slow_policy_is_kept_alive():
    """A 2 Hz policy is slower than the WAN t_stop's comfort zone only by design of the
    keep-alive: the robot must stay 'ok' and keep moving the base between ticks."""
    async with mock_session(MockRobot(descriptor=A3_DESCRIPTOR)) as robot:
        info = await robot.wait_ready()
        jog = JogBuilder(info.descriptor).base(linear=0.5).build()
        safety = []

        def policy(obs):
            safety.append(obs.safety)
            return {"jog": jog}

        await run_policy(robot, policy, hz=2, max_steps=4)
        assert set(safety[1:]) == {"ok"}


async def test_hung_policy_lets_watchdog_fire():
    async with mock_session(MockRobot(descriptor=A3_DESCRIPTOR)) as robot:
        info = await robot.wait_ready()
        jog = JogBuilder(info.descriptor).base(linear=0.5).build()
        safety = []

        async def policy(obs):
            safety.append(obs.safety)
            if obs.step == 1:
                await asyncio.sleep(1.5)  # "inference" hangs past t_stop
            return {"jog": jog}

        await run_policy(robot, policy, hz=10, max_steps=3)
        assert "safe_hold" in safety  # the keep-alive did NOT cover for the hung policy


async def test_error_latches_estop():
    async with mock_session(MockRobot(descriptor=A3_DESCRIPTOR)) as robot:
        await robot.wait_ready()

        def policy(obs):
            raise RuntimeError("model exploded")

        with pytest.raises(RuntimeError):
            await run_policy(robot, policy, hz=10, max_steps=5)
        await robot.hold(100)
        assert robot.telemetry.safety == "latched"


async def test_stop_and_estop_end_the_run():
    async with mock_session(MockRobot(descriptor=A3_DESCRIPTOR)) as robot:
        await robot.wait_ready()
        report = await run_policy(robot, lambda obs: STOP if obs.step == 2 else None, hz=20)
        assert report.reason == "stop" and report.steps == 2
        robot.estop()
        await robot.hold(100)
        report = await run_policy(robot, lambda obs: None, hz=20, max_steps=50)
        assert report.reason == "estop" and report.steps == 0


async def test_sim_with_camera_observations_and_sim_clock():
    from nori_a3_mujoco import MujocoRobot, mujoco_session

    bot = MujocoRobot()
    async with mujoco_session(bot) as robot:
        info = await robot.wait_ready()
        shapes = []

        def policy(obs):
            if "front" in obs.images:
                shapes.append(obs.images["front"].shape)
            return {"jog": JogBuilder(info.descriptor).base(linear=0.5).build()}

        x0 = bot.base_pose()[0]
        report = await run_policy(robot, policy, hz=10, duration=1.0, cameras=["front"],
                                  clock=lambda: bot.sim_time)
        assert report.duration_s == pytest.approx(1.0, abs=0.15)  # SIM seconds
        assert shapes and shapes[-1] == (240, 320, 3)
        assert bot.base_pose()[0] - x0 > 0.1
