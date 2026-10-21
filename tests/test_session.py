"""The SDK's RemoteTeleop against the sim: same verbs, physical outcomes."""

import asyncio

import pytest
from nori_sdk.motion import JogBuilder

from nori_a3_mujoco import MujocoRobot, mujoco_session

pytestmark = pytest.mark.asyncio


async def test_ready_and_descriptor():
    async with mujoco_session(MujocoRobot(), video=False) as robot:
        info = await robot.wait_ready()
        assert info.norm_mode == "range_m100_100"
        assert "lift" in info.descriptor.aux
        assert len(info.descriptor.joints) == 16
        assert robot.camera_layout.tiles == ["left_wrist", "right_wrist", "overhead", "front"]


async def test_jog_drives_the_base():
    bot = MujocoRobot()
    async with mujoco_session(bot, video=False) as robot:
        info = await robot.wait_ready()
        x0 = bot.base_pose()[0]
        await robot.jog(JogBuilder(info.descriptor).base(linear=0.5).build(), duration=1.0)
        assert bot.base_pose()[0] - x0 > 0.1
        await robot.hold(500)
        assert abs(robot.telemetry.state["x.vel"]) < 0.05


async def test_action_waits_for_physics():
    bot = MujocoRobot()
    async with mujoco_session(bot, video=False) as robot:
        await robot.wait_ready()
        res = await robot.action({"left_arm_elbow_pitch.pos": 50.0}, wait=True)
        assert res.state == "done"
        assert bot.measured("left_arm_elbow_pitch.pos") == pytest.approx(50.0, abs=2.0)
        res = await robot.action({"left_arm_elbow_pitch.pos": 150.0}, wait=True)
        assert res.state == "clamped"  # out of range: clamped robot-side, never rejected
        res = await robot.action({"nonexistent.pos": 1.0}, wait=True)
        assert res.state == "blocked" and res.reason.startswith("unknown_joint")


async def test_estop_latches_and_blocks():
    async with mujoco_session(MujocoRobot(), video=False) as robot:
        await robot.wait_ready()
        robot.estop()
        await robot.hold(200)
        assert robot.telemetry.safety == "latched"
        res = await robot.action({"left_arm_gripper.pos": 50.0}, wait=True)
        assert res.state == "blocked"
        robot.reset_latch()
        await robot.hold(200)
        assert robot.telemetry.safety == "ok"


async def test_watchdog_silence_safe_holds():
    bot = MujocoRobot()
    async with mujoco_session(bot, video=False) as robot:
        info = await robot.wait_ready()
        await robot.jog(JogBuilder(info.descriptor).base(linear=0.5).build())  # ONE frame
        await asyncio.sleep(1.4)  # > WAN t_stop (1000 ms)
        assert robot.telemetry.safety == "safe_hold"
        assert abs(bot.base_velocity()[0]) < 0.05


async def test_pose_uses_ik():
    bot = MujocoRobot(lift_mm=300.0)
    async with mujoco_session(bot, video=False) as robot:
        await robot.wait_ready()
        res = await robot.pose("left", [0.25, 0.15, 0.70], wait=True)
        assert res.state == "done"
        x, y, z = bot.data.site("left_tcp").xpos
        bx, by, _ = bot.base_pose()
        assert (x - bx, y - by, z) == pytest.approx((0.25, 0.15, 0.70), abs=0.02)
        res = await robot.pose("left", [1.5, 0.15, 0.70], wait=True)
        assert res.state == "blocked" and res.reason == "no_ik_solution"


async def test_snapshot_per_role():
    async with mujoco_session(MujocoRobot()) as robot:
        await robot.wait_ready()
        img = await robot.snapshot(role="front", settle=0.0)
        assert img.shape == (240, 320, 3)
        assert img.mean() > 5  # not a black tile
        png = await robot.snapshot_png(role="overhead", settle=0.0)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


async def test_sensor_streams_from_sim():
    async with mujoco_session(MujocoRobot(), video=False) as robot:
        await robot.wait_ready()
        await robot.configure_sensor_streams(lidar_hz=10, imu_hz=20)
        await robot.hold(400)
        assert robot.imu_sample.linear_acceleration_m_s2[2] == pytest.approx(9.81, abs=0.2)
        assert min(robot.lidar_scan.ranges_m) > 0.0
