"""The grasp is real: the cube leaves the table, rides in the jaw, and lands on the marker.

Verified from sim state (not from action statuses): cube height, cube position in the claw's
grasp frame during transport, final resting pose. Several start positions / yaws.
"""

import math

import mujoco
import numpy as np
import pytest

from nori_a3_mujoco import MujocoRobot, mujoco_session
from nori_a3_mujoco.grasp import GRASP_IN_TCP, grasp_in_tcp_from_model
from nori_a3_mujoco.perception import SceneLocator
from nori_a3_mujoco.pick_place import (LIFT_MM, drive_to_standoff, locate, move_lift,
                                       pick_and_place)

TABLE_CONTACT_Z = 0.78   # cube centre when resting on the table


def test_grasp_offset_constant_matches_model():
    bot = MujocoRobot()
    for side in ("left", "right"):
        got = grasp_in_tcp_from_model(bot.model, bot.data, side)
        assert np.allclose(got, GRASP_IN_TCP[side], atol=5e-4)


def test_finger_pads_replace_the_solid_jaw_boxes():
    m = MujocoRobot().model
    for side in ("left", "right"):
        for finger in ("gripper", "gripper_idler"):
            names = {m.geom(g).name for g in range(m.ngeom)
                     if m.geom_bodyid[g] == m.body(f"{side}_{finger}_link").id
                     and m.geom_contype[g]}
            assert names == {f"{side}_{finger}_{p}" for p in ("arm", "pad", "hook")}
            pad = m.geom(f"{side}_{finger}_pad")
            assert pad.condim[0] == 4 and pad.friction[0] >= 1.0


def _place_cube(bot, x, y, yaw_deg=0.0):
    adr = bot.model.joint("cube_free").qposadr[0]
    bot.data.qpos[adr:adr + 3] = [x, y, TABLE_CONTACT_Z + 0.001]
    t = math.radians(yaw_deg)
    bot.data.qpos[adr + 3:adr + 7] = [math.cos(t / 2), 0, 0, math.sin(t / 2)]
    mujoco.mj_forward(bot.model, bot.data)


def test_overhead_camera_locates_the_cube():
    bot = MujocoRobot(lift_mm=LIFT_MM)
    loc = SceneLocator("overhead")
    r = mujoco.Renderer(bot.model, 240, 320)
    try:
        base = bot.model.body("base_footprint").id
        for bx, cy in ((0.67, 0.12), (0.55, 0.16), (0.45, 0.09)):
            adr = bot.model.joint("base_free").qposadr[0]
            bot.data.qpos[adr] = bx
            _place_cube(bot, 1.0, cy)
            r.update_scene(bot.data, "overhead")
            est = loc.cube(r.render(), bot.measured_state())
            rb = bot.data.xmat[base].reshape(3, 3)
            true = rb.T @ (bot.data.body("cube").xpos - bot.data.xpos[base])
            assert est is not None
            assert abs(est[0] - true[0]) < 0.006 and abs(est[1] - true[1]) < 0.010
    finally:
        r.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("x,y,yaw", [(1.0, 0.12, 0.0), (0.985, 0.16, 15.0),
                                     (1.015, 0.10, -20.0)])
async def test_pick_and_place_from_sim_state(x, y, yaw):
    bot = MujocoRobot()
    _place_cube(bot, x, y, yaw)
    m, d = bot.model, bot.data
    grasp = m.site("left_grasp").id
    phase = ["setup"]
    trace = []

    def record(b):
        c = d.body("cube").xpos.copy()
        r = d.site_xmat[grasp].reshape(3, 3)
        trace.append((phase[0], c, r.T @ (c - d.site_xpos[grasp])))

    bot.step_callbacks.append(record)
    async with mujoco_session(bot) as robot:
        info = await robot.wait_ready()
        locator = SceneLocator("overhead")
        await move_lift(robot, info.descriptor, LIFT_MM)
        cube = await drive_to_standoff(robot, info.descriptor, locator)
        place = await locate(robot, locator, what="marker")
        assert cube is not None and place is not None
        report = await pick_and_place(robot, cube, place,
                                      on_step=lambda s: phase.__setitem__(0, s.name))
        await robot.hold(500)
    assert report.ok, report

    def rows(name):
        return [t for t in trace if t[0] == name]

    # 1) lifted clear of the table: >= 5 cm above its resting height
    lifted = rows("lift") + rows("carry")
    assert max(c[2] for _, c, _ in lifted) > TABLE_CONTACT_Z + 0.05
    # 2) during transport it stays in the jaw: never more than 2 cm from the grasp point,
    #    and always well above the table
    carry = rows("carry")
    assert carry
    assert max(np.linalg.norm(j) for _, _, j in carry) < 0.02
    assert min(c[2] for _, c, _ in carry) > TABLE_CONTACT_Z + 0.04
    # 3) it ends resting on the table, upright, on the marker
    final = d.body("cube").xpos
    marker = d.geom("place_marker").xpos
    assert final[2] == pytest.approx(TABLE_CONTACT_Z, abs=0.003)
    assert np.hypot(*(final[:2] - marker[:2])) < 0.015
    up = d.body("cube").xmat.reshape(3, 3)[:, 2]
    assert abs(up[2]) > 0.99
