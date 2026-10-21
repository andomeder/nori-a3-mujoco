"""The converted MJCF: structure, names, mimic constraints."""

import mujoco
import numpy as np
import pytest
from nori_sdk.mock import A3_DESCRIPTOR

from nori_a3_mujoco import ROBOT_XML, SCENE_XML, MujocoRobot


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def test_robot_xml_compiles_standalone():
    m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    assert m.nu == 19


def test_actuators_match_descriptor(model):
    names = {model.actuator(i).name for i in range(model.nu)}
    for key in A3_DESCRIPTOR["joints"]:
        assert key.removesuffix(".pos") in names
    assert {"lift", "left_wheel", "right_wheel"} <= names


def test_camera_roles_present(model):
    for role in A3_DESCRIPTOR["cameras"]:
        assert model.camera(role).id >= 0


def test_three_mimic_equalities(model):
    pairs = set()
    for i in range(model.neq):
        assert model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        j1 = model.joint(model.eq_obj1id[i]).name
        j2 = model.joint(model.eq_obj2id[i]).name
        pairs.add((j1, j2, round(float(model.eq_data[i][1]), 3)))
    assert pairs == {
        ("lift_middle_joint", "lift_extension_joint", 0.5),
        ("left_gripper_idler_joint", "left_gripper_joint", -1.0),
        ("right_gripper_idler_joint", "right_gripper_joint", -1.0),
    }


def test_robot_mass_matches_description(model):
    # NOTICE: links sum to ~18.3 kg (real robot ~20.5 kg); we do not rescale.
    assert model.body_subtreemass[model.body("base_footprint").id] == pytest.approx(18.34, abs=0.05)


def test_lift_middle_stage_moves_at_half_rate():
    bot = MujocoRobot(lift_mm=100.0)
    d = bot.data
    for target in (600.0, 300.0):
        bot.pose["lift.pos"] = target
        for _ in range(300):
            bot.handle({"type": "control", "jog": {}})
            bot.step(0.01)
        top = d.joint("lift_extension_joint").qpos[0]
        mid = d.joint("lift_middle_joint").qpos[0]
        assert top == pytest.approx(target / 1000.0, abs=0.003)
        assert mid / top == pytest.approx(0.5, abs=0.01)


def test_idler_finger_counter_rotates():
    bot = MujocoRobot()
    bot.pose["left_arm_gripper.pos"] = 60.0
    for _ in range(100):
        bot.handle({"type": "control", "jog": {}})
        bot.step(0.01)
    drv = bot.data.joint("left_gripper_joint").qpos[0]
    idl = bot.data.joint("left_gripper_idler_joint").qpos[0]
    assert drv == pytest.approx(0.6 * np.pi / 2, abs=0.02)
    assert idl == pytest.approx(-drv, abs=0.01)
