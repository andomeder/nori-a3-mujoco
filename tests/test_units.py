import pytest

from nori_a3_mujoco.units import JointMap, wheel_speeds


@pytest.mark.parametrize("kind,lo,hi,wire", [
    ("arm", -1.57, 1.57, -100.0), ("arm", -1.57, 1.57, 37.5), ("gripper", 0.0, 1.57, 30.0),
    ("lift", 0.0, 0.7, 350.0)])
def test_roundtrip(kind, lo, hi, wire):
    jm = JointMap("k", "j", "a", lo, hi, kind)
    assert jm.to_wire(jm.to_sim(wire)) == pytest.approx(wire)


def test_clamps_to_joint_limits():
    lift = JointMap("lift.pos", "j", "a", 0.0, 0.7, "lift")
    assert lift.to_sim(720.0) == 0.7  # descriptor range is 0-720 mm; joint ceiling is 700


def test_wheel_speeds_sign_convention():
    l, r = wheel_speeds(1.0, 0.0)
    assert l == r > 0
    l, r = wheel_speeds(0.0, 1.0)  # +angular = turn LEFT (REP-103): right wheel faster
    assert r > 0 > l
