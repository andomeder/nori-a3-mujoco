import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from nori_a3_mujoco.gym_env import NoriA3Env  # noqa: E402


def test_reset_step_spaces():
    env = NoriA3Env(cameras=("front",), max_steps=5)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    action = np.zeros(19, dtype=np.float32)
    action[16] = 500.0      # lift target
    action[17] = 0.5        # drive
    action[15] = 100.0      # right gripper open
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(action)
    assert trunc and not term
    assert obs["state"][17] > 0.2              # x.vel (normalized)
    assert obs["state"][16] > 360.0            # lift rising toward 500
    assert obs["images/front"].shape == (240, 320, 3)
    env.estop()
    _, _, term, _, _ = env.step(action)
    assert term
    env.close()
