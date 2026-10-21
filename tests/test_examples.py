import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sdk_example_runs_against_the_sim():
    out = subprocess.run([sys.executable, str(ROOT / "examples" / "sim_pick_place.py")],
                         capture_output=True, text=True, timeout=120,
                         env={**os.environ, "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")})
    assert out.returncode == 0, out.stderr
    assert "gripper -> 30: done (reached=True)" in out.stdout
    assert "kept 1 episode(s)" in out.stdout
    assert "estopped -- latched" in out.stdout


def test_example_differs_from_sdk_only_in_session_lines():
    sdk = ROOT.parent / "nori-sdk-py" / "examples" / "mock_pick_place.py"
    if not sdk.exists():
        import pytest
        pytest.skip("nori-sdk-py checkout not beside this repo")
    ours = [ln for ln in (ROOT / "examples" / "sim_pick_place.py").read_text().splitlines()
            if not ln.startswith("# ")]
    theirs = sdk.read_text().splitlines()
    diff = [(a, b) for a, b in zip(ours, theirs) if a != b]
    assert len(ours) == len(theirs)
    assert len(diff) == 2 and all("<- changed" in a for a, _ in diff)
