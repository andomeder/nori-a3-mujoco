"""Conversions between SDK wire units and MuJoCo joint units.

The SDK speaks `norm_mode = "range_m100_100"`: every arm joint is a number in [-100, 100]
across its calibrated range, grippers are [0, 100], the central lift is millimetres, and
base jog rates are normalized [-1, 1]. The real robot maps those through per-unit servo
calibration that is NOT published; this sim maps them LINEARLY onto the URDF joint limits.
That is the single biggest "approximate" in this package -- see README.
"""

from __future__ import annotations

from dataclasses import dataclass

# Base speed at a full-rate (1.0) jog. Not published for the A3; these are assumptions,
# picked to look like a small indoor differential-drive base. Change them if you know better.
MAX_LINEAR_MPS = 0.5
MAX_ANGULAR_RADPS = 1.5

# From the URDF (properties.xacro).
WHEEL_RADIUS_M = 0.0762
WHEEL_SEPARATION_M = 0.30
LIFT_COMMAND_CEILING_M = 0.70   # lift_extension_joint upper limit (operating ceiling)


@dataclass(frozen=True)
class JointMap:
    key: str           # SDK key, e.g. "left_arm_elbow_pitch.pos"
    joint: str         # MuJoCo joint name
    actuator: str      # MuJoCo actuator name
    lower: float       # rad (or m)
    upper: float
    kind: str          # "arm" | "gripper" | "lift"

    def to_sim(self, value: float) -> float:
        """Wire value -> joint position (rad / m), clamped to the joint range."""
        if self.kind == "lift":
            q = value / 1000.0
        elif self.kind == "gripper":
            q = self.lower + (value / 100.0) * (self.upper - self.lower)
        else:
            q = self.lower + (value + 100.0) / 200.0 * (self.upper - self.lower)
        return min(max(q, self.lower), self.upper)

    def to_wire(self, q: float) -> float:
        """Joint position -> wire value."""
        if self.kind == "lift":
            return q * 1000.0
        span = (self.upper - self.lower) or 1.0
        frac = (q - self.lower) / span
        if self.kind == "gripper":
            return frac * 100.0
        return frac * 200.0 - 100.0


def wheel_speeds(linear: float, angular: float) -> tuple[float, float]:
    """Normalized base rates -> (left, right) wheel angular velocity in rad/s."""
    v = max(-1.0, min(1.0, linear)) * MAX_LINEAR_MPS
    w = max(-1.0, min(1.0, angular)) * MAX_ANGULAR_RADPS
    half = WHEEL_SEPARATION_M / 2.0
    return (v - w * half) / WHEEL_RADIUS_M, (v + w * half) / WHEEL_RADIUS_M
