#!/usr/bin/env python3
"""Rebuild the MuJoCo model of the Nori A3 from the upstream robot description.

    python scripts/build_model.py [--description PATH] [--tag v0.1.1]

Pipeline (each step is plain and re-runnable):

1. Get `Nori-Robotics/nori_description` at a pinned tag (clone, or use --description).
2. Expand `urdf/nori.urdf.xacro` with the standalone `xacro` package. The upstream file
   resolves its mass table with `$(find nori_description)`, which needs ROS; we copy the
   description into a scratch dir and substitute the absolute path instead.
3. Rewrite `package://nori_description/meshes/...` to paths relative to a meshdir and add a
   `<mujoco><compiler .../></mujoco>` block so MuJoCo keeps visuals, keeps fixed frames
   (the camera optical frames are fixed links) and keeps mesh sub-directories.
4. Load with `mujoco.MjSpec.from_file` (MuJoCo's own URDF importer, which turns the three
   `<mimic>` joints into joint equality constraints) and then edit the spec:
   free joint on the base, collision filtering, wheel/caster friction, actuators named
   after the SDK descriptor, the four SDK camera roles, TCP / IMU / lidar sites and sensors.
5. Write `src/nori_a3_mujoco/assets/a3.xml` + meshes, and compile it once as a check.

The scene (floor, table, cube) lives in `assets/scene.xml`, which <include>s a3.xml.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "nori_a3_mujoco" / "assets"
UPSTREAM = "https://github.com/Nori-Robotics/nori_description"
DEFAULT_TAG = "v0.1.1"

SIDES = ("left", "right")
ARM_JOINTS = (
    "shoulder_pitch", "shoulder_roll", "bicep_yaw", "elbow_pitch",
    "forearm_yaw", "wrist_pitch", "wrist_roll", "gripper",
)
# Holding torque per joint, N*m. The A3 uses Feetech STS servos in three torque grades
# (95 / 50 / 30 kg*cm ~= 9.3 / 4.9 / 2.9 N*m) "matched to each joint's load"; WHICH grade
# sits on which joint is not published, so this allocation is our guess (heaviest at the
# shoulder, lightest at the wrist). The URDF's own effort="4.0" is a uniform placeholder.
ARM_FORCE = {
    "shoulder_pitch": 9.3, "shoulder_roll": 9.3, "bicep_yaw": 4.9, "elbow_pitch": 9.3,
    "forearm_yaw": 4.9, "wrist_pitch": 2.9, "wrist_roll": 2.9, "gripper": 2.9,
}
ARM_KP = 100.0         # N*m/rad. Servo-like stiff position loop; not identified from hardware.
ARM_KV = 2.0
ARM_ARMATURE = 0.01    # reflected rotor inertia stand-in; keeps light links stable at 2 ms
LIFT_KP = 40000.0      # N/m. Stiff position hold; the real column drive is not modelled
LIFT_FORCE = 400.0     # N. URDF says effort=1.0, which cannot lift ~8 kg; that is a ros2_control
                       # placeholder, so the sim uses a force that can actually hold the column.
WHEEL_KV = 2.0
WHEEL_FORCE = 6.0      # N*m per wheel
CAMERA_FOVY = 58.0     # deg. The U20CAM's FOV is not in the description; this is a guess.

ROBOT_CONTYPE, ROBOT_CONAFF = 2, 1   # robot never collides with itself (see README)
WORLD_CONTYPE, WORLD_CONAFF = 1, 2   # floor: collides with robot


def fetch(description: Path | None, tag: str, work: Path) -> Path:
    if description is not None:
        src = description
    else:
        src = work / "nori_description_src"
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", tag, UPSTREAM,
                        str(src)], check=True)
    dst = work / "nori_description"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
    for xacro_file in (dst / "urdf").glob("*.xacro"):
        text = xacro_file.read_text()
        if "$(find nori_description)" in text:
            xacro_file.write_text(text.replace("$(find nori_description)", str(dst)))
    return dst


def expand_xacro(desc: Path) -> str:
    import xacro  # standalone PyPI package

    doc = xacro.process_file(str(desc / "urdf" / "nori.urdf.xacro"))
    return doc.toprettyxml(indent="  ")


def prepare_urdf(urdf: str) -> str:
    urdf = urdf.replace("package://nori_description/meshes/", "")
    block = ('<mujoco><compiler meshdir="meshes" strippath="false" fusestatic="false" '
             'discardvisual="false" balanceinertia="true"/></mujoco>')
    return re.sub(r"(<robot[^>]*>)", r"\1\n" + block, urdf, count=1)


def _first(value):
    try:
        return value[0]
    except (TypeError, IndexError):
        return value


def _damping(value: float):
    """MuJoCo >= 3.5 stores joint damping as a 3-vector (polynomial terms); older takes a float."""
    probe = mujoco.MjSpec().worldbody.add_body().add_joint()
    try:
        probe.damping = value
        return value
    except TypeError:
        return [value, 0.0, 0.0]


def edit_spec(spec: mujoco.MjSpec) -> None:
    spec.modelname = "nori_a3"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compiler.degree = False

    base = spec.body("base_footprint")
    base.add_freejoint(name="base_free")
    base.pos = [0, 0, 0.001]

    # --- collision filtering and friction -----------------------------------------------
    for geom in spec.geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            continue  # visual
        geom.contype, geom.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFF
        body = geom.parent.name
        if body.endswith("_wheel_link") and "caster" not in body:
            geom.friction = [1.2, 0.01, 0.001]
            geom.condim = 3
        elif "caster" in body:
            # Casters are FIXED links in the URDF (the swivel is not modelled), so a sticky
            # caster would drag the base. Frictionless contact = an ideal free caster.
            geom.condim = 1
            geom.friction = [0.0, 0.0, 0.0]
            geom.priority = 1  # so the caster's condim/friction win over the floor's

    # --- joint dynamics -------------------------------------------------------------------
    for joint in spec.joints:
        # The importer turns URDF <limit effort> into actuatorfrcrange. Those efforts are
        # ros2_control placeholders (lift effort=1.0 N cannot raise the column), so the
        # per-actuator forcerange below is the only force limit in the sim.
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
        if any(joint.name == f"{s}_{j}_joint" for s in SIDES for j in ARM_JOINTS) or \
                joint.name.endswith("_gripper_idler_joint"):
            joint.armature = ARM_ARMATURE
            joint.damping = _damping(max(float(_first(joint.damping)), 0.2))
        if joint.name.startswith("lift_"):
            joint.armature = 0.5
            joint.damping = _damping(50.0)

    # --- actuators, named exactly like the SDK descriptor (minus ".pos") ---------------------
    def position(name: str, joint: str, kp: float, kv: float, force: float) -> None:
        j = spec.joint(joint)
        act = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kv)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = list(j.range)
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-force, force]

    for side in SIDES:
        for dof in ARM_JOINTS:
            position(f"{side}_arm_{dof}", f"{side}_{dof}_joint", ARM_KP, ARM_KV, ARM_FORCE[dof])
    position("lift", "lift_extension_joint", LIFT_KP, 0.0, LIFT_FORCE)
    for side in SIDES:
        act = spec.add_actuator(name=f"{side}_wheel", target=f"{side}_wheel_joint",
                                trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=WHEEL_KV)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-20.0, 20.0]  # rad/s
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-WHEEL_FORCE, WHEEL_FORCE]

    # --- cameras: the four SDK roles, on the URDF's REP-103 optical frames ---------------
    # ROS optical frame: +Z out of the lens, +X right, +Y down. A MuJoCo camera looks down
    # its -Z with +Y up, i.e. the optical frame rotated 180 deg about X.
    for role in ("left_wrist", "right_wrist", "overhead", "front"):
        body = spec.body(f"{role}_camera_optical_frame")
        body.add_camera(name=role, quat=[0, 1, 0, 0], fovy=CAMERA_FOVY)
        body.add_site(name=f"{role}_camera", size=[0.005, 0, 0], group=4)

    for side in SIDES:
        spec.body(f"{side}_gripper_tcp_link").add_site(name=f"{side}_tcp", size=[0.008, 0, 0],
                                                        rgba=[1, 0.3, 0.1, 0.6], group=4)
    spec.body("imu_link").add_site(name="imu", size=[0.004, 0, 0], group=4)
    spec.body("lidar_scan_link").add_site(name="lidar", size=[0.004, 0, 0], group=4)

    for kind, name in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "imu_accel"),
                       (mujoco.mjtSensor.mjSENS_GYRO, "imu_gyro"),
                       (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "imu_quat")):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--description", type=Path, help="local nori_description checkout")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--keep-urdf", action="store_true", help="also write assets/nori.urdf")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        desc = fetch(args.description, args.tag, work)
        urdf = prepare_urdf(expand_xacro(desc))
        # MuJoCo resolves meshdir relative to the URDF file, so write it beside the meshes.
        urdf_path = desc / "nori_mj.urdf"
        urdf_path.write_text(urdf)
        spec = mujoco.MjSpec.from_file(str(urdf_path))
        n_eq = len(spec.equalities)
        if n_eq != 3:
            raise SystemExit(f"expected 3 mimic equalities from the importer, got {n_eq}")
        edit_spec(spec)
        spec.compile()  # fail loudly before writing anything

        mesh_out = ASSETS / "meshes"
        if mesh_out.exists():
            shutil.rmtree(mesh_out)
        shutil.copytree(desc / "meshes", mesh_out, ignore=shutil.ignore_patterns("LICENSE"))
        shutil.copy(desc / "meshes" / "LICENSE", mesh_out / "LICENSE")
        for name in ("LICENSE", "NOTICE"):
            shutil.copy(desc / name, ASSETS / f"UPSTREAM_{name}")
        if args.keep_urdf:
            (ASSETS / "nori.urdf").write_text(urdf)
        xml = spec.to_xml()
        header = (f"<!-- Generated by scripts/build_model.py from {UPSTREAM} @ {args.tag}.\n"
                  "     Nori A3 description (c) 2026 Nori Robotics Inc., CC BY-NC-SA 4.0.\n"
                  "     This converted model is a derivative and carries the same licence.\n"
                  "     Do not edit by hand; edit the build script and regenerate. -->\n")
        (ASSETS / "a3.xml").write_text(header + xml)

    model = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml")) \
        if (ASSETS / "scene.xml").exists() else mujoco.MjModel.from_xml_path(str(ASSETS / "a3.xml"))
    print(f"ok: nbody={model.nbody} njnt={model.njnt} nu={model.nu} neq={model.neq} "
          f"ncam={model.ncam} mass={sum(model.body_mass):.2f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
