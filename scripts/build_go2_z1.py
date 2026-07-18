#!/usr/bin/env python3
"""Assemble the Go2+Z1(gripper) merged MJCF (19 DoF) from MuJoCo Menagerie.

Reads the menagerie ``unitree_go2/go2_mjx.xml`` (12-DoF quadruped, MJX-tuned)
and ``unitree_z1/z1_gripper.xml`` (6-DoF arm + 1-DoF gripper), and emits a
single ``go2_z1.xml`` whose actuator order matches the frozen project contract
in ``src/amd_robo/contracts.py`` (``ACTION_LAYOUT``)::

    legs   [0:12] -> FL/FR/RL/RR x hip/thigh/calf   (from go2_mjx.xml)
    arm    [12:18] -> motor1..motor6                 (from z1_gripper.xml)
    gripper [18:19] -> motorGripper                  (from z1_gripper.xml)

The Z1 base body ``link00`` (joint-less in menagerie, i.e. a fixed stator) is
welded onto the Go2 trunk body ``base`` as its LAST child body, so the seven
arm/gripper joints follow the twelve leg joints in qpos order and line up with
the contract slices above.

Meshes stay in their menagerie dirs and are referenced by a path relative to
the merged model (``meshdir="."``), so the model resolves wherever
``scripts/fetch_menagerie.sh`` has been run. No mesh bytes are copied.

The arm carry pose is a PLACEHOLDER (welded on top of the trunk, reaching
forward); it is calibrated during Phase 2 locomotion work, not at this model
gate, whose only bar is "compiles + 19-DoF contract + finite under home ctrl".

Run after fetching menagerie::

    python scripts/build_go2_z1.py

Outputs the robot model at ``assets/menagerie/go2_z1/go2_z1.xml`` and a flat
ground scene at ``assets/menagerie/go2_z1/scene_mjx.xml``, then prints compile,
DoF, and ground-plane audits.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = REPO_ROOT / "assets" / "menagerie"
GO2_XML = MENAGERIE / "unitree_go2" / "go2_mjx.xml"
GO2_SCENE_XML = MENAGERIE / "unitree_go2" / "scene_mjx.xml"
Z1_XML = MENAGERIE / "unitree_z1" / "z1_gripper.xml"
OUT_DIR = MENAGERIE / "go2_z1"
OUT_XML = OUT_DIR / "go2_z1.xml"
OUT_SCENE_XML = OUT_DIR / "scene_mjx.xml"

# Actuator order must match src/amd_robo/contracts.py::ACTION_LAYOUT exactly.
EXPECTED_LEG_ACTUATORS = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
)
EXPECTED_ARM_ACTUATORS = ("motor1", "motor2", "motor3", "motor4", "motor5", "motor6")
EXPECTED_GRIPPER_ACTUATOR = ("motorGripper",)
EXPECTED_ACTUATORS = (
    list(EXPECTED_LEG_ACTUATORS)
    + list(EXPECTED_ARM_ACTUATORS)
    + list(EXPECTED_GRIPPER_ACTUATOR)
)

# Placeholder carry pose: Z1 stator welded on top of the trunk, arm reaching
# forward over the head (180 deg about z maps the Z1 home reach from -x to +x).
ARM_MOUNT_POS = "0.12 0 0.058"
ARM_MOUNT_QUAT = "0 0 0 1"

# MuJoCo default class names are GLOBAL, so Z1's "visual"/"collision" classes
# clash with Go2's even though they sit under different parent classes. Prefix
# the Z1 ones; the gripper subclasses are already z1_-prefixed and unique.
Z1_CLASS_RENAME = {"visual": "z1_visual", "collision": "z1_collision"}

# MJX (mujoco-mjx 3.10.0) does not implement contact margin/gap for
# (sphere, mesh) pairs. Go2's feet are spheres carrying margin=0.001, so the Z1
# gripper's mesh collision hulls would make mjx.put_model raise
# NotImplementedError. Drop collision on those mesh geoms; the gripper still
# collides via its primitive box pads. Gripper mesh-precise collision only
# matters for grasping, which is post-MVP.
Z1_MESH_COLLISION_CLASSES = (
    "z1_gripper_stator_collision",
    "z1_gripper_mover_collision",
)


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _rename_z1_classes(root: ET.Element) -> None:
    """Prefix Z1's visual/collision default classes to avoid a global name clash.

    Renames every ``class`` attribute (both the ``<default class=...>`` definitions
    and the ``<geom class=...>`` references) so Z1's visual/collision resolve to
    the prefixed classes.
    """
    for el in root.iter():
        c = el.get("class")
        if c in Z1_CLASS_RENAME:
            el.set("class", Z1_CLASS_RENAME[c])


def _disable_z1_mesh_collision(root: ET.Element) -> None:
    """Make the Z1 gripper mesh collision geoms non-colliding (MJX compat)."""
    for geom in root.iter("geom"):
        if geom.get("class") in Z1_MESH_COLLISION_CLASSES:
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def _z1_collision_to_capsule(root: ET.Element) -> None:
    """Convert the Z1 arm-link collision cylinders to capsules (MJX compat).

    MJX has no cylinder-vs-{box,sphere,capsule} contact. MuJoCo cylinder size is
    ``[radius, half-length]``, while a capsule adds hemispherical ends outside
    its cylindrical half-length. Reusing the two numbers verbatim lengthens the
    arm geoms and creates self-penetration at the home pose. Use a conservative
    capsule whose outer half-length does not exceed the source cylinder. The
    gripper pad boxes set their own type and are unaffected.
    """
    for d in root.iter("default"):
        if d.get("class") == "z1_collision":
            g = d.find("geom")
            if g is not None and g.get("type", "cylinder") == "cylinder":
                g.set("type", "capsule")

    for geom in root.iter("geom"):
        if geom.get("class") != "z1_collision" or geom.get("type") is not None:
            continue
        size = geom.get("size")
        if size is None:
            raise SystemExit("Z1 cylinder collision geom is missing size")
        radius, cylinder_half_length = map(float, size.split())
        capsule_radius = min(radius, cylinder_half_length)
        capsule_half_length = max(cylinder_half_length - capsule_radius, 1.0e-6)
        geom.set("size", f"{capsule_radius:g} {capsule_half_length:g}")


def _deepcopy(el: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(el))


def _rewrite_file_paths(asset_el: ET.Element, robot_dir: str, meshdir: str) -> None:
    """Point each asset file at its menagerie dir, relative to go2_z1/."""
    prefix = f"../{robot_dir}/{meshdir}/" if meshdir else f"../{robot_dir}/"
    for child in asset_el:
        f = child.get("file")
        if f and not f.startswith("../"):
            child.set("file", prefix + f)


def build() -> ET.Element:
    go2 = _parse(GO2_XML)
    z1 = _parse(Z1_XML)
    _rename_z1_classes(z1)
    _disable_z1_mesh_collision(z1)
    _z1_collision_to_capsule(z1)

    merged = ET.Element("mujoco", {"model": "go2_z1"})

    # compiler: Go2 settings, but resolve meshes per-file against the model dir.
    go2_compiler = go2.find("compiler")
    compiler = ET.SubElement(merged, "compiler", dict(go2_compiler.attrib))
    compiler.set("meshdir", ".")

    # option: Go2's contact settings, but with Z1's implicitfast integrator and
    # implicit joint/actuator damping. Go2's option (default Euler integrator,
    # eulerdamp disabled) is fine for the soft leg PD (kp~50) but goes unstable
    # with the Z1 arm's stiff PD (kp 1000..1500) on light links: max joint
    # velocity diverges within ~10 steps. implicitfast + implicit damping is the
    # config z1.xml itself uses and stabilises the arm.
    go2_option = go2.find("option")
    if go2_option is not None:
        opt = _deepcopy(go2_option)
        opt.set("integrator", "implicitfast")
        flag = opt.find("flag")
        if flag is not None:
            flag.attrib.pop("eulerdamp", None)
            if not flag.attrib:
                opt.remove(flag)
        merged.append(opt)

    # defaults: Go2 classes and Z1 classes side by side under one <default>.
    default = ET.SubElement(merged, "default")
    for src in (go2, z1):
        src_default = src.find("default")
        if src_default is not None:
            for child in list(src_default):
                default.append(child)

    # asset: Go2 + Z1 meshes/materials, file paths rewritten to the menagerie.
    asset = ET.SubElement(merged, "asset")
    for src, robot_dir in ((go2, "unitree_go2"), (z1, "unitree_z1")):
        src_asset = src.find("asset")
        meshdir = src.find("compiler").get("meshdir", "")
        _rewrite_file_paths(src_asset, robot_dir, meshdir)
        for child in list(src_asset):
            asset.append(child)

    # worldbody: Go2 trunk + legs, with the Z1 arm welded as the LAST child of
    # the trunk body ``base`` so arm joints follow leg joints in qpos order.
    go2_world = go2.find("worldbody")
    base = go2_world.find(".//body[@name='base']")
    if base is None:
        raise SystemExit("go2 trunk body 'base' not found in go2_mjx.xml")
    z1_world = z1.find("worldbody")
    z1_link00 = z1_world.find("body[@name='link00']")
    if z1_link00 is None:
        raise SystemExit("z1 base body 'link00' not found in z1_gripper.xml")
    arm = _deepcopy(z1_link00)
    arm.set("pos", ARM_MOUNT_POS)
    arm.set("quat", ARM_MOUNT_QUAT)
    base.append(arm)
    merged.append(_deepcopy(go2_world))

    # contact: carry over Go2 contact exclusions if any (self-collision policy).
    go2_contact = go2.find("contact")
    if go2_contact is not None:
        merged.append(_deepcopy(go2_contact))

    # actuator: 12 legs ++ 6 arm ++ 1 gripper = 19, in contract order.
    actuator = ET.SubElement(merged, "actuator")
    for src in (go2, z1):
        for a in list(src.find("actuator")):
            actuator.append(a)

    # sensor: keep Go2 IMU + joint sensors (arm sensors can be added later).
    go2_sensor = go2.find("sensor")
    if go2_sensor is not None:
        merged.append(_deepcopy(go2_sensor))

    # keyframe: one merged "home" = Go2 home qpos/ctrl ++ Z1 home qpos/ctrl.
    # Valid because arm joints are the last 7 in qpos (welded as last child).
    go2_key = go2.find("keyframe/key[@name='home']")
    z1_key = z1.find("keyframe/key[@name='home']")
    if go2_key is None or z1_key is None:
        raise SystemExit("home keyframe missing in a source model")
    keyframe = ET.SubElement(merged, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": "home",
            "qpos": f"{go2_key.get('qpos')} {z1_key.get('qpos')}",
            "ctrl": f"{go2_key.get('ctrl')} {z1_key.get('ctrl')}",
        },
    )

    return merged


def build_scene() -> ET.Element:
    """Reuse the official Go2 MJX flat scene with the assembled robot."""
    scene = _parse(GO2_SCENE_XML)
    scene.set("model", "go2_z1 scene")
    include = scene.find("include")
    if include is None:
        raise SystemExit(f"robot include missing in {GO2_SCENE_XML}")
    include.set("file", OUT_XML.name)
    return scene


def audit(xml_path: Path, *, expect_floor: bool = False) -> None:
    """Compile with CPU MuJoCo and assert the 19-DoF contract."""
    try:
        import mujoco
    except ImportError:
        print("mujoco not importable here; skipping compile audit (run on RGC).")
        return
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    except Exception as exc:  # noqa: BLE001
        print(f"COMPILE FAILED: {exc}", file=sys.stderr)
        raise

    def names(obj) -> list[str]:
        return [
            mujoco.mj_id2name(model, obj, i)
            for i in range(
                model.nu if obj == mujoco.mjtObj.mjOBJ_ACTUATOR else model.njnt
            )
        ]

    actuator_names = names(mujoco.mjtObj.mjOBJ_ACTUATOR)
    joint_names = names(mujoco.mjtObj.mjOBJ_JOINT)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nkey={model.nkey}")
    print(f"actuator order ({len(actuator_names)}): {actuator_names}")
    print(f"last 7 joints (arm+gripper): {joint_names[-7:]}")

    assert model.nu == 19, f"nu={model.nu} != 19"
    assert actuator_names == EXPECTED_ACTUATORS, (
        f"actuator order mismatch:\n got {actuator_names}\n exp {EXPECTED_ACTUATORS}"
    )
    assert joint_names[-7:] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "jointGripper",
    ], f"arm joints not last in qpos: {joint_names[-7:]}"
    assert model.key_qpos[0].shape[0] == model.nq, (
        f"home qpos len {model.key_qpos[0].shape[0]} != nq {model.nq}"
    )
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if expect_floor:
        assert floor_id >= 0, "flat scene is missing the floor geom"
        assert model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE, (
            "floor geom is not a plane"
        )
        data = mujoco.MjData(model)
        data.qpos[:] = model.key_qpos[0]
        data.ctrl[:] = model.key_ctrl[0]
        mujoco.mj_forward(model, data)
        non_floor_contacts = [
            (contact.geom1, contact.geom2)
            for contact in data.contact[: data.ncon]
            if floor_id not in (contact.geom1, contact.geom2)
        ]
        assert not non_floor_contacts, (
            f"home pose has non-floor contacts: {non_floor_contacts}"
        )
    print("COMPILE + 19-DoF CONTRACT AUDIT PASSED")


def main() -> int:
    missing_sources = [
        path for path in (GO2_XML, GO2_SCENE_XML, Z1_XML) if not path.exists()
    ]
    if missing_sources:
        print(
            f"menagerie source missing: {missing_sources}\n"
            f"run scripts/fetch_menagerie.sh first",
            file=sys.stderr,
        )
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build()
    ET.indent(merged, space="  ")
    ET.ElementTree(merged).write(OUT_XML, encoding="utf-8", xml_declaration=True)
    scene = build_scene()
    ET.indent(scene, space="  ")
    ET.ElementTree(scene).write(OUT_SCENE_XML, encoding="utf-8", xml_declaration=True)
    print(f"wrote {OUT_XML.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_SCENE_XML.relative_to(REPO_ROOT)}")
    audit(OUT_XML)
    audit(OUT_SCENE_XML, expect_floor=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
