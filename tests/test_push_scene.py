from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSH_SCENE_XML = (
    REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "scene_push_mjx.xml"
)


def _id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"{name} is missing"
    return object_id


def test_push_scene_has_physical_box_and_noncontact_task_markers():
    model = mujoco.MjModel.from_xml_path(str(PUSH_SCENE_XML))
    box_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "push_box_body")
    box_joint_id = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "push_box_joint")
    box_geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "push_box")
    push_pad_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "push_pad_stator_left",
            "push_pad_stator_right",
            "push_pad_mover_left",
            "push_pad_mover_right",
        )
    ]
    push_key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, "push_home")
    end_effector_site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "z1_ee")
    push_contact_site_id = _id(
        model, mujoco.mjtObj.mjOBJ_SITE, "push_contact_site"
    )
    prepush_site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "prepush_site")
    goal_site_id = _id(model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
    floor_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    assert model.nu == 19
    assert model.opt.iterations == 8
    assert model.jnt_type[box_joint_id] == mujoco.mjtJoint.mjJNT_FREE
    assert model.body_jntnum[box_body_id] == 1
    assert model.geom_bodyid[box_geom_id] == box_body_id
    assert all(
        model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        for geom_id in push_pad_ids
    )
    assert model.site_bodyid[push_contact_site_id] == box_body_id
    assert model.site_bodyid[prepush_site_id] == box_body_id
    assert model.site_bodyid[goal_site_id] == 0
    np.testing.assert_allclose(
        model.site_pos[end_effector_site_id], [0.186, 0.0, -0.009]
    )
    np.testing.assert_allclose(
        model.site_pos[push_contact_site_id], [-0.105, 0.0, 0.0]
    )

    box_qpos_adr = model.jnt_qposadr[box_joint_id]
    np.testing.assert_allclose(
        model.key_qpos[push_key_id, box_qpos_adr : box_qpos_adr + 7],
        [0.8, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        model.site_pos[prepush_site_id], [-0.3, 0.0, -0.085]
    )
    np.testing.assert_allclose(model.site_pos[goal_site_id], [1.2, 0.0, 0.005])

    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[push_key_id]
    data.ctrl[:] = model.key_ctrl[push_key_id]
    mujoco.mj_forward(model, data)
    assert all(
        floor_id in (contact.geom1, contact.geom2)
        for contact in data.contact[: data.ncon]
    )

    initial_box_position = data.qpos[box_qpos_adr : box_qpos_adr + 3].copy()
    # Match the 1,600-control-step task preflight (five physics substeps each)
    # so short-lived box-plane stability cannot mask long-horizon drift.
    for _ in range(8000):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert np.linalg.norm(
        data.qpos[box_qpos_adr : box_qpos_adr + 3] - initial_box_position
    ) < 1.0e-4
