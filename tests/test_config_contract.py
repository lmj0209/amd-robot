from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[1]


def _mesh_file_set_sha256(directory: Path) -> tuple[int, int, str]:
    files = sorted(path for path in directory.iterdir() if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{path.name}\t{file_sha256}\n".encode())
    return len(files), sum(path.stat().st_size for path in files), digest.hexdigest()


def test_env_config_matches_python_contract() -> None:
    from amd_robo.contracts import ACTION_LAYOUT, REQUIRED_TERMINATION_SIGNALS

    config = yaml.safe_load((REPO_ROOT / "configs" / "env.yaml").read_text())
    assert config["implementation"] == "jax"
    assert config["robot"]["route"] == "go2_z1"
    assert config["control"]["simulation_dt"] == 0.002
    assert config["control"]["control_dt"] == 0.01
    assert config["control"]["action_size"] == ACTION_LAYOUT.size
    assert config["control"]["action_layout"] == {
        "legs": [0, 12],
        "arm": [12, 18],
        "gripper": [18, 19],
    }
    assert set(config["termination_requires"]) == REQUIRED_TERMINATION_SIGNALS


def test_assembled_robot_matches_the_asset_manifest() -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "assets" / "manifest.yaml").read_text()
    )
    assembled = next(
        asset for asset in manifest["assets"] if asset["id"] == "go2_z1"
    )
    robot_path = REPO_ROOT / assembled["repository_path"] / "go2_z1.xml"

    assert robot_path.read_bytes().endswith(b"\n")
    assert (
        hashlib.sha256(robot_path.read_bytes()).hexdigest()
        == assembled["sha256"]["go2_z1.xml"]
    )


def test_menagerie_fetch_defaults_to_the_manifest_commit() -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "assets" / "manifest.yaml").read_text()
    )
    pinned_commit = manifest["menagerie_commit"]
    script = (
        REPO_ROOT / "scripts" / "fetch_menagerie.sh"
    ).read_text(encoding="utf-8")

    assert f'DEFAULT_MENAGERIE_REF="{pinned_commit}"' in script
    assert 'origin "$MENAGERIE_REF"' in script
    assert "checkout --quiet --detach FETCH_HEAD" in script
    assert '"$COMMIT" != "$MENAGERIE_REF"' in script
    assert "/etc/ssl/certs/ca-certificates.crt" in script
    assert 'GIT_CA_ARGS=(-c "http.sslCAInfo=$ca_file")' in script
    assert 'git "${GIT_CA_ARGS[@]}" -C "$TMP/menagerie" fetch' in script
    assert "http.sslVerify=false" not in script

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "!assets/menagerie/unitree_go2/assets/*.obj" in gitignore
    assert "!assets/menagerie/unitree_z1/assets/*.stl" in gitignore


def test_selected_assets_have_pinned_redistribution_metadata() -> None:
    manifest = yaml.safe_load(
        (REPO_ROOT / "assets" / "manifest.yaml").read_text()
    )
    selected = {
        asset["id"]: asset
        for asset in manifest["assets"]
        if asset["id"] in {"unitree_go2", "unitree_z1_gripper"}
    }

    assert set(selected) == {"unitree_go2", "unitree_z1_gripper"}
    for asset in selected.values():
        assert asset["source_commit"] == manifest["menagerie_commit"]
        assert asset["license"] == "BSD-3-Clause"
        assert asset["redistribution"] == "allowed"
        license_path = REPO_ROOT / asset["license_file"]
        assert hashlib.sha256(license_path.read_bytes()).hexdigest() == (
            asset["license_sha256"]
        )
        mesh_directory = REPO_ROOT / asset["repository_path"] / "assets"
        file_count, byte_count, file_set_sha256 = _mesh_file_set_sha256(
            mesh_directory
        )
        assert file_count == asset["file_count"]
        assert byte_count == asset["bytes"]
        assert file_set_sha256 == asset["file_set_sha256"]

    robot_xml = (
        REPO_ROOT / "assets" / "menagerie" / "go2_z1" / "go2_z1.xml"
    ).read_text(encoding="utf-8")
    mesh_refs = [
        line.split('file="', 1)[1].split('"', 1)[0]
        for line in robot_xml.splitlines()
        if "<mesh " in line and 'file="' in line
    ]
    robot_directory = REPO_ROOT / "assets" / "menagerie" / "go2_z1"
    assert len(mesh_refs) == 36
    assert all((robot_directory / mesh_ref).is_file() for mesh_ref in mesh_refs)

    assert manifest["excluded_assets"] == [
        {
            "id": "b2_piper",
            "reason": (
                "not selected; no ATEC, Unitree B2, or AgileX Piper files "
                "are distributed"
            ),
        }
    ]


def test_rgc_lock_and_installers_use_audited_versions() -> None:
    lock = (REPO_ROOT / "requirements" / "rgc.lock").read_text()
    required_pins = {
        "brax==0.14.2",
        "jax==0.10.2",
        "jax-rocm7-pjrt==0.10.2",
        "jax-rocm7-plugin==0.10.2",
        "jaxlib==0.10.2",
        "mujoco==3.10.0",
        "mujoco-mjx==3.10.0",
        (
            "playground @ git+https://github.com/google-deepmind/"
            "mujoco_playground.git@"
            "43d180a226da3aae091d918b63c06c3a343519ad"
        ),
    }
    assert required_pins <= set(lock.splitlines())
    assert "git+ssh://" not in lock
    assert "gh-proxy.com" not in lock

    expected_jax = '"jax[rocm7-local]==0.10.2"'
    expected_playground_commit = "43d180a226da3aae091d918b63c06c3a343519ad"
    for relative_path in ("scripts/rgc_setup.sh", "docker/Dockerfile"):
        installer = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_jax in installer
        assert expected_playground_commit in installer


def test_smoke_config_is_fail_closed() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "smoke.yaml").read_text())
    assert config == {
        "schema_version": 1,
        "require_rocm": True,
        "require_mjx_impl": "jax",
        "num_envs": 256,
        "num_steps": 1000,
        "require_playground": True,
        "require_ppo_update": True,
        "require_checkpoint_roundtrip": True,
    }


def test_standing_config_preserves_rocm_training_guardrails() -> None:
    config = yaml.safe_load((REPO_ROOT / "configs" / "standing.yaml").read_text())

    assert config["algorithm"] == "brax_ppo"
    assert config["environment"]["control_timestep"] == 0.01
    assert config["environment"]["foot_condim"] == 6
    assert config["reward"] == {
        "profile": "smooth_height_velocity_pose_v1",
        "termination_cost": 2.0,
        "height_sigma": 0.02,
        "linear_velocity_sigma": 0.25,
        "angular_velocity_sigma": 0.25,
        "linear_velocity_scale": 1.0,
        "angular_velocity_scale": 0.5,
        "pose_scale": 0.5,
        "alive_scale": 0.1,
        "action_cost_scale": 0.001,
        "action_rate_cost_scale": 0.01,
    }
    assert config["status"] == "standing_qualified_zero_residual"
    assert config["ppo"]["num_timesteps"] == 5120
    assert config["ppo"]["episode_length"] == 128
    assert config["ppo"]["learning_rate"] == 0.0001
    assert config["rocm_guardrails"] == {
        "max_physics_substeps_per_control": 5,
        "max_training_steps_per_host_call": 2,
        "brax_run_evals": False,
    }
    assert config["manual_evaluation"]["implementation"] == ("sequential_python_loop")
    assert config["manual_evaluation"]["repeat_count"] == 8
    assert config["manual_evaluation"]["height_tolerance"] == 0.2
    assert config["checkpoint"] == {
        "scope": "full_training_session",
        "interval_steps": 5120,
        "includes_rollout_state": True,
        "policy_snapshot_scope": "inference_params",
        "policy_snapshot_interval_steps": 5120,
    }
    assert config["precision"]["status"] == "default_measured_for_diagnostic"
