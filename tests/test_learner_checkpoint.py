from __future__ import annotations

import json
import sys
import types

import pytest

from amd_robo.training.learner_checkpoint import (
    LearnerCheckpointError,
    load_learner_checkpoint,
    load_training_session_checkpoint,
    make_learner_checkpoint_callback,
    make_training_session_checkpoint_callback,
    save_learner_checkpoint,
    save_training_session_checkpoint,
)


@pytest.fixture
def fake_flax(monkeypatch):
    serialization = types.SimpleNamespace(
        to_bytes=lambda value: json.dumps(value, sort_keys=True).encode(),
        from_bytes=lambda _template, value: json.loads(value),
    )
    flax = types.ModuleType("flax")
    flax.serialization = serialization
    monkeypatch.setitem(sys.modules, "flax", flax)
    tree_util = types.SimpleNamespace(
        tree_flatten=lambda value: ([value], "single_leaf"),
        tree_unflatten=lambda _definition, leaves: leaves[0],
    )
    jax = types.ModuleType("jax")
    jax.tree_util = tree_util
    monkeypatch.setitem(sys.modules, "jax", jax)
    return serialization


def test_learner_checkpoint_round_trip_and_manifest(tmp_path, fake_flax) -> None:
    state = {"optimizer": [1, 2], "env_steps": 512}
    checkpoint = save_learner_checkpoint(
        tmp_path / "checkpoint",
        step=512,
        training_state=state,
        metadata={"seed": 0},
    )

    restored = load_learner_checkpoint(
        checkpoint,
        training_state_template={"optimizer": [], "env_steps": 0},
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text())

    assert restored == state
    assert manifest["step"] == 512
    assert manifest["metadata"] == {"seed": 0}
    assert manifest["includes_rollout_state"] is False


def test_learner_checkpoint_rejects_corrupt_state(tmp_path, fake_flax) -> None:
    checkpoint = save_learner_checkpoint(
        tmp_path / "checkpoint",
        step=512,
        training_state={"optimizer": [1]},
    )
    (checkpoint / "learner_state.msgpack").write_bytes(b"corrupt")

    with pytest.raises(LearnerCheckpointError, match="SHA256"):
        load_learner_checkpoint(checkpoint, training_state_template={})


def test_checkpoint_callback_uses_interval_and_step_directory(
    tmp_path, fake_flax
) -> None:
    announcements = []
    callback = make_learner_checkpoint_callback(
        tmp_path,
        interval_steps=512,
        announce=announcements.append,
    )

    callback(0, {"value": 0})
    callback(256, {"value": 1})
    callback(512, {"value": 2})

    checkpoint = tmp_path / "step_000000000512"
    assert checkpoint.is_dir()
    assert load_learner_checkpoint(
        checkpoint, training_state_template={"value": 0}
    ) == {"value": 2}
    assert announcements == [f"TRAINING_STATE_SAVED step=512 path={checkpoint}"]


def test_training_session_round_trip_includes_rollout_and_rng(
    tmp_path, fake_flax
) -> None:
    session = (
        {"optimizer": [1], "env_steps": 512},
        {"rollout": [2]},
        [3, 4],
        [[5, 6]],
    )
    checkpoint = save_training_session_checkpoint(
        tmp_path / "session",
        step=512,
        training_session=session,
    )

    restored = load_training_session_checkpoint(
        checkpoint,
        training_session_template=({}, {}, [], []),
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text())

    assert restored == [
        {"env_steps": 512, "optimizer": [1]},
        {"rollout": [2]},
        [3, 4],
        [[5, 6]],
    ]
    assert manifest["scope"] == "brax_ppo_training_session"
    assert manifest["includes_rollout_state"] is True


def test_training_session_callback_uses_distinct_event_name(
    tmp_path, fake_flax
) -> None:
    announcements = []
    callback = make_training_session_checkpoint_callback(
        tmp_path,
        interval_steps=512,
        announce=announcements.append,
    )

    callback(512, ({"learner": 1}, {"rollout": 2}, [3], [[4]]))

    checkpoint = tmp_path / "step_000000000512"
    assert (checkpoint / "training_session.msgpack").is_file()
    assert announcements == [f"TRAINING_SESSION_SAVED step=512 path={checkpoint}"]
