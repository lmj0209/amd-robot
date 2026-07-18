"""Full Brax learner and training-session checkpoints for resumable PPO."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STATE_FILENAME = "learner_state.msgpack"
SESSION_FILENAME = "training_session.msgpack"
MANIFEST_FILENAME = "manifest.json"


class LearnerCheckpointError(RuntimeError):
    """Raised when a learner or training-session checkpoint is invalid."""


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _save_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    value: Any,
    state_filename: str,
    scope: str,
    includes: list[str],
    includes_rollout_state: bool,
    flatten_pytree: bool,
    metadata: Mapping[str, Any] | None,
) -> Path:
    if step < 0:
        raise ValueError("step must be non-negative")

    from flax import serialization

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    serialized_value = value
    leaf_count = None
    if flatten_pytree:
        import jax

        serialized_value, _ = jax.tree_util.tree_flatten(value)
        leaf_count = len(serialized_value)
    state_bytes = serialization.to_bytes(serialized_value)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "scope": scope,
        "state_file": state_filename,
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "includes": includes,
        "includes_rollout_state": includes_rollout_state,
        "serialization": ("jax_pytree_leaves" if flatten_pytree else "flax_state_dict"),
        "leaf_count": leaf_count,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
    }
    _atomic_write(checkpoint_dir / state_filename, state_bytes)
    _atomic_write(
        checkpoint_dir / MANIFEST_FILENAME,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return checkpoint_dir


def _load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    template: Any,
    expected_state_filename: str,
    expected_scope: str,
    flatten_pytree: bool,
) -> Any:
    from flax import serialization

    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / MANIFEST_FILENAME
    state_path = checkpoint_dir / expected_state_filename
    if not manifest_path.is_file() or not state_path.is_file():
        raise LearnerCheckpointError(f"incomplete checkpoint: {checkpoint_dir}")

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LearnerCheckpointError(f"invalid checkpoint manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LearnerCheckpointError(
            f"unsupported checkpoint schema: {manifest.get('schema_version')}"
        )
    if manifest.get("scope") != expected_scope:
        raise LearnerCheckpointError("unexpected checkpoint scope")
    if manifest.get("state_file") != expected_state_filename:
        raise LearnerCheckpointError("unexpected checkpoint state filename")
    expected_serialization = (
        "jax_pytree_leaves" if flatten_pytree else "flax_state_dict"
    )
    if manifest.get("serialization") != expected_serialization:
        raise LearnerCheckpointError("unexpected checkpoint serialization")

    state_bytes = state_path.read_bytes()
    actual_sha256 = hashlib.sha256(state_bytes).hexdigest()
    if actual_sha256 != manifest.get("state_sha256"):
        raise LearnerCheckpointError("checkpoint state SHA256 mismatch")
    try:
        if not flatten_pytree:
            return serialization.from_bytes(template, state_bytes)

        import jax

        template_leaves, tree_definition = jax.tree_util.tree_flatten(template)
        if manifest.get("leaf_count") != len(template_leaves):
            raise LearnerCheckpointError("checkpoint PyTree leaf count mismatch")
        restored_leaves = serialization.from_bytes(template_leaves, state_bytes)
        return jax.tree_util.tree_unflatten(tree_definition, restored_leaves)
    except Exception as exc:
        if isinstance(exc, LearnerCheckpointError):
            raise
        raise LearnerCheckpointError(
            f"checkpoint does not match the initialized template: {exc}"
        ) from exc


def save_learner_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    training_state: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Saves optimizer, networks, normalizer, and cumulative environment steps."""

    return _save_checkpoint(
        checkpoint_dir,
        step=step,
        value=training_state,
        state_filename=STATE_FILENAME,
        scope="brax_ppo_learner_state",
        includes=[
            "optimizer_state",
            "policy_params",
            "value_params",
            "normalizer_params",
            "env_steps",
        ],
        includes_rollout_state=False,
        flatten_pytree=False,
        metadata=metadata,
    )


def load_learner_checkpoint(
    checkpoint_dir: str | Path,
    *,
    training_state_template: Any,
) -> Any:
    """Loads a learner checkpoint into a freshly initialized state template."""

    return _load_checkpoint(
        checkpoint_dir,
        template=training_state_template,
        expected_state_filename=STATE_FILENAME,
        expected_scope="brax_ppo_learner_state",
        flatten_pytree=False,
    )


def save_training_session_checkpoint(
    checkpoint_dir: str | Path,
    *,
    step: int,
    training_session: Any,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Saves learner, environment rollout, and PRNG state."""

    return _save_checkpoint(
        checkpoint_dir,
        step=step,
        value=training_session,
        state_filename=SESSION_FILENAME,
        scope="brax_ppo_training_session",
        includes=[
            "optimizer_state",
            "policy_params",
            "value_params",
            "normalizer_params",
            "env_steps",
            "environment_rollout_state",
            "learner_prng_key",
            "environment_prng_keys",
        ],
        includes_rollout_state=True,
        flatten_pytree=True,
        metadata=metadata,
    )


def load_training_session_checkpoint(
    checkpoint_dir: str | Path,
    *,
    training_session_template: Any,
) -> Any:
    """Loads an exact training session into freshly initialized templates."""

    return _load_checkpoint(
        checkpoint_dir,
        template=training_session_template,
        expected_state_filename=SESSION_FILENAME,
        expected_scope="brax_ppo_training_session",
        flatten_pytree=True,
    )


def _make_checkpoint_callback(
    checkpoint_root: str | Path,
    *,
    interval_steps: int,
    metadata: Mapping[str, Any] | None,
    announce: Callable[[str], None],
    save: Callable[..., Path],
    value_keyword: str,
    event_name: str,
) -> Callable[[int, Any], None]:
    if interval_steps <= 0:
        raise ValueError("interval_steps must be positive")
    checkpoint_root = Path(checkpoint_root)
    checkpoint_metadata = dict(metadata or {})

    def callback(step: int, value: Any) -> None:
        if step <= 0 or step % interval_steps:
            return
        destination = checkpoint_root / f"step_{step:012d}"
        save(
            destination,
            step=step,
            metadata=checkpoint_metadata,
            **{value_keyword: value},
        )
        announce(f"{event_name} step={step} path={destination}")

    return callback


def make_learner_checkpoint_callback(
    checkpoint_root: str | Path,
    *,
    interval_steps: int,
    metadata: Mapping[str, Any] | None = None,
    announce: Callable[[str], None] = print,
) -> Callable[[int, Any], None]:
    """Builds a callback for learner-only checkpoints."""

    return _make_checkpoint_callback(
        checkpoint_root,
        interval_steps=interval_steps,
        metadata=metadata,
        announce=announce,
        save=save_learner_checkpoint,
        value_keyword="training_state",
        event_name="TRAINING_STATE_SAVED",
    )


def make_training_session_checkpoint_callback(
    checkpoint_root: str | Path,
    *,
    interval_steps: int,
    metadata: Mapping[str, Any] | None = None,
    announce: Callable[[str], None] = print,
) -> Callable[[int, Any], None]:
    """Builds a callback for exact learner, rollout, and PRNG checkpoints."""

    return _make_checkpoint_callback(
        checkpoint_root,
        interval_steps=interval_steps,
        metadata=metadata,
        announce=announce,
        save=save_training_session_checkpoint,
        value_keyword="training_session",
        event_name="TRAINING_SESSION_SAVED",
    )
