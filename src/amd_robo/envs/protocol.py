"""Typing protocol for the project MuJoCo Playground-style environment."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProjectMjxEnv(Protocol):
    """Minimum interface required by training, evaluation, and benchmarks."""

    @property
    def action_size(self) -> int: ...

    @property
    def mj_model(self) -> Any: ...

    @property
    def mjx_model(self) -> Any: ...

    def reset(self, rng: Any) -> Any: ...

    def step(self, state: Any, action: Any) -> Any: ...
