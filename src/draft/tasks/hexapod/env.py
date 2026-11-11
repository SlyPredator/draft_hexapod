"""Flat-ground hexapod locomotion task: the base policy for challenge curricula."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from draft.tasks.hexapod.base_env import make_base_env_cfg


def make_env_cfg(**base) -> ManagerBasedRlEnvCfg:
    """The base hexapod environment, with no curriculum on top."""
    cfg, _ctx = make_base_env_cfg(**base)
    return cfg
