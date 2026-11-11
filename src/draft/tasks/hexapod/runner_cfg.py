"""PPO runner configuration for the hexapod locomotion tasks."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from draft.tasks.quadruped.runner_cfg import (
    QuadrupedLocomotionRunnerCfg,
    make_runner_cfg as _make_quadruped_runner_cfg,
)


@dataclass
class HexapodLocomotionRunnerCfg(QuadrupedLocomotionRunnerCfg):
    run_name: str = "hexapod_locomotion"
    wandb_project: str = "hexapod_locomotion"


def make_runner_cfg(**overrides) -> HexapodLocomotionRunnerCfg:
    """Build an RslRlOnPolicyRunnerCfg for hexapod locomotion."""
    cfg = HexapodLocomotionRunnerCfg()
    if overrides:
        cfg.apply_overrides(overrides)
    return cfg
