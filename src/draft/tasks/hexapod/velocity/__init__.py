"""Top-speed challenge task for the sprawled hexapod.

The base env with the velocity command ramped up to COMMON_V_CEILING.
Reward shaping and loosened gait terms are scaled by estimate_top_speed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from draft.tasks.hexapod.base_env import make_base_env_cfg
from draft.tasks.hexapod.speed_limits import (
    COMMON_V_CEILING,
    SHAPE_HEADROOM,
    estimate_top_speed,
)
from draft.tasks.hexapod.curriculums import EpisodeSpeedMetric
from draft.tasks.hexapod.runner_cfg import make_runner_cfg  # noqa: F401

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _forward_speed_bonus(env: "ManagerBasedRlEnv", v_cap: float) -> torch.Tensor:
    """Body speed toward the command, clamped to v_cap and normalised by it."""
    from mjlab.entity import Entity
    asset: Entity = env.scene["robot"]
    cmd_x = env.command_manager.get_command("twist")[:, 0]
    v_x = asset.data.root_link_lin_vel_b[:, 0]
    aligned = v_x * torch.sign(cmd_x)
    bonus = torch.clamp(aligned, min=0.0, max=v_cap) / v_cap
    gate = (cmd_x.abs() > 0.5 * v_cap).float()
    return bonus * gate


def make_env_cfg(*, play: bool = False, robot_xml, task_geometry_xml=None,
                 **base) -> ManagerBasedRlEnvCfg:
    """The speed task for hexapod."""
    cfg, ctx = make_base_env_cfg(play=play, robot_xml=robot_xml,
                                 task_geometry_xml=task_geometry_xml, **base)
    if play:
        return cfg

    cfg.curriculum.pop("terrain_levels", None)

    v_max = COMMON_V_CEILING
    v_shape = SHAPE_HEADROOM * estimate_top_speed(task_geometry_xml or robot_xml)

    lat, yaw = 0.5, 0.5
    stages = [(0, 0.4), (400, 0.6), (900, 0.8), (1400, 1.0)]

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-stages[0][1] * v_max, stages[0][1] * v_max)
    twist.ranges.lin_vel_y = (-lat, lat)
    twist.ranges.ang_vel_z = (-yaw, yaw)

    command_vel = ctx.base_curriculum["command_vel"]
    command_vel.params["velocity_stages"] = [
        {"step": it * 24,
         "lin_vel_x": (-f * v_max, f * v_max),
         "lin_vel_y": (-lat, lat),
         "ang_vel_z": (-yaw, yaw)}
        for it, f in stages
    ]
    cfg.curriculum["command_vel"] = command_vel

    cfg.rewards["track_linear_velocity"].params["std"] = 0.4 * v_shape
    cfg.rewards["air_time"].params["threshold_max"] = 0.4
    cfg.rewards["air_time"].weight = 0.3
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=_forward_speed_bonus, weight=0.6, params={"v_cap": v_shape}
    )

    cfg.curriculum["episode_speed"] = CurriculumTermCfg(func=EpisodeSpeedMetric, params={})
    return cfg
