"""Push-robustness challenge task for the sprawled hexapod."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from draft.tasks.hexapod.base_env import make_base_env_cfg
from draft.tasks.hexapod.curriculums import EpisodeSpeedMetric, PushScheduleCurriculum
from draft.tasks.hexapod.runner_cfg import make_runner_cfg  # noqa: F401

_PUSH_BASE_RANGE: dict[str, tuple[float, float]] = {
    "x":     (-0.5, 0.5),
    "y":     (-0.5, 0.5),
    "z":     (-0.3, 0.3),
    "roll":  (-0.50, 0.50),
    "pitch": (-0.50, 0.50),
    "yaw":   (-0.80, 0.80),
}

_PUSH_STAGES = [(0, 1.0), (400, 2.0), (800, 3.5), (1200, 5.0)]

_PUSH_COMMAND_RANGE = {
    "lin_vel_x": (-0.8, 0.8),
    "lin_vel_y": (-0.8, 0.8),
    "ang_vel_z": (-1.0, 1.0),
}

_PUSH_TRACK_STD = 0.4


def make_env_cfg(*, play: bool = False, **base) -> ManagerBasedRlEnvCfg:
    """The push challenge task for hexapod."""
    cfg, ctx = make_base_env_cfg(play=play, **base)
    if play:
        return cfg

    cfg.curriculum.pop("terrain_levels", None)

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = _PUSH_COMMAND_RANGE["lin_vel_x"]
    twist.ranges.lin_vel_y = _PUSH_COMMAND_RANGE["lin_vel_y"]
    twist.ranges.ang_vel_z = _PUSH_COMMAND_RANGE["ang_vel_z"]
    cfg.rewards["track_linear_velocity"].params["std"] = _PUSH_TRACK_STD

    push_event = ctx.push_event
    push_event.params["velocity_range"] = dict(_PUSH_BASE_RANGE)
    push_event.interval_range_s = (1.0 * ctx.s_sqrt, 3.0 * ctx.s_sqrt)
    cfg.events["push_robot"] = push_event

    cfg.curriculum["push_schedule"] = CurriculumTermCfg(
        func=PushScheduleCurriculum,
        params={
            "event_name": "push_robot",
            "base_range": dict(_PUSH_BASE_RANGE),
            "stages": [{"step": it * 24, "scale": sc} for it, sc in _PUSH_STAGES],
            "ema_alpha": 0.05,
        },
    )
    cfg.curriculum["episode_speed"] = CurriculumTermCfg(func=EpisodeSpeedMetric, params={})
    return cfg
