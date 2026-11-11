"""Rough-terrain challenge task for the sprawled hexapod."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from draft.tasks.hexapod.base_env import _FOOT_SITES, make_base_env_cfg
from draft.tasks.hexapod.runner_cfg import make_runner_cfg  # noqa: F401

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_MAX_STEP_HEIGHT = 0.40
_MAX_NOISE = 0.25
_MAX_WAVE_AMP = 0.40


def _apply_ceiling(st, hfield_resolution: float, s: float = 1.0):
    """Cap height parameters of sub-terrain and set hfield resolution."""
    overrides: dict = {}
    if hasattr(st, "step_height_range"):
        overrides["step_height_range"] = (st.step_height_range[0], _MAX_STEP_HEIGHT * s)
    if hasattr(st, "noise_range"):
        overrides["noise_range"] = (st.noise_range[0], _MAX_NOISE * s)
    if hasattr(st, "amplitude_range"):
        overrides["amplitude_range"] = (st.amplitude_range[0], _MAX_WAVE_AMP * s)
    if hasattr(st, "horizontal_scale"):
        overrides["horizontal_scale"] = hfield_resolution
    return replace(st, **overrides) if overrides else st


def _hexapod_terrain_base_too_low(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate only if trunk root drops below minimum_height above the lowest foot.

    In rough terrain, comparing against mean(feet_z) causes false resets whenever
    legs are split across stair ledges or slope crests. Comparing against min(feet_z)
    cleanly detects actual chassis collapse without penalizing stepped descent.
    """
    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    return (root_z - torch.amin(feet_z, dim=-1)) < minimum_height


def _lin_vel_z_downward_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Only penalize downward vertical crashing (vz < 0); allow upward climbing (vz > 0)."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_link_lin_vel_b[:, 2]
    return torch.square(torch.clamp(vz, max=0.0))


def make_env_cfg(*, play: bool = False, terrain_scale_with_size: bool = False,
                 **base) -> ManagerBasedRlEnvCfg:
    """The step-height axis task for hexapod."""
    cfg, ctx = make_base_env_cfg(
        play=play, terrain_scale_with_size=terrain_scale_with_size, **base)
    ts = ctx.s if terrain_scale_with_size else 1.0

    # Ensure foot sphere never spans >1 hfield cell
    hfield_scale = max(0.08, 2.0 * ctx.foot_radius)

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    gen = replace(ROUGH_TERRAINS_CFG)
    gen.curriculum = True
    gen.sub_terrains = {
        name: _apply_ceiling(st, hfield_scale, ts)
        for name, st in gen.sub_terrains.items()
    }

    _HEIGHT_WEIGHTS = {
        "pyramid_stairs": 0.25, "pyramid_stairs_inv": 0.25,
        "wave_terrain": 0.20, "random_rough": 0.15,
        "hf_pyramid_slope": 0.05, "hf_pyramid_slope_inv": 0.05, "flat": 0.05,
    }
    _STEP_FLOOR = 0.06 * ts
    reweighted: dict = {}
    for name, st in gen.sub_terrains.items():
        ov: dict = {}
        if name in _HEIGHT_WEIGHTS and hasattr(st, "proportion"):
            ov["proportion"] = _HEIGHT_WEIGHTS[name]
        if "stairs" in name and hasattr(st, "step_height_range"):
            ov["step_height_range"] = (_STEP_FLOOR, st.step_height_range[1])
        if name == "random_rough" and hasattr(st, "downsampled_scale"):
            ov["downsampled_scale"] = 0.25 * ts
        reweighted[name] = replace(st, **ov) if ov else st
    gen.sub_terrains = reweighted
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.sim.nconmax = None

    # ── Forward-biased terrain traversal commands ─────────────────────────────
    twist = cfg.commands["twist"]
    s_sqrt = ctx.s_sqrt
    twist.ranges.lin_vel_x = (0.35 * s_sqrt, 0.75 * s_sqrt)
    twist.ranges.lin_vel_y = (-0.20 * s_sqrt, 0.20 * s_sqrt)
    twist.ranges.ang_vel_z = (-0.40 / s_sqrt, 0.40 / s_sqrt)
    twist.resampling_time_range = (10.0 * s_sqrt, 20.0 * s_sqrt)
    twist.rel_standing_envs = 0.0

    # ── Climbing-friendly reward shaping ──────────────────────────────────────
    cfg.rewards["lin_vel_z"].func = _lin_vel_z_downward_l2
    cfg.rewards["lin_vel_z"].weight = -0.5 / ts
    if "foot_clearance" in cfg.rewards:
        cfg.rewards["foot_clearance"].weight = -0.40 / (ts**1.5)
        cfg.rewards["foot_clearance"].params["target_height"] = 0.12 * ts

    # ── Robust rough-terrain terminations ─────────────────────────────────────
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=mdp.out_of_terrain_bounds
    )
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=_hexapod_terrain_base_too_low,
        params={"minimum_height": 0.35 * ctx.init_height,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES)},
    )

    if not play:
        cfg.curriculum["terrain_levels"] = ctx.base_curriculum["terrain_levels"]
    else:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.terminations.pop("out_of_terrain_bounds", None)

    return cfg
