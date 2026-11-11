"""Rough-terrain challenge task for the sprawled hexapod."""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from draft.tasks.hexapod.base_env import make_base_env_cfg
from draft.tasks.hexapod.runner_cfg import make_runner_cfg  # noqa: F401

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
        reweighted[name] = replace(st, **ov) if ov else st
    gen.sub_terrains = reweighted
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.sim.nconmax = None

    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=mdp.out_of_terrain_bounds
    )

    if not play:
        cfg.curriculum["terrain_levels"] = ctx.base_curriculum["terrain_levels"]
    else:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.terminations.pop("out_of_terrain_bounds", None)

    return cfg
