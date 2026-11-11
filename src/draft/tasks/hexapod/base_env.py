"""Shared base environment for the sprawled hexapod locomotion tasks.

Flat/base, push, terrain and velocity tasks all build on ``make_base_env_cfg``.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise

import draft.tasks.quadruped.mdp as q_mdp
from draft.tasks.quadruped.base_env import (
    TERRAIN_MIXES,
    DEFAULT_TERRAIN_MIX,
    TERRAIN_SCAN_FIXES,
    apply_terrain_ceiling,
    apply_terrain_mix,
    _lin_vel_z_l2,
)
from draft.tasks.hexapod.hexapod_entity import (
    KP_PER_NM,
    REF_LEG_REACH,
    make_hexapod_entity_cfg,
    measure_foot_sphere_radius,
    measure_max_body_speed,
    measure_max_joint_speed,
    measure_hexapod_reach,
    measure_standing_height,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
_FOOT_GEOMS = tuple(f"{leg}_tibia_link_sphere" for leg in _LEGS)
_FOOT_SITES = tuple(f"{leg}_foot" for leg in _LEGS)
_UPPER_LEG_BODIES = tuple(f"{leg}_femur_link" for leg in _LEGS)

# Action scale: ~25% of the motor saturation angle
_ACTION_SCALE = 0.25 / KP_PER_NM

# Gentle terrain ceiling for the base
_BASE_STEP_H, _BASE_NOISE, _BASE_WAVE = 0.12, 0.08, 0.14
_BASE_TERRAIN_LEVEL = 3
_BASE_PUSH_SCALE = 1.0
_VIEW_DISTANCE_PER_LEG = 3.0
_JOINT_VEL_NOISE_FRAC = 0.0104
_FOOT_SCAN_RING_M = 0.025


class BaseCtx:
    """Task-geometry context object for hexapod."""

    def __init__(self, robot_xml: Path | str, task_geometry_xml: Path | str | None = None,
                 v_max_override: float | None = None):
        geom_xml = task_geometry_xml if task_geometry_xml is not None else robot_xml
        self.reach = measure_hexapod_reach(geom_xml)
        self.leg_length = self.reach
        self.s = self.reach / REF_LEG_REACH
        self.s_sqrt = math.sqrt(self.s)
        self.v_max = measure_max_body_speed(geom_xml)
        self.omega_nl = measure_max_joint_speed(geom_xml)
        self.foot_radius = measure_foot_sphere_radius(geom_xml)
        self.hfield_scale = max(0.1, 2.0 * self.foot_radius)
        self.init_height = measure_standing_height(geom_xml)
        if v_max_override is not None:
            self.v_max = float(v_max_override)
        self.terrain_mix = DEFAULT_TERRAIN_MIX
        self.command_clamped = False
        self.dimensional_weights = False
        self.terrain_scan_fix = "all"
        self.episode_length_s = 0.0
        self.joint_vel_noise = 0.0
        s_sqrt = self.s_sqrt
        self.push_base_range: dict[str, tuple[float, float]] = {
            "x":     (-0.5 * s_sqrt, 0.5 * s_sqrt),
            "y":     (-0.5 * s_sqrt, 0.5 * s_sqrt),
            "z":     (-0.3 * s_sqrt, 0.3 * s_sqrt),
            "roll":  (-0.50 / s_sqrt, 0.50 / s_sqrt),
            "pitch": (-0.50 / s_sqrt, 0.50 / s_sqrt),
            "yaw":   (-0.80 / s_sqrt, 0.80 / s_sqrt),
        }


def make_base_env_cfg(
    *,
    robot_xml: Path | str,
    n_envs: int = 4096,
    episode_length_s: float = 20.0,
    play: bool = False,
    task_geometry_xml: Path | str | None = None,
    terrain_mix: str = DEFAULT_TERRAIN_MIX,
    clamp_command_to_vmax: bool = False,
    v_max_override: float | None = None,
    terrain_scale_with_size: bool = False,
    dimensional_weights: bool = False,
    terrain_scan_fix: str = "all",
) -> tuple[ManagerBasedRlEnvCfg, BaseCtx]:
    """Base environment configuration for hexapod locomotion."""
    if terrain_scan_fix not in TERRAIN_SCAN_FIXES:
        raise ValueError(f"unknown terrain_scan_fix {terrain_scan_fix!r}; known: {TERRAIN_SCAN_FIXES}")
    ctx = BaseCtx(robot_xml, task_geometry_xml, v_max_override)
    ctx.terrain_scan_fix = terrain_scan_fix
    ts = ctx.s if terrain_scale_with_size else 1.0
    s, s_sqrt = ctx.s, ctx.s_sqrt

    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": make_hexapod_entity_cfg(
        robot_xml, init_height=ctx.init_height, stance_xml=task_geometry_xml,
        geom_group=None if terrain_scan_fix == "none" else 1)}

    # ── Terrain ───────────────────────────────────────────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    gen = replace(ROUGH_TERRAINS_CFG)
    gen.curriculum = True
    gen.sub_terrains = apply_terrain_mix({
        name: apply_terrain_ceiling(st, _BASE_STEP_H * ts, _BASE_NOISE * ts,
                                    _BASE_WAVE * ts, ctx.hfield_scale)
        for name, st in gen.sub_terrains.items()
    }, terrain_mix)
    ctx.terrain_mix = terrain_mix
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = _BASE_TERRAIN_LEVEL
    cfg.sim.nconmax = None

    # ── Sensors ───────────────────────────────────────────────────────────────
    for sensor in cfg.scene.sensors:
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "torso_link"
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s_, entity="robot") for s_ in _FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(
                radius=_FOOT_SCAN_RING_M, num_samples=4)
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=_FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True,
    )
    upper_leg_ground_cfg = ContactSensorCfg(
        name="upper_leg_ground_contact",
        primary=ContactMatch(mode="body", pattern=_UPPER_LEG_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",), reduce="none", num_slots=1,
    )
    existing = tuple(s_ for s_ in cfg.scene.sensors
                     if s_.name not in {"feet_ground_contact", "upper_leg_ground_contact",
                                        "torso_ground_contact"})
    cfg.scene.sensors = existing + (feet_ground_cfg, upper_leg_ground_cfg)
    upright_scan = "terrain_scan"
    if terrain_scan_fix == "reward":
        scan = next(s_ for s_ in cfg.scene.sensors if s_.name == "terrain_scan")
        scan.include_geom_groups = (0, 1)
        cfg.scene.sensors = cfg.scene.sensors + (
            replace(scan, name="terrain_scan_ground", include_geom_groups=(0,),
                    debug_vis=False),)
        upright_scan = "terrain_scan_ground"

    _jv = _JOINT_VEL_NOISE_FRAC * ctx.omega_nl
    for group in cfg.observations.values():
        if "joint_vel" in group.terms:
            group.terms["joint_vel"].noise = Unoise(n_min=-_jv, n_max=_jv)
    ctx.joint_vel_noise = _jv

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = _ACTION_SCALE

    # ── Velocity commands (omnidirectional twist) ─────────────────────────────
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-0.8 * s_sqrt, 0.8 * s_sqrt)
    twist.ranges.lin_vel_y = (-0.8 * s_sqrt, 0.8 * s_sqrt)
    twist.ranges.ang_vel_z = (-1.0 / s_sqrt, 1.0 / s_sqrt)
    command_scale = 1.0
    if clamp_command_to_vmax:
        v = ctx.v_max
        wide = twist.ranges.lin_vel_x[1]
        twist.ranges.lin_vel_x = (max(-v, twist.ranges.lin_vel_x[0]),
                                  min(v, twist.ranges.lin_vel_x[1]))
        twist.ranges.lin_vel_y = (max(-v, twist.ranges.lin_vel_y[0]),
                                  min(v, twist.ranges.lin_vel_y[1]))
        command_scale = twist.ranges.lin_vel_x[1] / wide if wide > 0 else 1.0
    ctx.command_clamped = clamp_command_to_vmax
    ctx.dimensional_weights = dimensional_weights
    ctx.command_scale = command_scale

    # ── Rewards: tripod / wave / hexapod locomotion ───────────────────────────
    cfg.rewards["track_linear_velocity"].weight = 1.0
    cfg.rewards["track_linear_velocity"].params["std"] = (
        math.sqrt(0.25) * s_sqrt * command_scale)
    cfg.rewards["track_angular_velocity"].weight = 1.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.25) / s_sqrt

    cfg.rewards["upright"].weight = 1.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.2) / s_sqrt
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["upright"].params["terrain_sensor_names"] = (upright_scan,)

    # Hexapod steps (tripod/ripple/wave): shorter swing duty factor
    cfg.rewards["air_time"].weight = 0.4 / (s_sqrt if dimensional_weights else 1.0)
    cfg.rewards["air_time"].params["sensor_name"] = "feet_ground_contact"
    cfg.rewards["air_time"].params["threshold_min"] = 0.0
    cfg.rewards["air_time"].params["threshold_max"] = 0.25 * s_sqrt
    cfg.rewards["air_time"].params["command_threshold"] = 0.1 * s_sqrt

    cfg.rewards["foot_clearance"].weight = -2.0 / s**1.5
    cfg.rewards["foot_clearance"].params["target_height"] = 0.08 * s
    cfg.rewards["foot_clearance"].params["height_sensor_name"] = "foot_height_scan"
    cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = _FOOT_SITES

    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.05 * (s if dimensional_weights else 1.0)
    cfg.rewards["lin_vel_z"] = RewardTermCfg(func=_lin_vel_z_l2, weight=-2.0 / s)
    cfg.rewards["upper_leg_contact"] = RewardTermCfg(
        func=mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": "upper_leg_ground_contact"},
    )
    cfg.rewards["base_height"] = RewardTermCfg(
        func=q_mdp.base_height_exp, weight=0.4,
        params={"target_height": 0.95 * ctx.init_height, "std": 0.25 * ctx.init_height},
    )
    for key in ("pose", "angular_momentum", "body_ang_vel",
                "foot_swing_height", "foot_slip", "soft_landing"):
        cfg.rewards.pop(key, None)

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": 1.0},
    )
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=q_mdp.root_height_above_feet_below,
        params={"minimum_height": 0.45 * ctx.init_height,
                "asset_cfg": SceneEntityCfg("robot", site_names=_FOOT_SITES)},
    )

    # ── Events / Push ─────────────────────────────────────────────────────────
    ctx.base_curriculum = dict(cfg.curriculum)
    cfg.events["foot_friction"] = EventTermCfg(
        mode="startup", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=_FOOT_GEOMS),
                "operation": "abs", "ranges": (0.4, 1.1), "shared_random": True},
    )
    cfg.events["base_com"] = EventTermCfg(
        mode="startup", func=dr.body_com_offset,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.02 * s, 0.02 * s), 1: (-0.02 * s, 0.02 * s),
                           2: (-0.02 * s, 0.02 * s)}},
    )
    ctx.modest_push_range = {k: (v[0] * _BASE_PUSH_SCALE, v[1] * _BASE_PUSH_SCALE)
                             for k, v in ctx.push_base_range.items()}
    cfg.events["push_robot"].params["velocity_range"] = ctx.modest_push_range
    cfg.events["push_robot"].interval_range_s = (2.0 * s_sqrt, 5.0 * s_sqrt)
    ctx.push_event = cfg.events["push_robot"]

    cfg.curriculum = {}

    # ── Viewer ────────────────────────────────────────────────────────────────
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
    cfg.viewer.entity_name = "robot"
    cfg.viewer.body_name = "torso_link"
    cfg.viewer.distance = _VIEW_DISTANCE_PER_LEG * ctx.reach
    cfg.viewer.elevation = -30.0
    cfg.viewer.azimuth = 120.0
    cfg.scene.num_envs = n_envs
    cfg.episode_length_s = episode_length_s * s_sqrt
    ctx.episode_length_s = cfg.episode_length_s
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
    else:
        cfg.episode_length_s = ctx.episode_length_s

    return cfg, ctx
