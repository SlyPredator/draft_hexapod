"""Parametric hexapod EntityCfg factory.

Builds a 18-DOF radially mounted arthropod entity with DcMotorActuatorCfg groups,
foot contact spheres ({fl,fr,ml,mr,rl,rr}_tibia_link_sphere), and nominal arched sprawled stance.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path

import mujoco

from draft.tasks.hexapod.speed_limits import (  # noqa: F401
    COMMON_V_CEILING,
    NOMINAL_FEMUR_ELEVATION_RAD,
    NOMINAL_TIBIA_FLEXION_RAD,
    REF_LEG_REACH,
    _resolve_robot_xml,
    estimate_top_speed,
    measure_hexapod_reach,
    measure_hexapod_segments,
    measure_standing_height,
)

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")

# PD gain scaling — calibrated so 35 Nm gives kp=50, kd=2.5; 70 Nm gives kp=100, kd=5.
KP_PER_NM: float = 100.0 / 70.0
KD_PER_NM: float = 5.0 / 70.0

# Only the foot spheres collide; falls are detected from torso/state in the env.
_TERRAIN_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_tibia_link_sphere",),  # 6 foot contact spheres
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    disable_other_geoms=True,
)


def measure_max_body_speed(xml_path: Path | str) -> float:
    """Estimated top speed for command scaling."""
    return estimate_top_speed(xml_path)


def measure_max_joint_speed(xml_path: Path | str) -> float:
    """Largest no-load joint speed (rad/s) declared by the model."""
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    limits = _velocity_limits_from_xml(model)
    vals = [v for v in limits.values() if v > 0.0]
    return max(vals) if vals else 35.0


def measure_foot_sphere_radius(xml_path: Path | str) -> float:
    """Return the foot sphere radius from the fl_tibia_link_sphere geom."""
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fl_tibia_link_sphere")
    if gid < 0:
        raise ValueError(f"fl_tibia_link_sphere geom not found in {xml_path}")
    return float(model.geom_size[gid, 0])


_STANCE_HEADROOM: float = 0.15


def _soft_range(model: mujoco.MjModel, joint: str,
                factor: float = 0.9) -> tuple[float, float]:
    """The joint range the dof_pos_limits reward leaves usable."""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if j < 0:
        return -math.pi, math.pi
    lo, hi = (float(x) for x in model.jnt_range[j])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def default_stance(xml_path: Path | str,
                   stance_xml: Path | str | None = None) -> dict[str, float]:
    """The nominal sprawled stance: coxa_yaw = 0.0, femur_pitch = +0.35, tibia_pitch = -1.45.

    Clamped against each joint's soft range to guarantee safe headroom.
    """
    models = [mujoco.MjModel.from_xml_path(str(_resolve_robot_xml(Path(x))))
              for x in ({xml_path} if stance_xml is None else {xml_path, stance_xml})]

    stance = {
        ".*coxa_yaw": 0.0,
        ".*femur_pitch": NOMINAL_FEMUR_ELEVATION_RAD,
        ".*tibia_pitch": NOMINAL_TIBIA_FLEXION_RAD,
    }

    for model in models:
        for pat, val in list(stance.items()):
            role = pat.lstrip(".*")
            lo, hi = _soft_range(model, f"fl_{role}")
            clamped = float(min(max(val, lo + _STANCE_HEADROOM), hi - _STANCE_HEADROOM))
            stance[pat] = clamped

    return stance


def _build_robot_spec(xml_path: Path, geom_group: int | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for act in list(spec.actuators):
        spec.delete(act)
    if geom_group is not None:
        for geom in spec.geoms:
            geom.group = geom_group
    return spec


def _velocity_limits_from_xml(model: mujoco.MjModel) -> dict[str, float]:
    """Read per-joint no-load speed (omega_NL) from '<joint>_velocity_limit' numerics."""
    suffix = "_velocity_limit"
    out: dict[str, float] = {}
    for i in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        if name and name.endswith(suffix):
            adr = int(model.numeric_adr[i])
            out[name[: -len(suffix)]] = float(model.numeric_data[adr])
    return out


def _actuator_cfgs_from_xml(
    xml_path: Path,
) -> tuple[DcMotorActuatorCfg | BuiltinPositionActuatorCfg, ...]:
    """Derive DcMotorActuatorCfg groups (torque-speed model) from the robot XML."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    vel_limits = _velocity_limits_from_xml(model)

    joints: list[tuple[str, float]] = []
    if model.nu > 0:
        for i in range(model.nu):
            jnt_id = int(model.actuator_trnid[i, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if not name:
                continue
            joints.append((name, float(model.actuator_forcerange[i, 1])))
    else:
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not name:
                continue
            joints.append((name, float(model.jnt_actfrcrange[i, 1])))

    have_dc = bool(joints) and all(n in vel_limits and e > 0.0 for n, e in joints)

    if not have_dc:
        warnings.warn(
            "No-load speeds (omega_NL) missing or non-positive efforts in XML; "
            "falling back to position actuators.",
            UserWarning,
            stacklevel=2,
        )
        pos_groups: dict[tuple[float, float, float], list[str]] = defaultdict(list)
        for name, effort in joints:
            kp = round(KP_PER_NM * effort, 4)
            kd = round(KD_PER_NM * effort, 4)
            pos_groups[(kp, kd, effort)].append(name)
        return tuple(
            BuiltinPositionActuatorCfg(
                target_names_expr=tuple(names),
                stiffness=kp,
                damping=kd,
                effort_limit=effort if effort > 0.0 else None,
            )
            for (kp, kd, effort), names in pos_groups.items()
        )

    dc_groups: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
    for name, effort in joints:
        kp = round(KP_PER_NM * effort, 4)
        kd = round(KD_PER_NM * effort, 4)
        dc_groups[(kp, kd, effort, round(vel_limits[name], 4))].append(name)
    return tuple(
        DcMotorActuatorCfg(
            target_names_expr=tuple(names),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            saturation_effort=effort,
            velocity_limit=velocity,
        )
        for (kp, kd, effort, velocity), names in dc_groups.items()
    )


def make_hexapod_entity_cfg(
    xml_path: Path | str,
    init_height: float | None = None,
    stance_xml: Path | str | None = None,
    geom_group: int | None = None,
) -> EntityCfg:
    """Create an EntityCfg for the 18-DOF hexapod."""
    xml_path = _resolve_robot_xml(Path(xml_path))
    if init_height is None:
        init_height = measure_standing_height(xml_path)
    actuators = _actuator_cfgs_from_xml(xml_path)
    captured = xml_path

    return EntityCfg(
        spec_fn=lambda: _build_robot_spec(captured, geom_group),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_height),
            joint_pos=default_stance(xml_path, stance_xml),
            joint_vel={".*": 0.0},
        ),
        collisions=(_TERRAIN_COLLISION,),
    )
