"""Kinematics, standing height, and speed limits for the sprawled hexapod.

Kept free of mjlab so unit tests run without a GPU stack.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np

#: Reference sprawled leg reach (m): nominal coxa + femur + tibia length.
REF_LEG_REACH = 0.475

#: Shared velocity task ceiling (m/s).
COMMON_V_CEILING = 15.0

#: Headroom on the speed-shaping scale.
SHAPE_HEADROOM = 1.5

#: Nominally arched sprawled posture angles (rad) for standing height calculations.
NOMINAL_FEMUR_ELEVATION_RAD = 0.35    # alpha > 0 elevates femur
NOMINAL_TIBIA_FLEXION_RAD = -1.45    # beta < -pi/2 flexes tibia downward


def _resolve_robot_xml(xml_path: Path) -> Path:
    """If xml_path is a scene file that includes a robot xml, resolve the robot xml."""
    text = xml_path.read_text()
    m = re.search(r'<include\s+file="([^"]+)"', text)
    if m:
        return xml_path.parent / m.group(1)
    return xml_path


def measure_hexapod_segments(xml_path: Path | str) -> tuple[float, float, float]:
    """Return (coxa_length, femur_length, tibia_length) in metres from the fl leg.

    Accepts scene XMLs.
    """
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    femur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fl_femur_link")
    tibia_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fl_tibia_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")
    if femur_id < 0 or tibia_id < 0 or foot_id < 0:
        raise ValueError(f"Hexapod leg bodies or fl_foot site not found in {xml_path}")

    coxa_l = float(np.linalg.norm(model.body_pos[femur_id]))
    femur_l = float(np.linalg.norm(model.body_pos[tibia_id]))
    tibia_l = float(np.linalg.norm(model.site_pos[foot_id]))
    return coxa_l, femur_l, tibia_l


def measure_hexapod_reach(xml_path: Path | str) -> float:
    """Return total radial reach = coxa_length + femur_length + tibia_length."""
    coxa_l, femur_l, tibia_l = measure_hexapod_segments(xml_path)
    return coxa_l + femur_l + tibia_l


def measure_standing_height(xml_path: Path | str,
                            alpha: float = NOMINAL_FEMUR_ELEVATION_RAD,
                            beta: float = NOMINAL_TIBIA_FLEXION_RAD) -> float:
    """Nominal sprawled standing height: z_init = L_tibia*sin(|beta| - alpha) - L_femur*sin(alpha) + r_foot."""
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fl_tibia_link_sphere")
    r_foot = float(model.geom_size[gid, 0]) if gid >= 0 else 0.018

    _coxa_l, femur_l, tibia_l = measure_hexapod_segments(xml_path)
    z_init = tibia_l * math.sin(abs(beta) - alpha) - femur_l * math.sin(alpha) + r_foot
    return max(z_init, 0.05)


#: Coxa sweep and Froude speed coefficients.
_COXA_SWEEP_EFFICIENCY = 0.45
_FROUDE_SPEED_COEF = 1.10


def estimate_top_speed(xml_path: Path | str) -> float:
    """Top body speed: min(coxa_yaw sweep speed, Froude limit)."""
    import yaml
    xml_path = _resolve_robot_xml(Path(xml_path))
    coxa_l, femur_l, tibia_l = measure_hexapod_segments(xml_path)
    reach = coxa_l + femur_l + tibia_l
    froude = _FROUDE_SPEED_COEF * math.sqrt(9.81 * reach)

    params_yaml = xml_path.parent / "parameters_from_gui.yaml"
    if not params_yaml.exists():
        params_yaml = xml_path.parent / "parameters_resolved.yaml"
    if not params_yaml.exists():
        params_yaml = xml_path.parent / "parameters.yaml"
    if not params_yaml.exists():
        return froude

    params = yaml.safe_load(params_yaml.read_text())
    coxa_cls = params.get("coxa_yaw_mot", "S")
    omega_coxa = float(params.get(f"{coxa_cls}_motor_velocity", 35.0))
    # Effective lever arm for lateral sweep in sprawled posture
    lever = coxa_l + femur_l * math.cos(NOMINAL_FEMUR_ELEVATION_RAD)
    v_sweep = omega_coxa * lever * _COXA_SWEEP_EFFICIENCY
    return min(v_sweep, froude)
