"""Exhaustive unit test suite for the first-class radially symmetric sprawled hexapod."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from draft.editor.core.schema import RobotSchema
from draft.generation.generator import RobotGenerator
from draft.robots import available, robot_dir
from draft.tasks import TASKS, get_task
from draft.tasks.gait_metrics import (
    CANONICAL_HEXAPOD_GAITS,
    classify_hexapod_gait,
    contact_gait_metrics,
    joint_symmetry_metrics,
)
from draft.tasks.hexapod.speed_limits import (
    estimate_top_speed,
    measure_hexapod_reach,
    measure_hexapod_segments,
    measure_standing_height,
)


@pytest.fixture(scope="module")
def generated_hexapod(tmp_path_factory):
    """Generate base hexapod into a temp directory once for the module."""
    out = tmp_path_factory.mktemp("hexapod_base")
    RobotGenerator(robot_dir("hexapod")).generate(out)
    return out


def test_hexapod_available():
    """Hexapod is discoverable in available robots and has expected presets."""
    assert "hexapod" in available()
    preset_dir = robot_dir("hexapod") / "presets"
    presets = [p.stem for p in preset_dir.glob("*.yaml")]
    assert "tarantula" in presets
    assert "beetle" in presets
    assert "strider" in presets


def test_base_hexapod_compilation_and_topology(generated_hexapod):
    """Base hexapod compiles with 18 hinge joints, 6 foot sites, and 0 feasibility errors."""
    rep_file = generated_hexapod / "feasibility_report.yaml"
    assert rep_file.exists()
    rep = yaml.safe_load(rep_file.read_text())
    assert rep.get("errors") == [] or not rep.get("errors")

    xml_file = next(p for p in generated_hexapod.glob("*.xml") if p.name != "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_file))

    # Topology checks: 1 freejoint + 18 hinge joints
    jnt_types = [model.jnt_type[i] for i in range(model.njnt)]
    assert jnt_types.count(mujoco.mjtJoint.mjJNT_FREE) == 1
    assert jnt_types.count(mujoco.mjtJoint.mjJNT_HINGE) == 18

    # 6 foot sites
    expected_feet = ["fl_foot", "fr_foot", "ml_foot", "mr_foot", "rl_foot", "rr_foot"]
    for foot in expected_feet:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot)
        assert site_id >= 0, f"Missing site {foot}"

    # Verify 18 expected joints exist
    leg_prefixes = ["fl", "fr", "ml", "mr", "rl", "rr"]
    joint_suffixes = ["coxa_yaw", "femur_pitch", "tibia_pitch"]
    for pfx in leg_prefixes:
        for sfx in joint_suffixes:
            jname = f"{pfx}_{sfx}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"Missing joint {jname}"


def test_hexapod_presets_compile(tmp_path):
    """All 3 hexapod presets compile cleanly with 0 errors."""
    for preset in ("tarantula", "beetle", "strider"):
        out = tmp_path / preset
        preset_file = robot_dir("hexapod") / "presets" / f"{preset}.yaml"
        assert preset_file.exists()
        overrides = yaml.safe_load(preset_file.read_text())
        RobotGenerator(robot_dir("hexapod")).generate(out, params_override=overrides)
        xml_file = next(p for p in out.glob("*.xml") if p.name != "scene.xml")
        model = mujoco.MjModel.from_xml_path(str(xml_file))
        assert model.njnt == 19
        rep = yaml.safe_load((out / "feasibility_report.yaml").read_text())
        assert not rep.get("errors")


def test_hexapod_bilateral_symmetry(generated_hexapod):
    """Left-right leg pairs have exact mirror symmetry across Y=0 in neutral pose."""
    xml_file = next(p for p in generated_hexapod.glob("*.xml") if p.name != "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    pairs = [("fl", "fr"), ("ml", "mr"), ("rl", "rr")]
    for left_pfx, right_pfx in pairs:
        # Foot sites
        l_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{left_pfx}_foot")
        r_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{right_pfx}_foot")
        l_pos = data.site_xpos[l_site]
        r_pos = data.site_xpos[r_site]

        np.testing.assert_allclose(l_pos[0], r_pos[0], atol=1e-5, err_msg=f"{left_pfx}/{right_pfx} X mismatch")
        np.testing.assert_allclose(l_pos[1], -r_pos[1], atol=1e-5, err_msg=f"{left_pfx}/{right_pfx} Y mismatch")
        np.testing.assert_allclose(l_pos[2], r_pos[2], atol=1e-5, err_msg=f"{left_pfx}/{right_pfx} Z mismatch")

        # Femur body positions
        l_femur = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{left_pfx}_femur_link")
        r_femur = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{right_pfx}_femur_link")
        l_bpos = data.xpos[l_femur]
        r_bpos = data.xpos[r_femur]

        np.testing.assert_allclose(l_bpos[0], r_bpos[0], atol=1e-5)
        np.testing.assert_allclose(l_bpos[1], -r_bpos[1], atol=1e-5)
        np.testing.assert_allclose(l_bpos[2], r_bpos[2], atol=1e-5)


def test_hexapod_mass_composition(generated_hexapod):
    """Hexapod mass composition matches empirical hexapod population trends."""
    rep = yaml.safe_load((generated_hexapod / "feasibility_report.yaml").read_text())
    mc = rep.get("mass_composition_check", {})
    assert mc.get("reference_population") == "hexapod"

    checks = {c["quantity"]: c for c in mc.get("checks", [])}
    assert checks["actuator_mass_fraction"]["status"] == "ok"
    assert checks["trunk_mass_fraction"]["status"] == "ok"
    assert checks["total_mass_vs_size_kg"]["status"] == "ok"

    actuator_val = checks["actuator_mass_fraction"]["value"]
    trunk_val = checks["trunk_mass_fraction"]["value"]
    assert 0.40 <= actuator_val <= 0.65
    assert 0.25 <= trunk_val <= 0.50


def test_hexapod_speed_limits_and_standing_height(generated_hexapod):
    """Kinematics, standing height, and top speed calculations work on generated hexapod."""
    xml_file = next(p for p in generated_hexapod.glob("*.xml") if p.name != "scene.xml")
    coxa_l, femur_l, tibia_l = measure_hexapod_segments(xml_file)
    assert 0.05 < coxa_l < 0.20
    assert 0.10 < femur_l < 0.30
    assert 0.15 < tibia_l < 0.40

    reach = measure_hexapod_reach(xml_file)
    assert np.isclose(reach, coxa_l + femur_l + tibia_l)

    z_stand = measure_standing_height(xml_file)
    assert 0.10 <= z_stand <= 0.35

    top_spd = estimate_top_speed(xml_file)
    assert 1.0 <= top_spd <= 10.0


def test_hexapod_schema_parsing():
    """RobotSchema correctly parses hexapod parameters.yaml and classifies motor classes."""
    params = yaml.safe_load((robot_dir("hexapod") / "parameters.yaml").read_text())
    schema = RobotSchema(params)
    assert len(schema.specs) > 40
    assert schema.motor_classes == ("M", "S") or "M" in schema.motor_classes
    assert "coxa_length" in schema.specs
    assert "femur_length" in schema.specs
    assert "tibia_length" in schema.specs
    assert "body_radius" in schema.specs


def test_hexapod_gait_classification():
    """Canonical 6-foot gaits classify accurately."""
    # Exact canonical tripod
    name, dist = classify_hexapod_gait(CANONICAL_HEXAPOD_GAITS["tripod"])
    assert name == "tripod"
    assert dist == pytest.approx(0.0, abs=1e-6)

    # Exact canonical ripple
    name, dist = classify_hexapod_gait(CANONICAL_HEXAPOD_GAITS["ripple"])
    assert name == "ripple"
    assert dist == pytest.approx(0.0, abs=1e-6)

    # Exact canonical wave
    name, dist = classify_hexapod_gait(CANONICAL_HEXAPOD_GAITS["wave"])
    assert name == "wave"
    assert dist == pytest.approx(0.0, abs=1e-6)


def test_hexapod_contact_gait_metrics_tripod():
    """contact_gait_metrics correctly classifies a simulated 6-foot tripod gait."""
    T = 200
    B = 4
    F = 6
    dt = 0.02
    period_samples = 50  # 1.0 s stride

    contact = np.zeros((T, B, F), dtype=bool)
    # Tripod 1: fl (0), mr (3), rl (4) in stance for first half of cycle
    # Tripod 2: fr (1), ml (2), rr (5) in stance for second half of cycle
    for t in range(T):
        phase = (t % period_samples) / period_samples
        if phase < 0.5:
            contact[t, :, 0] = True
            contact[t, :, 3] = True
            contact[t, :, 4] = True
        else:
            contact[t, :, 1] = True
            contact[t, :, 2] = True
            contact[t, :, 5] = True

    metrics = contact_gait_metrics(contact, dt)
    assert metrics["gait_class"] == "tripod"
    assert metrics["gait_class_dist"] < 0.05
    assert metrics["gait_lr_phase_asymmetry"] < 0.05
    assert np.isclose(metrics["duty_factor"], 0.5, atol=0.05)


def test_hexapod_joint_symmetry_metrics():
    """joint_symmetry_metrics automatically detects the 3 bilateral pairs for hexapods."""
    joint_names = [
        "fl_coxa_yaw", "fl_femur_pitch", "fl_tibia_pitch",
        "fr_coxa_yaw", "fr_femur_pitch", "fr_tibia_pitch",
        "ml_coxa_yaw", "ml_femur_pitch", "ml_tibia_pitch",
        "mr_coxa_yaw", "mr_femur_pitch", "mr_tibia_pitch",
        "rl_coxa_yaw", "rl_femur_pitch", "rl_tibia_pitch",
        "rr_coxa_yaw", "rr_femur_pitch", "rr_tibia_pitch",
    ]
    T, B, J = 100, 2, 18
    # Symmetric joint trajectory
    q = np.sin(np.linspace(0, 4 * np.pi, T)[:, None, None] + np.zeros((1, B, J)))
    res = joint_symmetry_metrics(q, joint_names)
    assert "gait_lr_excursion_symmetry_index" in res
    assert res["gait_lr_excursion_symmetry_index"] == pytest.approx(0.0, abs=1e-5)


def test_hexapod_task_registry_and_cfg_instantiation(generated_hexapod):
    """All 4 hexapod tasks in task registry instantiate environment cfgs."""
    xml_path = str(next(p for p in generated_hexapod.glob("*.xml") if p.name != "scene.xml"))

    for task_name in ("hexapod", "hexapod_push", "hexapod_terrain", "hexapod_velocity"):
        assert task_name in TASKS
        entry = get_task(task_name)
        runner_cfg = entry.make_runner_cfg(seed=123)
        assert runner_cfg is not None
        from draft.tasks.hexapod.runner_cfg import HexapodLocomotionRunnerCfg
        assert isinstance(runner_cfg, HexapodLocomotionRunnerCfg)
        assert runner_cfg.run_name == "hexapod_locomotion"
        assert runner_cfg.wandb_project == "hexapod_locomotion"
        assert runner_cfg.seed == 123

        env_cfg = entry.make_env_cfg(play=True, robot_xml=xml_path)
        assert env_cfg is not None
        assert "robot" in env_cfg.scene.entities
        assert env_cfg.scene.entities["robot"] is not None

    # Verify speed estimation reads parameters_resolved.yaml
    assert (generated_hexapod / "parameters_resolved.yaml").exists()
    assert not (generated_hexapod / "parameters.yaml").exists()
    spd = estimate_top_speed(xml_path)
    assert spd > 0.0
