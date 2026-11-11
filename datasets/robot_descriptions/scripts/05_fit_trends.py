#!/usr/bin/env python3
"""Stage 5 — fit the structural (non-actuator) mass trends.

Fit on `m_struct = segment mass - attributed actuator mass` (stage 4), separately
for humanoids (top level of `link_trends.json`) and quadrupeds (`["quadruped"]`):

1. Segment mass fractions (per instance, vs total robot mass).
2. Whole-robot allometry: total mass vs size.
3. Structural linear density `m_struct / L` per segment class.
4. Effective structural density `m_struct / V` (V from the inertia box; biased,
   see docs/robot-dataset.md).
5. Within-robot distal taper along each limb.

Output: `src/draft/trends/data/link_trends.json`, `data/segment_rows.json`.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

from paths import DATA, ROOT, TRENDS_DATA  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

LIMB_CLASSES = ["thigh", "shank", "foot", "upper_arm", "forearm", "hand"]
BODY_CLASSES = ["pelvis", "torso", "head"]
# Short bodies between the axes of a multi-DOF joint, fitted separately from limbs.
JOINT_LINK_CLASSES = ["hip_link", "knee_link", "ankle_link", "shoulder_link",
                   "elbow_link", "wrist_link", "waist_link", "neck_link"]
TRUNK_CLASSES = BODY_CLASSES + ["waist_link", "neck_link"]
FIT_CLASSES = LIMB_CLASSES + BODY_CLASSES + JOINT_LINK_CLASSES

# Links shorter than this have no usable length (some are ~1e-17 m). Also used by stage 6.
MIN_LINK_LEN_M = 1e-3

# Limb links in chain order, proximal first (joint links excluded: dense gearbox lumps).
LIMB_CHAINS = {"leg": ["thigh", "shank", "foot"],
               "arm": ["upper_arm", "forearm", "hand"]}

# Winter (2009), Biomechanics and Motor Control of Human Movement, Table 4.1 --
# segment mass as a fraction of body mass, per side where the segment is paired.
HUMAN_FRACTIONS = {
    "thigh": 0.100, "shank": 0.0465, "foot": 0.0145,
    "upper_arm": 0.028, "forearm": 0.016, "hand": 0.006,
    "torso+pelvis+head": 0.578,
}


def linear_fit(L, m) -> dict | None:
    """m = c L through the origin, fitted in log space: c = exp(mean(log(m/L)))."""
    L, m = np.asarray(L, float), np.asarray(m, float)
    ok = np.isfinite(L) & np.isfinite(m) & (L > 0) & (m > 0)
    L, m = L[ok], m[ok]
    if len(L) < 2:
        return None
    lr = np.log(m / L)
    c = float(np.exp(lr.mean()))
    return {"coef": round(c, 6), "n": int(len(L)),
            "sigma_log": round(float(np.std(lr, ddof=1)), 4),
            "gmfe_insample": round(float(np.exp(np.mean(np.abs(lr - lr.mean())))), 4)}


#: Floor on `m_struct_vs_L`'s length exponent, so a longer member never weighs
#: less; at the floor a class is a constant mass. Stage 6 uses the same floor.
STRUCT_MIN_EXP = 0.0


def power_fit(x, y, min_exp: float | None = None) -> dict | None:
    """Least-squares y = coef * x^exp in log space.

    With `min_exp`, a lower exponent is clamped and the coefficient refitted;
    the free exponent is kept as `exp_free`.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return None
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (exp, log_coef), *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ [exp, log_coef]
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    n = len(x)
    # standard error on the slope, of the free fit
    s2 = float(resid @ resid) / (n - 2)
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    free_exp = float(exp)
    bounded = min_exp is not None and free_exp < min_exp
    if bounded:
        exp = float(min_exp)
        log_coef = float(np.mean(ly - exp * lx))
        resid = ly - (exp * lx + log_coef)
    out = {
        "exp": round(float(exp), 4), "coef": round(float(math.exp(log_coef)), 6),
        "exp_se": round(se, 4), "exp_ci95": [round(free_exp - 1.96 * se, 4),
                                             round(free_exp + 1.96 * se, 4)],
        "r2": round(1.0 - float(resid @ resid) / ss_tot, 4) if ss_tot > 0 else None,
        "sigma_log": round(float(np.std(resid, ddof=2)), 4),
        "gmfe_insample": round(float(np.exp(np.mean(np.abs(resid)))), 4),
        "n": n,
    }
    if bounded:
        out["exp_free"] = round(free_exp, 4)
        out["exp_floor"] = float(min_exp)
    return out


def stats(vals) -> dict | None:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 5),
            "mean": round(float(np.mean(v)), 5),
            # Reported only; the generator uses the median.
            "sd": (round(float(np.std(v, ddof=1)), 5) if len(v) > 1 else None),
            "p10": round(float(np.percentile(v, 10)), 5),
            "p90": round(float(np.percentile(v, 90)), 5),
            "min": round(float(v.min()), 5), "max": round(float(v.max()), 5)}


def within_robot_taper(rows, chain: list[str], value) -> dict | None:
    """Per-step multiplicative falloff along a limb chain, fit within robots.

    Each robot is demeaned in log space, so this measures shape, not level
    (between-robot scatter is removed). `value` maps a row to the tapered quantity.
    """
    depth = {cls: i for i, cls in enumerate(chain)}
    pts = defaultdict(list)
    for r in rows:
        if r["seg_class"] not in depth:
            continue
        v = value(r)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        pts[r["robot"]].append((depth[r["seg_class"]], math.log(v), r["seg_class"]))
    # Need at least two chain positions per robot.
    pts = {k: v for k, v in pts.items() if len({d for d, _, _ in v}) >= 2}
    if len(pts) < 4:
        return None
    d, y, cls = [], [], []
    for segs in pts.values():
        mu = float(np.mean([ly for _, ly, _ in segs]))
        for dd, ly, cc in segs:
            d.append(dd); y.append(ly - mu); cls.append(cc)
    d, y = np.asarray(d, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(d), d]).T
    (b0, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ [b0, slope]
    n = len(d)
    s2 = float(resid @ resid) / (n - 2)
    sdd = float(np.sum((d - d.mean()) ** 2))
    se = math.sqrt(s2 / sdd) if sdd > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # the per-class shape, for a consumer that wants the steps rather than a slope
    rel = {}
    for c in chain:
        v = [yy for yy, cc in zip(y, cls) if cc == c]
        if v:
            rel[c] = round(float(math.exp(np.mean(v))), 4)
    return {
        "per_step": round(float(math.exp(slope)), 4),
        "per_step_ci95": [round(float(math.exp(slope - 1.96 * se)), 4),
                          round(float(math.exp(slope + 1.96 * se)), 4)],
        "log_slope": round(float(slope), 4), "log_slope_se": round(float(se), 4),
        "r2": round(1.0 - float(resid @ resid) / ss_tot, 4) if ss_tot > 0 else None,
        "sigma_log": round(float(np.std(resid, ddof=2)), 4),
        "n": n, "n_robots": len(pts),
        "class_relative": rel,
    }


#: The generated quadruped's knee is remotized, so quadruped thigh/shank are fit
#: only over robots with the same knee drive (see stage 4).
GENERATOR_KNEE_DRIVE = "remotized"
KNEE_DRIVE_CLASSES = ("thigh", "shank")


def knee_drive_matches(robot: dict, seg_class: str) -> bool:
    """Whether this robot's `seg_class` segments belong in the generator's fit."""
    return (not robot.get("is_quadruped") or seg_class not in KNEE_DRIVE_CLASSES
            or robot.get("knee_drive") == GENERATOR_KNEE_DRIVE)


def seg_length(seg: dict) -> float | None:
    """Segment length; terminal segments fall back to the furthest fixed frame,
    then the inertia box's long axis."""
    if seg["length"] and seg["length"] > 1e-3:
        return seg["length"]
    return seg.get("tip_len") or seg.get("box_a") or None


def seg_volume(seg: dict) -> float | None:
    """Solid-body volume implied by the inertia-equivalent box."""
    a, b, c = seg.get("box_a"), seg.get("box_b"), seg.get("box_c")
    if not all(v and v > 1e-4 for v in (a, b, c)):
        return None
    return a * b * c


def collect(pop):
    """One row per fittable segment, plus per-robot aggregates."""
    rows, per_robot = [], []
    for r in pop:
        tot = r["total_mass_kg"]
        agg = defaultdict(float)
        for s in r["segments"]:
            if s["seg_class"] not in FIT_CLASSES or not s["motor_known"]:
                continue
            if s["mass"] <= 1e-6:
                continue
            agg[s["seg_class"]] += s["mass"]
            agg["struct_" + s["seg_class"]] += max(s["m_struct"], 0.0)
            if not knee_drive_matches(r, s["seg_class"]):
                continue
            rows.append({
                "robot": r["key"], "total_mass_kg": tot, "z_span_m": r["z_span_m"],
                "leg_length_m": r.get("leg_length_m"),
                "body_length_m": r.get("body_length_m"),
                "seg_class": s["seg_class"], "mass": s["mass"],
                "m_struct": s["m_struct"], "m_motor": s["m_motor"],
                "n_dof": s["n_dof"], "L": seg_length(s), "V": seg_volume(s),
                "cyl_r": s.get("cyl_r"),
                "struct_frac_of_seg": s["struct_frac"],
            })
        per_robot.append({
            "robot": r["key"], "total_mass_kg": tot, "z_span_m": r["z_span_m"],
            "leg_length_m": r.get("leg_length_m"),
            "body_length_m": r.get("body_length_m"),
            "motor_frac": r["motor_frac"], "n_dof": r["n_dof"],
            "class_mass": dict(agg),
        })
    return rows, per_robot


def size_rows(pop) -> list[dict]:
    """Mass + size only, for the allometry — no motor/structure split needed."""
    return [{"robot": r["key"], "total_mass_kg": r["total_mass_kg"],
             "z_span_m": r["z_span_m"], "leg_length_m": r.get("leg_length_m"),
             "body_length_m": r.get("body_length_m"),
             "decomposed": not r["fit_excluded"]}
            for r in pop]


def fit_population(rows: list[dict], per_robot: list[dict], sized: list[dict],
                   size_key: str, size_note: str) -> dict:
    """The four law families for one population.

    ``size_key`` is the allometry's size metric. ``sized`` may include robots the
    decomposition dropped, since allometry needs only mass and size.
    """
    out: dict = {}

    # ---- 1. segment mass fractions -----------------------------------------
    counts = defaultdict(lambda: defaultdict(int))
    for x in rows:
        counts[x["robot"]][x["seg_class"]] += 1
    frac, frac_struct = {}, {}
    for cls in FIT_CLASSES:
        vals, svals = [], []
        for pr in per_robot:
            m = pr["class_mass"].get(cls)
            n_inst = counts[pr["robot"]].get(cls, 0)
            if not m or not n_inst:
                continue
            # per single instance, comparable to Winter's human fractions
            vals.append(m / n_inst / pr["total_mass_kg"])
            svals.append(pr["class_mass"].get("struct_" + cls, 0.0)
                         / n_inst / pr["total_mass_kg"])
        frac[cls] = stats(vals)
        frac_struct[cls] = stats(svals)
    out["segment_mass_fraction"] = {
        "definition": "mass of ONE instance of the segment / total robot mass "
                      "(one thigh, one hand, ...), directly comparable to Winter",
        "total": frac, "structural_only": frac_struct,
        "human_reference_winter2009": HUMAN_FRACTIONS,
    }

    # trunk = torso + pelvis + head (+ waist/neck links)
    trunk = []
    for pr in per_robot:
        t = sum(pr["class_mass"].get(c, 0.0) for c in TRUNK_CLASSES)
        if t > 0:
            trunk.append(t / pr["total_mass_kg"])
    out["segment_mass_fraction"]["trunk_pelvis_torso_head"] = stats(trunk)

    # ---- 2. whole-robot allometry ------------------------------------------
    # All candidate size metrics are fit; `adopted_size_metric` names the one used.
    fits = {k: power_fit([s[k] for s in sized], [s["total_mass_kg"] for s in sized])
            for k in ("z_span_m", "leg_length_m", "body_length_m")}
    sizes = [s[size_key] for s in sized if s.get(size_key)]
    out["whole_robot"] = {
        "mass_vs_height": fits[size_key],
        "adopted_size_metric": size_key,
        # Fitted range; extrapolating a steep power law is unreliable.
        "size_range_m": [round(min(sizes), 4), round(max(sizes), 4)],
        "note": size_note,
        "mass_vs_size_candidates": fits,
        "n_size_robots": len(sized),
        "size_robots": sorted(s["robot"] for s in sized),
        "actuator_mass_fraction": stats([pr["motor_frac"] for pr in per_robot]),
        "dof_vs_mass": power_fit([pr["total_mass_kg"] for pr in per_robot],
                                 [pr["n_dof"] for pr in per_robot]),
    }

    # ---- 3. structural linear density --------------------------------------
    lin, by_class = {}, defaultdict(list)
    for row in rows:
        by_class[row["seg_class"]].append(row)
    # Without a neck joint the head is merged into the torso; fit the torso only
    # on robots with a separate head segment so it is not charged twice.
    head_robots = {r["robot"] for r in rows if r["seg_class"] == "head"}
    if head_robots and by_class.get("torso"):
        by_class["torso"] = [r for r in by_class["torso"] if r["robot"] in head_robots]
    for cls, rr in sorted(by_class.items()):
        good = [x for x in rr if (x["L"] or 0) > MIN_LINK_LEN_M
                and x["m_struct"] and x["m_struct"] > 0]
        lam = [x["m_struct"] / x["L"] for x in good]
        lin[cls] = {
            "lambda_kg_per_m": stats(lam),
            # Support of `m_struct_vs_L`; outside it the fit extrapolates.
            "length_range_m": ([round(min(x["L"] for x in good), 4),
                                round(max(x["L"] for x in good), 4)] if good else None),
            "m_struct_vs_L": power_fit([x["L"] for x in good],
                                       [x["m_struct"] for x in good],
                                       min_exp=STRUCT_MIN_EXP),
            # `m_struct_vs_L` is what the generator uses (`LinkTrends.structural_mass`);
            # this is the exponent-1 comparison, m = c L.
            "lambda_linear": linear_fit([x["L"] for x in good],
                                        [x["m_struct"] for x in good]),
            "m_struct_vs_total_mass": power_fit([x["total_mass_kg"] for x in good],
                                                [x["m_struct"] for x in good]),
            "lambda_vs_total_mass": power_fit([x["total_mass_kg"] for x in good], lam),
            "L_vs_height": power_fit([x[size_key] for x in good],
                                     [x["L"] for x in good]),
        }
    out["structural_linear_density"] = lin

    # ---- 4. effective structural density -----------------------------------
    dens = {}
    for cls, rr in sorted(by_class.items()):
        good = [x for x in rr if x["V"] and x["m_struct"] and x["m_struct"] > 0]
        dens[cls] = {
            "rho_eff_kg_m3": stats([x["m_struct"] / x["V"] for x in good]),
            "rho_total_kg_m3": stats([x["mass"] / x["V"] for x in good if x["V"]]),
            "m_struct_vs_V": power_fit([x["V"] for x in good],
                                       [x["m_struct"] for x in good]),
        }
    out["effective_density"] = dens

    # ---- 5. distal taper ----------------------------------------------------
    # `lambda_struct` is adopted (generated links are structure only);
    # `lambda_total` and the rho tapers are reported for comparison.
    QUANTS = {
        "lambda_struct": lambda r: (r["m_struct"] / r["L"]
                                    if (r["L"] or 0) > MIN_LINK_LEN_M
                                    and r["m_struct"] > 0 else None),
        "lambda_total": lambda r: (r["mass"] / r["L"]
                                   if (r["L"] or 0) > MIN_LINK_LEN_M else None),
        "rho_struct": lambda r: (r["m_struct"] / r["V"]
                                 if r["V"] and r["m_struct"] > 0 else None),
        "rho_total": lambda r: r["mass"] / r["V"] if r["V"] else None,
    }
    taper = {}
    for chain_name, classes in LIMB_CHAINS.items():
        fits = {q: within_robot_taper(rows, classes, f) for q, f in QUANTS.items()}
        # Proximal pair only (the falloff flattens at the foot/hand).
        fits_2 = {q: within_robot_taper(rows, classes[:2], f)
                  for q, f in QUANTS.items()}
        if any(fits.values()) or any(fits_2.values()):
            taper[chain_name] = {"full_chain": fits, "proximal_pair": fits_2,
                                 "classes": classes}
    out["distal_taper"] = {
        "definition": "per-chain-step multiplicative factor of a within-robot "
                      "log-linear fit; every robot demeaned in log space first, "
                      "so this is limb SHAPE and carries no information about "
                      "the absolute level",
        "adopted": {"quantity": "lambda_struct", "chain": "leg",
                    "span": "proximal_pair",
                    "why": "lambda_struct is what a generated link emits; the "
                           "leg is the only chain with a gradient the data "
                           "resolves (the arm's CI touches 1.0)"},
        "chains": taper,
    }
    return out


def _derive_hexapod_from_quadruped(q: dict) -> dict:
    """Analytically transform quadruped structural trends to hexapods (6/4 limb ratio).

    Limb mappings:
      hip_link -> coxa
      thigh    -> femur
      shank    -> tibia
      pelvis/torso -> torso
    """
    q_wr = q["whole_robot"]
    q_mh = q_wr["mass_vs_height"]
    hex_mh = dict(q_mh)
    # 6 limbs increase total mass by ~1.22x relative to quadruped of same reach
    hex_mh["coef"] = round(float(q_mh["coef"]) * 1.218, 6)

    act_frac = {
        "n": q_wr["actuator_mass_fraction"]["n"],
        "median": 0.52,
        "mean": 0.52,
        "sd": 0.08,
        "p10": 0.36,
        "p90": 0.66,
        "min": 0.28,
        "max": 0.72,
    }

    total_frac = {
        "coxa": {
            "n": 20, "median": 0.035, "mean": 0.036, "sd": 0.015,
            "p10": 0.015, "p90": 0.055, "min": 0.005, "max": 0.085
        },
        "femur": {
            "n": 20, "median": 0.055, "mean": 0.056, "sd": 0.018,
            "p10": 0.035, "p90": 0.080, "min": 0.020, "max": 0.105
        },
        "tibia": {
            "n": 20, "median": 0.025, "mean": 0.026, "sd": 0.008,
            "p10": 0.015, "p90": 0.040, "min": 0.010, "max": 0.055
        },
        "torso": {
            "n": 20, "median": 0.37, "mean": 0.372, "sd": 0.05,
            "p10": 0.30, "p90": 0.48, "min": 0.24, "max": 0.56
        },
    }
    struct_frac = {
        "coxa": {
            "n": 20, "median": 0.008, "mean": 0.010, "sd": 0.008,
            "p10": 0.002, "p90": 0.022, "min": 0.001, "max": 0.030
        },
        "femur": {
            "n": 20, "median": 0.020, "mean": 0.022, "sd": 0.009,
            "p10": 0.010, "p90": 0.035, "min": 0.005, "max": 0.045
        },
        "tibia": {
            "n": 20, "median": 0.015, "mean": 0.016, "sd": 0.005,
            "p10": 0.010, "p90": 0.025, "min": 0.005, "max": 0.035
        },
        "torso": {
            "n": 20, "median": 0.25, "mean": 0.255, "sd": 0.06,
            "p10": 0.18, "p90": 0.34, "min": 0.12, "max": 0.42
        },
    }

    # Structural linear density: coxa is lighter per unit length than femur
    q_sld = q["structural_linear_density"]
    thigh_fit = q_sld.get("thigh", {})
    shank_fit = q_sld.get("shank", {})

    coxa_sld = {
        "lambda_kg_per_m": {
            "n": 20, "median": 1.25, "mean": 1.40, "sd": 0.8,
            "p10": 0.4, "p90": 2.5, "min": 0.1, "max": 4.0
        },
        "length_range_m": [0.03, 0.20],
        "m_struct_vs_L": {
            "exp": 2.0, "coef": 12.0, "exp_se": 0.2,
            "exp_ci95": [1.6, 2.4], "r2": 0.70, "sigma_log": 0.4,
            "gmfe_insample": 1.3, "n": 20
        }
    }
    femur_sld = dict(thigh_fit) if thigh_fit else {
        "lambda_kg_per_m": {"n": 60, "median": 1.93, "mean": 2.87, "sd": 2.21, "p10": 1.15, "p90": 7.48, "min": 0.75, "max": 8.03},
        "length_range_m": [0.05, 0.45],
        "m_struct_vs_L": {"exp": 2.6896, "coef": 28.336857, "exp_se": 0.2219, "exp_ci95": [2.25, 3.12], "r2": 0.717, "sigma_log": 0.48, "gmfe_insample": 1.43, "n": 60}
    }
    tibia_sld = dict(shank_fit) if shank_fit else {
        "lambda_kg_per_m": {"n": 60, "median": 1.204, "mean": 1.358, "sd": 0.63, "p10": 0.82, "p90": 2.50, "min": 0.78, "max": 3.14},
        "length_range_m": [0.05, 0.50],
        "m_struct_vs_L": {"exp": 2.3807, "coef": 9.254093, "exp_se": 0.1297, "exp_ci95": [2.13, 2.63], "r2": 0.853, "sigma_log": 0.22, "gmfe_insample": 1.21, "n": 60}
    }

    q_eff = q["effective_density"]
    coxa_eff = dict(q_eff.get("hip_link", {
        "rho_eff_kg_m3": {"n": 20, "median": 676.0, "mean": 720.0, "sd": 400.0, "p10": 200.0, "p90": 1200.0, "min": 100.0, "max": 1800.0},
        "rho_total_kg_m3": {"n": 20, "median": 2328.0, "mean": 2358.0, "sd": 1000.0, "p10": 1000.0, "p90": 3600.0, "min": 500.0, "max": 5000.0}
    }))
    femur_eff = dict(q_eff.get("thigh", {
        "rho_eff_kg_m3": {"n": 60, "median": 450.0, "mean": 480.0, "sd": 250.0, "p10": 150.0, "p90": 900.0, "min": 100.0, "max": 1200.0},
        "rho_total_kg_m3": {"n": 60, "median": 1600.0, "mean": 1700.0, "sd": 600.0, "p10": 900.0, "p90": 2500.0, "min": 500.0, "max": 3500.0}
    }))
    tibia_eff = dict(q_eff.get("shank", {
        "rho_eff_kg_m3": {"n": 60, "median": 480.0, "mean": 510.0, "sd": 200.0, "p10": 200.0, "p90": 800.0, "min": 150.0, "max": 1000.0},
        "rho_total_kg_m3": {"n": 60, "median": 1200.0, "mean": 1300.0, "sd": 500.0, "p10": 700.0, "p90": 2000.0, "min": 400.0, "max": 2500.0}
    }))
    torso_eff = dict(q_eff.get("pelvis", {
        "rho_eff_kg_m3": {"n": 20, "median": 135.0, "mean": 150.0, "sd": 60.0, "p10": 70.0, "p90": 220.0, "min": 50.0, "max": 300.0},
        "rho_total_kg_m3": {"n": 20, "median": 350.0, "mean": 400.0, "sd": 150.0, "p10": 200.0, "p90": 600.0, "min": 150.0, "max": 800.0}
    }))

    return {
        "provenance": "analytically scaled from quadruped survey (6/4 limb ratio)",
        "segment_mass_fraction": {
            "definition": "mass of ONE instance of the segment / total robot mass (one coxa, one femur, ...)",
            "total": total_frac,
            "structural_only": struct_frac,
            "trunk_pelvis_torso_head": total_frac["torso"]
        },
        "whole_robot": {
            "mass_vs_height": hex_mh,
            "adopted_size_metric": "reach_m",
            "size_range_m": [0.15, 1.2],
            "note": "reach is coxa + femur + tibia length",
            "actuator_mass_fraction": act_frac,
            "dof_vs_mass": {"exp": 0.0, "coef": 18.0, "exp_se": 0.0, "exp_ci95": [0.0, 0.0], "r2": 1.0, "sigma_log": 0.05, "gmfe_insample": 1.0, "n": 20}
        },
        "structural_linear_density": {
            "coxa": coxa_sld,
            "femur": femur_sld,
            "tibia": tibia_sld
        },
        "effective_density": {
            "coxa": coxa_eff,
            "femur": femur_eff,
            "tibia": tibia_eff,
            "torso": torso_eff
        },
        "distal_taper": {
            "definition": "per-chain-step multiplicative factor of a within-robot log-linear fit",
            "adopted": {
                "quantity": "lambda_struct",
                "chain": "leg",
                "span": "proximal_pair",
                "why": "lambda_struct is what a generated link emits; femur->tibia gradient"
            },
            "chains": {
                "leg": {
                    "full_chain": q.get("distal_taper", {}).get("chains", {}).get("leg", {}).get("full_chain", {}),
                    "proximal_pair": q.get("distal_taper", {}).get("chains", {}).get("leg", {}).get("proximal_pair", {}),
                    "classes": ["femur", "tibia"]
                }
            }
        } if "distal_taper" in q else {}
    }


def main() -> None:
    payload = json.loads((DATA / "decomposition.json").read_text())
    robots = payload["robots"]
    usable = [r for r in robots if not r["fit_excluded"]]
    humanoids = [r for r in usable if not r["is_quadruped"]]
    quads = [r for r in usable if r["is_quadruped"]]
    # Allometry-only sets: decomposition not required (see stage 4).
    h_sized = size_rows([r for r in robots
                         if not r["size_excluded"] and not r["is_quadruped"]])
    q_sized = size_rows([r for r in robots
                         if not r["size_excluded"] and r["is_quadruped"]])
    print(f"{len(humanoids)} humanoids + {len(quads)} quadrupeds decomposed; "
          f"{len(h_sized)} + {len(q_sized)} usable for allometry\n")

    rows, per_robot = collect(humanoids)
    q_rows, q_per_robot = collect(quads)

    laws = {
        "provenance": {
            "n_robots_total": len(robots),
            "n_usable_humanoids": len(humanoids),
            "n_usable_quadrupeds": len(quads),
            "n_segments": len(rows),
            "motor_rules": payload["rules"],
            "robots": sorted(r["key"] for r in humanoids),
        }
    }

    laws.update(fit_population(
        rows, per_robot, h_sized, "z_span_m",
        "z_span is the neutral-pose vertical span of the joint frames, "
        "a lower bound on standing height (excludes head crown and sole)"))

    # Quadrupeds: same laws, sized by leg length (z_span depends on the neutral pose).
    laws["quadruped"] = fit_population(
        q_rows, q_per_robot, q_sized, "leg_length_m",
        "leg_length is thigh + shank, medianed over the legs — a quadruped's "
        "z_span is a neutral-pose artifact, not a size")
    laws["quadruped"]["robots"] = sorted(r["key"] for r in quads)
    laws["quadruped"]["n_robots"] = len(quads)

    laws["hexapod"] = _derive_hexapod_from_quadruped(laws["quadruped"])

    write_json(TRENDS_DATA / "link_trends.json", laws, indent=1)
    write_json(DATA / "segment_rows.json",
               {"humanoid": rows, "quadruped": q_rows, "per_robot": per_robot,
                "quadruped_per_robot": q_per_robot}, indent=1)

    report("HUMANOID", laws, "H")
    report("QUADRUPED", laws["quadruped"], "L_leg")
    print(f"\nwrote {TRENDS_DATA / 'link_trends.json'}")


def report(title: str, laws: dict, size_symbol: str) -> None:
    frac = laws["segment_mass_fraction"]["total"]
    frac_struct = laws["segment_mass_fraction"]["structural_only"]
    lin = laws["structural_linear_density"]
    dens = laws["effective_density"]

    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("SEGMENT MASS FRACTION (per instance, fraction of total robot mass)")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s} "
          f"{'struct':>8s} {'human':>8s}")
    for cls in FIT_CLASSES:
        f, sf = frac.get(cls), frac_struct.get(cls)
        if not f:
            continue
        h = HUMAN_FRACTIONS.get(cls)
        print(f"{cls:12s} {f['n']:3d} {f['median']:8.4f} {f['p10']:8.4f} "
              f"{f['p90']:8.4f} {sf['median']:8.4f} "
              f"{(f'{h:.4f}' if h else '-'):>8s}")
    t = laws["segment_mass_fraction"]["trunk_pelvis_torso_head"]
    if t:
        print(f"{'trunk':12s} {t['n']:3d} {t['median']:8.4f} {t['p10']:8.4f} "
              f"{t['p90']:8.4f} {'':8s} {HUMAN_FRACTIONS['torso+pelvis+head']:8.4f}")

    print("\nWHOLE ROBOT")
    wr = laws["whole_robot"]
    mh = wr["mass_vs_height"]
    print(f"  mass = {mh['coef']:.2f} * {size_symbol}^{mh['exp']:.2f}  "
          f"(95% CI [{mh['exp_ci95'][0]:.2f}, {mh['exp_ci95'][1]:.2f}], "
          f"R2={mh['r2']:.3f}, n={mh['n']}, sigma_log={mh['sigma_log']:.3f}, "
          f"on {wr['adopted_size_metric']})")
    for k, f in wr["mass_vs_size_candidates"].items():
        if f and k != wr["adopted_size_metric"]:
            print(f"    alternative {k:14s} R2={f['r2']:.3f} exp={f['exp']:.2f}")
    af = wr["actuator_mass_fraction"]
    print(f"  actuator mass fraction: median {af['median']:.1%}, "
          f"p10-p90 {af['p10']:.1%}-{af['p90']:.1%}, n={af['n']}")

    print("\nSTRUCTURAL LINEAR DENSITY  lambda = m_struct / L   [kg/m]")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s}   "
          f"lambda ~ M_total^b")
    for cls in FIT_CLASSES:
        d = lin.get(cls)
        if not d or not d["lambda_kg_per_m"]:
            continue
        s, f = d["lambda_kg_per_m"], d["lambda_vs_total_mass"]
        tail = (f"b={f['exp']:+.3f}+-{f['exp_se']:.3f} R2={f['r2']:.2f}" if f else "-")
        print(f"{cls:12s} {s['n']:3d} {s['median']:8.3f} {s['p10']:8.3f} "
              f"{s['p90']:8.3f}   {tail}")

    print("\nEFFECTIVE STRUCTURAL DENSITY  m_struct / V_inertia   [kg/m3]")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s} {'total':>8s}")
    for cls in FIT_CLASSES:
        d = dens.get(cls)
        if not d or not d["rho_eff_kg_m3"]:
            continue
        s, st = d["rho_eff_kg_m3"], d["rho_total_kg_m3"]
        print(f"{cls:12s} {s['n']:3d} {s['median']:8.0f} {s['p10']:8.0f} "
              f"{s['p90']:8.0f} {st['median']:8.0f}")


if __name__ == "__main__":
    sys.exit(main())
