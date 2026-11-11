#!/usr/bin/env python3
"""Render the hexapod presets individually and as a lineup plate into docs/.

Renders:
  - docs/hexapod_tarantula.png (and docs/tarantula.png)
  - docs/hexapod_beetle.png (and docs/beetle.png)
  - docs/hexapod_strider.png (and docs/strider.png)
  - docs/hexapod_lineup.png (composite lineup plate with masses and actuator labels)

Usage:
  python scripts/figures/render_hexapods.py
  python scripts/figures/render_hexapods.py --designs tarantula beetle strider --mujoco # offscreen fallback
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np
from PIL import Image
import yaml

from draft.twins.render import (
    FRAME_MARGIN,
    PAIR_GAP,
    PairRenderer,
    _compose,
)
from draft.twins.targets import GENERATED_HEXAPOD_ROLES
from draft.twins.viser_render import ViserPairRenderer, whiten_ground

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
GENERATED = REPO / "generated"

HEXAPOD_DESIGNS = ["tarantula", "beetle", "strider"]

LABEL_RGB = {
    "tarantula": (240, 115, 15),
    "beetle": (50, 95, 55),
    "strider": (45, 120, 205),
}
DEFAULT_RGB = (60, 60, 60)


def get_label_color(name: str) -> tuple[int, int, int]:
    return LABEL_RGB.get(name, DEFAULT_RGB)


class HexapodRenderer(PairRenderer):
    """Renderer for hexapod designs individually or as a multi-robot plate."""

    @staticmethod
    def prefix(i: int) -> str:
        return f"lineup{i}/"

    def render_solo(
        self,
        name: str,
        xml_path: Path,
        out_path: Path,
        viser: bool = True,
        port: int = 8099,
    ) -> None:
        """Render a single hexapod in its sprawled stance with studio lighting."""
        spec = self._prepared(xml_path, role_map=GENERATED_HEXAPOD_ROLES)
        self._dress(spec, half=6.0, grid=3.0)

        model = spec.compile()
        data = mujoco.MjData(model)

        self._pose(model, data, GENERATED_HEXAPOD_ROLES, "hexapod")
        mujoco.mj_forward(model, data)

        c, h = self._extents(model, data, list(range(model.ngeom)))
        lo, _ = self._span(c, h, np.array([0.0, 0.0, 1.0]))
        free_adr = model.jnt_qposadr[0]
        data.qpos[free_adr + 2] += -lo + 0.001
        mujoco.mj_forward(model, data)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.0, data.qpos[free_adr + 2] * 0.70]
        cam.distance = 1.35
        cam.azimuth = 132.0
        cam.elevation = -22.0

        if viser:
            whiten_ground(model)
            with ViserPairRenderer(port=port, max_render=(self.width, self.height)) as vr:
                img = vr.render(model, data, cam, self.width, self.height)
        else:
            opt = mujoco.MjvOption()
            opt.geomgroup[3:] = 0
            with mujoco.Renderer(model, self.height, self.width) as r:
                r.update_scene(data, camera=cam, scene_option=opt)
                img = r.render().copy()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_path)
        print(f"  wrote {out_path}")

    def render_lineup_plate(
        self,
        designs: list[str],
        out_path: Path,
        title: str = "Hexapod design lineup",
        viser: bool = True,
        port: int = 8099,
    ) -> None:
        """Render the hexapods side-by-side in one scene at honest relative scale."""
        paths = [GENERATED / d / "hexapod.xml" for d in designs]
        spec = self._prepared(paths[0], role_map=GENERATED_HEXAPOD_ROLES)
        for i, path in enumerate(paths[1:], start=1):
            other = self._prepared(path, role_map=GENERATED_HEXAPOD_ROLES)
            frame = spec.worldbody.add_frame(pos=[0, 0, 0])
            spec.attach(other, prefix=self.prefix(i), frame=frame)

        k = max(1.0, len(paths) / 3.0)
        self._dress(spec, half=140.0 * k, grid=70.0 * k)
        model = spec.compile()
        data = mujoco.MjData(model)

        n = len(paths)
        prefixes = [self.prefix(i) for i in range(1, n)]
        root = np.zeros(model.nbody, dtype=int)
        for b in range(1, model.nbody):
            p = int(model.body_parentid[b])
            root[b] = b if p == 0 else root[p]

        def which(b: int) -> int:
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(root[b])) or ""
            return next((i for i, pref in enumerate(prefixes, start=1) if bname.startswith(pref)), 0)

        groups = [{"geoms": [], "joints": [], "free": None} for _ in range(n)]
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if b == 0:
                continue
            groups[which(b)]["geoms"].append(g)
        for j in range(model.njnt):
            grp = groups[which(int(model.jnt_bodyid[j]))]
            grp["joints"].append(j)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                grp["free"] = int(model.jnt_qposadr[j])

        # Pose each hexapod in nominal sprawled stance
        data.qpos[:] = model.qpos0
        for grp in groups:
            self._pose(model, data, GENERATED_HEXAPOD_ROLES, "hexapod", joints=grp["joints"])
        mujoco.mj_forward(model, data)

        # Drop each to floor
        for grp in groups:
            c, h = self._extents(model, data, grp["geoms"])
            lo, _ = self._span(c, h, np.array([0.0, 0.0, 1.0]))
            if grp["free"] is not None:
                data.qpos[grp["free"] + 2] += -lo + 0.002
        mujoco.mj_forward(model, data)

        # Space out laterally along camera right axis
        _, right, _ = self._basis()
        widths, mids = [], []
        for grp in groups:
            c, h = self._extents(model, data, grp["geoms"])
            lo, hi = self._span(c, h, right)
            widths.append(hi - lo)
            mids.append(0.5 * (lo + hi))

        gap = PAIR_GAP * float(np.mean(widths)) * 1.5
        total = sum(widths) + gap * (len(widths) - 1)
        want, cursor = [], -0.5 * total
        for w in widths:
            want.append(cursor + 0.5 * w)
            cursor += w + gap
        for grp, w, m in zip(groups, want, mids):
            if grp["free"] is not None:
                data.qpos[grp["free"]:grp["free"] + 2] += (w - m) * right[:2]
        mujoco.mj_forward(model, data)

        # Frame camera
        f, right, up = self._basis()
        every = [g for grp in groups for g in grp["geoms"]]
        pts = self._corners(model, data, every)
        tan_v = np.tan(np.radians(model.vis.global_.fovy) / 2.0) / FRAME_MARGIN
        tan_h = tan_v * (self.width / self.height)
        cam_pts = np.column_stack([pts @ right, pts @ up, pts @ f])

        look = cam_pts.mean(0)
        dist = 4.0
        for _ in range(3):
            a, b, c = (cam_pts - look).T
            dist = float(np.max(np.maximum(np.abs(a) / tan_h, np.abs(b) / tan_v) - c))
            sr, su = a / (c + dist), b / (c + dist)
            look = look + np.array([0.5 * (sr.max() + sr.min()) * dist,
                                    0.5 * (su.max() + su.min()) * dist, 0.0])

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look[0] * right + look[1] * up + look[2] * f
        cam.distance = float(dist)
        cam.azimuth, cam.elevation = self.azimuth, self.elevation

        xs = []
        for grp in groups:
            hp = self._corners(model, data, grp["geoms"])
            a, c = hp @ right - look[0], hp @ f - look[2]
            s = a / (c + dist)
            xs.append(0.5 + 0.5 * (0.5 * (float(s.min()) + float(s.max()))) / tan_h)

        if viser:
            whiten_ground(model)
            with ViserPairRenderer(port=port, max_render=(self.width, self.height)) as vr:
                img = vr.render(model, data, cam, self.width, self.height)
        else:
            opt = mujoco.MjvOption()
            opt.geomgroup[3:] = 0
            with mujoco.Renderer(model, self.height, self.width) as r:
                r.update_scene(data, camera=cam, scene_option=opt)
                img = r.render().copy()

        bare = out_path.with_name(f"{out_path.stem}_bare{out_path.suffix}")
        bare.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(bare)
        print(f"  wrote {bare}")

        # Mass metadata
        labels = []
        for p in paths:
            mass = float(np.sum(spec.compile().body_mass))
            dname = p.parent.name
            rep_path = p.parent / "feasibility_report.yaml"
            if rep_path.exists():
                rep = yaml.safe_load(rep_path.read_text())
                mass = rep.get("mass_composition_check", {}).get("total_mass_kg", mass)
            labels.append(f"{dname.capitalize()}   {mass:.1f} kg")

        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(
            {
                "designs": designs,
                "labels": labels,
                "x_fraction": [round(x, 5) for x in xs],
                "plate": bare.name,
                "size": [self.width, self.height],
                "azimuth": self.azimuth,
                "elevation": self.elevation,
            },
            indent=1,
        ))

        colours = [get_label_color(d) for d in designs]
        _compose(
            img,
            out_path,
            title,
            [(x, t, c) for x, t, c in zip(xs, labels, colours)],
        )
        print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--designs", nargs="+", default=HEXAPOD_DESIGNS)
    ap.add_argument("--mujoco", action="store_true", help="Draw with MuJoCo classic offscreen renderer")
    ap.add_argument("--port", type=int, default=8099, help="Viser headless browser port")
    args = ap.parse_args()

    DOCS.mkdir(parents=True, exist_ok=True)

    solo_renderer = HexapodRenderer(width=1440, height=1080, azimuth=132.0, elevation=-22.0)
    print("Rendering individual hexapods into docs/ ...")
    for name in args.designs:
        xml_path = GENERATED / name / "hexapod.xml"
        if not xml_path.exists():
            print(f"Skipping {name}: {xml_path} not found")
            continue
        out_primary = DOCS / f"hexapod_{name}.png"
        solo_renderer.render_solo(
            name=name,
            xml_path=xml_path,
            out_path=out_primary,
            viser=not args.mujoco,
            port=args.port,
        )
        shutil.copy2(out_primary, DOCS / f"{name}.png")

    print("\nRendering hexapod lineup plate into docs/ ...")
    lineup_width = min(9000, max(2000, 700 * len(args.designs)))
    lineup_renderer = HexapodRenderer(width=lineup_width, height=860, azimuth=128.0, elevation=-14.0)
    lineup_path = DOCS / "hexapod_lineup.png"
    lineup_renderer.render_lineup_plate(
        designs=args.designs,
        out_path=lineup_path,
        viser=not args.mujoco,
        port=args.port,
    )

    print("\nAll renders completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
