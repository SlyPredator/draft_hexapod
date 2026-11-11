#!/usr/bin/env python3
"""Generate the sprawled hexapod designs (tarantula, beetle, strider).

Each design is a preset in ``src/draft/robots/hexapod/presets/`` applied to
``src/draft/robots/hexapod/parameters.yaml`` and written to ``generated/<name>/``.

  python scripts/generate_hexapods.py              # all three presets
  python scripts/generate_hexapods.py --only tarantula
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from draft.generation.generator import RobotGenerator
from draft.paths import repo_root
from draft.robots import robot_dir

_REPO = repo_root()
_HEXAPOD_DIR = robot_dir("hexapod")
HEXAPOD_DESIGNS = ("tarantula", "beetle", "strider")


def generate_design(name: str) -> Path:
    """Generate ``generated/<name>/`` from ``presets/<name>.yaml``."""
    preset_file = _HEXAPOD_DIR / "presets" / f"{name}.yaml"
    if not preset_file.exists():
        raise FileNotFoundError(f"No hexapod preset config: {preset_file} (have: {HEXAPOD_DESIGNS})")
    override = yaml.safe_load(preset_file.read_text()) or {}
    out = _REPO / "generated" / name
    out.mkdir(parents=True, exist_ok=True)
    RobotGenerator(_HEXAPOD_DIR).generate(out, params_override=override)
    (out / "parameters_from_gui.yaml").write_text(preset_file.read_text())
    print(f"  {name:9s} -> {out.relative_to(_REPO)}/hexapod.xml")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--only",
        nargs="+",
        choices=HEXAPOD_DESIGNS,
        default=None,
        help="Generate only these hexapod designs.",
    )
    args = ap.parse_args()

    names = args.only or HEXAPOD_DESIGNS
    print(f"Generating {len(names)} hexapod design(s): {', '.join(names)}")
    for name in names:
        generate_design(name)
    print("\nAll hexapod designs generated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
