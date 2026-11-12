"""Generate a 2x3 grid animated GIF and MP4 showcase of the Beetle terrain policy.

Renders 6 synchronized sub-terrain rollouts at nominal, comfortable traversal levels:
  1. Flat Ground (Level 0 / 0 cm)
  2. Wave Terrain (Level 3.0 / 12 cm wave swells)
  3. Random Rough (Nominal 6 cm rocky rubble, un-smoothed)
  4. Pyramid Slope (Level 2.0 / 12° incline climb)
  5. Pyramid Stairs (7 cm risers, ascending)
  6. Descending Stairs (3.5 cm drops, descending)

Outputs:
  docs/beetle_terrain_showcase.gif
  docs/beetle_terrain_showcase.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# Ensure EGL on Linux for offscreen headless rendering
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import draft  # noqa: F401
from draft.tasks import get_task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


SHOWCASE_CELLS = [
    {
        "title": "Flat Ground",
        "sub_title": "Flat Plane",
        "sub_name": None,
        "difficulty": 0.0,
        "x_start": 0.0,
        "z_start": 0.0,
        "extra_cfg": {},
    },
    {
        "title": "Wave Terrain",
        "sub_title": "12 cm swells",
        "sub_name": "wave_terrain",
        "difficulty": 0.30,
        "x_start": 0.0,
        "z_start": 0.0,
        "extra_cfg": {},
    },
    {
        "title": "Random Rough",
        "sub_title": "6 cm rubble",
        "sub_name": "random_rough",
        "difficulty": 0.20,
        "x_start": 0.0,
        "z_start": 0.0,
        "extra_cfg": {
            "noise_range": (0.02, 0.07),
            "downsampled_scale": 0.08,
        },
    },
    {
        "title": "Pyramid Slope",
        "sub_title": "12 deg incline",
        "sub_name": "hf_pyramid_slope",
        "difficulty": 0.20,
        "x_start": -2.0,
        "z_start": 0.0,
        "extra_cfg": {},
    },
    {
        "title": "Ascending Stairs",
        "sub_title": "9 cm risers",
        "sub_name": "pyramid_stairs",
        "difficulty": 0.08,
        "x_start": -3.0,
        "z_start": 0.0,
        "cam_azimuth": -45.0,
        "loop_steps": 150,
        "extra_cfg": {
            "step_height_range": (0.09, 0.09),
            "step_width": 0.40,
            "platform_width": 4.0,
        },
    },
    {
        "title": "Descending Stairs",
        "sub_title": "3.5 cm drops",
        "sub_name": "pyramid_stairs_inv",
        "difficulty": 0.08,
        "x_start": -2.5,
        "z_start": 0.12,
        "cam_azimuth": 135.0,
        "loop_steps": 150,
        "extra_cfg": {
            "step_height_range": (0.06, 0.06),
            "platform_width": 4.0,
        },
    },
]


def _draw_badge(
    image: Image.Image,
    title: str,
    sub_title: str,
    font_title: ImageFont.FreeTypeFont,
    font_sub: ImageFont.FreeTypeFont,
) -> None:
    """Overlay a clean semi-transparent pill badge in the top-left."""
    draw = ImageDraw.Draw(image, "RGBA")
    
    t_bbox = draw.textbbox((0, 0), title, font=font_title)
    s_bbox = draw.textbbox((0, 0), sub_title, font=font_sub)
    text_w = max(t_bbox[2] - t_bbox[0], s_bbox[2] - s_bbox[0])
    
    pad_x, pad_y = 12, 7
    badge_x, badge_y = 14, 14
    badge_w = text_w + 2 * pad_x
    badge_h = (t_bbox[3] - t_bbox[1]) + (s_bbox[3] - s_bbox[1]) + 2 * pad_y + 4

    # Background pill (slate-900 with 85% opacity)
    pill_box = [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h]
    draw.rounded_rectangle(
        pill_box,
        radius=7,
        fill=(15, 23, 42, 220),
        outline=(51, 65, 85, 240),
        width=1,
    )

    # Title (clean crisp white)
    draw.text(
        (badge_x + pad_x, badge_y + pad_y),
        title,
        font=font_title,
        fill=(248, 250, 252, 255),
    )
    # Subtitle (vibrant cyan accent)
    draw.text(
        (badge_x + pad_x, badge_y + pad_y + (t_bbox[3] - t_bbox[1]) + 3),
        sub_title,
        font=font_sub,
        fill=(56, 189, 248, 255),
    )


def render_cell_rollout(
    cell_cfg: dict,
    checkpoint: str,
    robot_xml: str,
    num_steps: int = 300,
    cell_w: int = 480,
    cell_h: int = 320,
    speed: float = 0.45,
    device: str = "cuda:0",
) -> list[np.ndarray]:
    """Execute policy rollout on one specific terrain and record tracking frames."""
    task = get_task("hexapod_terrain")
    env_cfg = task.make_env_cfg(n_envs=1, play=True, robot_xml=robot_xml)

    # Disable sensor debug visualizer spheres/lines if present
    for s in env_cfg.scene.sensors:
        if hasattr(s, "debug_vis"):
            s.debug_vis = False

    # Pin constant forward command
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (speed, speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.resampling_time_range = (1e9, 1e9)

    sub_name = cell_cfg["sub_name"]
    diff = cell_cfg["difficulty"]
    x_start = cell_cfg["x_start"]
    z_start = cell_cfg.get("z_start", 0.0)
    extra_cfg = cell_cfg.get("extra_cfg", {})

    if sub_name is None:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
    else:
        gen = env_cfg.scene.terrain.terrain_generator
        gen.num_rows = 1
        gen.num_cols = 1
        gen.difficulty_range = (diff, diff)
        gen.curriculum = False
        st = gen.sub_terrains[sub_name]
        ov = {"proportion": 1.0, **extra_cfg}
        gen.sub_terrains = {sub_name: replace(st, **ov)}

    env_cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
    env_cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
    if x_start != 0.0:
        env_cfg.events["reset_base"].params["pose_range"]["x"] = (x_start, x_start)
    if z_start != 0.0:
        env_cfg.events["reset_base"].params["pose_range"]["z"] = (z_start, z_start)

    env = ManagerBasedRlEnv(env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cfg = task.make_runner_cfg()
    train_cfg = runner_cfg.build_rsl_rl_cfg()
    train_cfg["experiment_name"] = "showcase"
    runner = MjlabOnPolicyRunner(vec_env, train_cfg, device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Resolve exact body ID of robot torso (body 2 when terrain is present)
    robot_body_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link")
    if robot_body_id < 0:
        robot_body_id = env.scene["robot"].indexing.root_body_id

    # MuJoCo offscreen camera setup
    renderer = mujoco.Renderer(env.sim.mj_model, height=cell_h, width=cell_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    cam.distance = cell_cfg.get("cam_distance", 2.1)
    cam.azimuth = cell_cfg.get("cam_azimuth", 135.0)
    cam.elevation = cell_cfg.get("cam_elevation", -24.0)

    # Fonts for badges
    font_title = ImageFont.truetype(FONT_PATH, 15)
    font_sub = ImageFont.truetype(FONT_PATH, 11)

    steps_to_roll = cell_cfg.get("loop_steps", num_steps)
    obs = vec_env.get_observations()
    frames: list[np.ndarray] = []

    with torch.no_grad():
        for step_idx in range(steps_to_roll):
            actions = policy(obs)
            obs, _, _, _ = vec_env.step(actions)

            # Synchronize GPU state into CPU MjData for MuJoCo rendering
            env.sim.mj_data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
            env.sim.mj_data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(env.sim.mj_model, env.sim.mj_data)

            # Lock camera directly to robot torso position every frame
            cam.lookat = env.sim.mj_data.xpos[robot_body_id].copy()

            renderer.update_scene(env.sim.mj_data, camera=cam)
            # Disable directional shadow acne on heightfields to keep surface silky clean
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            raw_rgb = renderer.render()

            # Add overlay badge
            pil_img = Image.fromarray(raw_rgb)
            _draw_badge(pil_img, cell_cfg["title"], cell_cfg["sub_title"], font_title, font_sub)
            frames.append(np.asarray(pil_img))

    env.close()

    # If cell is shorter (e.g. 5.0s loop), repeat to match target num_steps (10.0s)
    if len(frames) < num_steps:
        repeats = (num_steps // len(frames)) + 1
        frames = (frames * repeats)[:num_steps]

    return frames


def build_composite_grid(
    cell_frames: list[list[np.ndarray]],
    num_steps: int,
    cell_w: int,
    cell_h: int,
) -> list[np.ndarray]:
    """Combine 6 cell sequences into a 2-row x 3-column mosaic with subtle border grid."""
    grid_rows, grid_cols = 2, 3
    total_w = grid_cols * cell_w
    total_h = grid_rows * cell_h

    composite_frames: list[np.ndarray] = []

    for t in range(num_steps):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        for idx in range(6):
            r = idx // grid_cols
            c = idx % grid_cols
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            canvas[y1:y2, x1:x2] = cell_frames[idx][t]

        # Draw subtle 2px dark border between cells
        canvas[cell_h - 1 : cell_h + 1, :] = 30
        for c in range(1, grid_cols):
            canvas[:, c * cell_w - 1 : c * cell_w + 1] = 30

        composite_frames.append(canvas)

    return composite_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2x3 terrain showcase GIF & MP4.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/beetle_terrain_v2/2026-11-11_23-16-17_beetle_terrain_v2/model_4999.pt",
        help="Path to trained policy checkpoint.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="generated/beetle/hexapod.xml",
        help="Path to robot XML.",
    )
    parser.add_argument(
        "--out-gif",
        type=str,
        default="docs/beetle_terrain_showcase.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--out-mp4",
        type=str,
        default="docs/beetle_terrain_showcase.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--num-steps", type=int, default=300, help="Frames to record (300 = 10.0s at 30 fps).")
    parser.add_argument("--fps", type=int, default=30, help="Framerate.")
    parser.add_argument("--width", type=int, default=480, help="Cell width.")
    parser.add_argument("--height", type=int, default=320, help="Cell height.")
    args = parser.parse_args()

    out_gif_p = Path(args.out_gif)
    out_mp4_p = Path(args.out_mp4)
    out_gif_p.parent.mkdir(parents=True, exist_ok=True)
    out_mp4_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Rendering 6 Terrain Sequences for Beetle Showcase ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Robot: {args.robot}")
    print(f"Duration: {args.num_steps} steps at {args.fps} FPS ({args.num_steps / args.fps:.1f}s)")

    all_cell_frames: list[list[np.ndarray]] = []
    for idx, cell_cfg in enumerate(SHOWCASE_CELLS):
        print(f"[{idx+1}/6] Rendering {cell_cfg['title']} ({cell_cfg['sub_title']})...")
        frames = render_cell_rollout(
            cell_cfg=cell_cfg,
            checkpoint=args.checkpoint,
            robot_xml=args.robot,
            num_steps=args.num_steps,
            cell_w=args.width,
            cell_h=args.height,
        )
        all_cell_frames.append(frames)
        print(f"      Recorded {len(frames)} frames.")

    print("\nAssembling 2x3 composite grid (1440x640)...")
    composite_frames = build_composite_grid(
        all_cell_frames,
        num_steps=args.num_steps,
        cell_w=args.width,
        cell_h=args.height,
    )

    print(f"Exporting MP4 to {out_mp4_p}...")
    imageio.mimsave(
        str(out_mp4_p),
        composite_frames,
        fps=args.fps,
        codec="libx264",
        quality=8,
    )
    mp4_size_mb = out_mp4_p.stat().st_size / (1024 * 1024)
    print(f"  MP4 exported: {out_mp4_p} ({mp4_size_mb:.2f} MB)")

    print(f"Optimizing and exporting GIF to {out_gif_p}...")
    gif_w, gif_h = 1080, 480
    pil_gif_frames: list[Image.Image] = []
    for frame in composite_frames:
        img = Image.fromarray(frame).resize((gif_w, gif_h), Image.Resampling.LANCZOS)
        pil_gif_frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = int(1000.0 / args.fps)
    pil_gif_frames[0].save(
        str(out_gif_p),
        save_all=True,
        append_images=pil_gif_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    gif_size_mb = out_gif_p.stat().st_size / (1024 * 1024)
    print(f"  GIF exported: {out_gif_p} ({gif_size_mb:.2f} MB)")

    print("\nShowcase generation complete!")


if __name__ == "__main__":
    main()
