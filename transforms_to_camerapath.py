"""
Convert nerfstudio transforms.json -> viewer-aligned camera_path.json

Features:
- Correct camera convention (OpenGL -> nerfstudio viewer)
- Implicit recenter (nerfstudio runtime behavior)
- Scene scale normalized by MAX camera radius (confirmed correct)
- Configurable render_width / render_height

Usage:
python transforms_to_camerapath.py \
    --transforms bear/transforms.json \
    --output bear/camera_paths/20251223.json \
    --fps 24 \
    --render-width 400 \
    --render-height 400
"""

import json
import math
import argparse
from pathlib import Path
import numpy as np


def get_fov_degrees(fl, h):
    if fl is None or fl == 0:
        return 75.0
    return 2 * math.atan(h / (2 * fl)) * (180.0 / math.pi)


def convert_transforms_to_camerapath(
    transforms_path,
    output_path,
    fps=30,
    render_width=None,
    render_height=None,
):
    transforms_path = Path(transforms_path)
    if not transforms_path.exists():
        raise FileNotFoundError(transforms_path)

    with open(transforms_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if len(frames) == 0:
        raise ValueError("No frames found in transforms.json")

    # deterministic order
    frames.sort(key=lambda x: x.get("file_path", ""))

    # ------------------------------------------------------------
    # Compute scene center & scale (viewer basis)
    #   scale = 1 / max_radius   ← confirmed correct
    # ------------------------------------------------------------
    positions = []
    for f in frames:
        T = np.array(f["transform_matrix"], dtype=np.float64)
        positions.append(T[:3, 3])
    positions = np.stack(positions, axis=0)

    scene_center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - scene_center, axis=1)
    max_radius = np.max(radii)

    if max_radius <= 0:
        scene_scale = 1.0
    else:
        scene_scale = 1.0 / max_radius

    print("=== Scene normalization ===")
    print(f"scene_center: {scene_center.tolist()}")
    print(f"max_radius : {max_radius:.6f}")
    print(f"scene_scale: {scene_scale:.9f}")
    print("==========================")

    # ------------------------------------------------------------
    # Render size (CLI > transforms.json > fallback)
    # ------------------------------------------------------------
    final_w = (
        int(render_width)
        if render_width is not None
        else int(data.get("w", 400))
    )
    final_h = (
        int(render_height)
        if render_height is not None
        else int(data.get("h", 400))
    )

    global_fl_y = data.get("fl_y")
    default_fov = get_fov_degrees(global_fl_y, final_h)

    keyframes = []
    camera_path = []

    # ------------------------------------------------------------
    # Process frames
    # ------------------------------------------------------------
    for frame in frames:
        w = final_w
        h = final_h

        fl_y = frame.get("fl_y", global_fl_y)
        if fl_y is None:
            fl_y = h / (2 * math.tan(math.radians(default_fov) / 2))

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)

        # 1) recenter
        c2w[:3, 3] -= scene_center

        # 2) scale (MAX radius)
        c2w[:3, 3] *= scene_scale
        
        matrix_16 = c2w.flatten().tolist()
        fov = get_fov_degrees(fl_y, h)
        aspect = w / h

        keyframes.append({
            "matrix": matrix_16,
            "fov": fov,
            "aspect": aspect,
            "override_transition_enabled": False,
            "override_transition_sec": None
        })

        camera_path.append({
            "camera_to_world": matrix_16,
            "fov": fov,
            "aspect": aspect
        })

    seconds = len(frames) / float(fps)

    output_data = {
        "camera_type": "perspective",
        "render_width": final_w,
        "render_height": final_h,
        "fps": float(fps),
        "seconds": seconds,
        "is_cycle": False,
        "smoothness_value": 0.0,  # exact geometry
        "default_fov": default_fov,
        "default_transition_sec": 0.2,
        "keyframes": keyframes,
        "camera_path": camera_path
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"[DONE] Camera path written to: {output_path}")
    print(f"[INFO] render size: {final_w} x {final_h}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert transforms.json to viewer-aligned camera_path.json (max-radius scaled)"
    )
    parser.add_argument("--transforms", required=True, help="input transforms.json")
    parser.add_argument("--output", required=True, help="output camera_path.json")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=None)

    args = parser.parse_args()

    convert_transforms_to_camerapath(
        transforms_path=args.transforms,
        output_path=args.output,
        fps=args.fps,
        render_width=args.render_width,
        render_height=args.render_height,
    )
