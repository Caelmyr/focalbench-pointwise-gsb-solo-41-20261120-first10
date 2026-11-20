"""Render the plaque demo preview tile shown on the library page (static/plaque.webp)."""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from focalbench.demos import DEMOS, composite_boss, frame_field_mask, optically_blur, serial_plaque


def main() -> None:
    demo = DEMOS["plaque"]
    scene, depth, boss_path = serial_plaque(demo)
    rng = np.random.default_rng(demo.seed + 900)
    # Middle slot: full field of view, boss near the center, everything recognizable.
    index = len(demo.positions) // 2
    framed, framed_depth = composite_boss(scene, depth, float(boss_path[index]), rng)
    blurred = optically_blur(framed, framed_depth, demo.positions[index])
    rgb = np.clip(blurred, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb)
    image.thumbnail((560, 380))
    target = Path(__file__).resolve().parent.parent / "static" / "plaque.webp"
    image.save(target, format="WEBP", quality=86, method=4)
    print("wrote", target, image.size)


if __name__ == "__main__":
    main()
