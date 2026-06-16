# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt2d>=0.4.0",
#     "jax[cuda]>=0.9.2",
# ]
# ///
from pathlib import Path

from differt2d.scene import Scene

scene = Scene.square_scene_with_obstacle()

folder = Path(__file__).parent.parent / "data"
folder.mkdir(exist_ok=True)

for order in [1, 2, 3]:
    with (
        open(folder / f"valid-ray-paths-{order}.txt", "w") as f_valid,
        open(folder / f"invalid-ray-paths-{order}.txt", "w") as f_invalid,
    ):
        for _, _, valid, path, _ in scene.all_paths(min_order=order, max_order=order):
            if valid:
                for x, y in path.xys:
                    f_valid.write(f"{x:.6f} {y:.6f}\n")
                f_valid.write("\n")
            else:
                for x, y in path.xys:
                    f_invalid.write(f"{x:.6f} {y:.6f}\n")
                f_invalid.write("\n")
