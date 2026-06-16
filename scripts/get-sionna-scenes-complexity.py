# ruff: noqa: F722
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt",
# ]
# ///

from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
    list_sionna_scenes,
)

download_sionna_scenes("v2.0.0")


for scene_name in list_sionna_scenes():
    file = get_sionna_scene(scene_name)
    scene = TriangleScene.load_xml(file)
    print(
        f"Scene: {scene_name}, #triangles: {scene.mesh.num_triangles}, #objects: {len(scene.mesh.object_bounds)}"
    )
