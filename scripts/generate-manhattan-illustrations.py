# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt>=0.8.2",
#     "kaleido>=0.2.1",
#     "pillow>=10.0.0",
#     "plotly>=5.20.0",
#     "equinox>=0.13.5",
#     "jax[cpu]>=0.8.1",
# ]
# ///
import io
import urllib.request
import zipfile
from pathlib import Path

import equinox as eqx
import numpy as np
import plotly.graph_objects as go
from differt.plotting import set_backend
from differt.scene import TriangleScene
from PIL import Image

set_backend("plotly")

URL = "https://github.com/jeertmans/sampling-paths/releases/download/npjwt2026/manhattan.zip"
DATA_DIR = Path(__file__).parent.parent / "data" / "manhattan"
XML_PATH = DATA_DIR / "manhattan.xml"

if not XML_PATH.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR.parent / "manhattan.zip"
    print(f"Downloading scene files from {URL}...")
    urllib.request.urlretrieve(URL, zip_path)
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    zip_path.unlink()
    print("Done!")

BASE_SCENE = TriangleScene.load_xml(str(XML_PATH))

s_medium = eqx.tree_at(
    lambda s: s.mesh,
    BASE_SCENE,
    BASE_SCENE.mesh.keep_all_within(
        x_min=-130, x_max=200, y_min=50, y_max=200, preserve_objects=False
    )
    .masked()
    .add_ground(x_scale=1.2, y_scale=1.2)
    .center(),
)
s_small = eqx.tree_at(
    lambda s: s.mesh,
    BASE_SCENE,
    BASE_SCENE.mesh.keep_all_within(
        x_min=-130, x_max=50, y_min=50, y_max=200, preserve_objects=False
    )
    .masked()
    .add_ground(x_scale=1.2, y_scale=1.2)
    .center(),
)


def save_fig(
    fig: go.Figure,
    filename: str,
    azim: float = 45.0,
    elev: float = 90.0,
    dist: float = 1.0,
    crop: bool = False,
):
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)
    camera = dict(
        eye=dict(
            x=float(dist * np.cos(elev_rad) * np.cos(azim_rad)),
            y=float(dist * np.cos(elev_rad) * np.sin(azim_rad)),
            z=float(dist * np.sin(elev_rad)),
        )
    )
    fig.update_layout(
        scene_camera=camera,
        width=1600,
        height=1200,
        margin=dict(t=0, r=0, l=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig_bytes = fig.to_image(format="png", scale=4)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    width, height = img.size
    if crop:
        img_array = np.asarray(img)
        y, x = img_array[:, :, 3].nonzero()  # get the nonzero alpha coordinates
        minx = np.min(x)
        miny = np.min(y)
        maxx = np.max(x)
        maxy = np.max(y)
        cropped_img_array = img_array[
            max(miny - 3, 0) : min(maxy + 3, height),
            max(minx - 3, 0) : min(maxx + 3, width),
        ]
        img = Image.fromarray(cropped_img_array)
    img = img.resize((width // 4, height // 4), resample=Image.LANCZOS)
    img.save(filename)


# Let's use a different viewing angle to make the scenes look different:
# Original was azim=-45, elev=60, dist=5.03.
# Let's use azim=-30, elev=55, dist=5.1.
for scene in [s_medium, s_small]:
    fig = scene.plot()
    output_path = (
        Path("/home/eertmans/repositories/DiffeRT-thesis/images")
        / f"scene_{scene.mesh.num_triangles}.png"
    )
    print(f"Generating {output_path}...")
    save_fig(
        fig,
        str(output_path),
        azim=-30,
        elev=55,
        dist=5.1,
        crop=True,
    )
    print(f"Saved {output_path}")
