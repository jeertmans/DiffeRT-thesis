# ruff: noqa: F722
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt",
#     "equinox",
#     "jax[cuda]",
#     "jaxtyping",
#     "matplotlib",
#     "plotly",
#     "tqdm",
# ]
# ///


import hashlib
import random

import equinox as eqx
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from differt.geometry import (
    merge_cell_ids,
)
from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from jaxtyping import Array, Bool, Int
from matplotlib.colors import ListedColormap
from plotly.colors import convert_to_RGB_255
from tqdm.auto import tqdm

mpl.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "text.usetex": True,
        "pgf.rcfonts": False,
    },
)

download_sionna_scenes("v2.0.0")


file = get_sionna_scene("simple_street_canyon")
scene = TriangleScene.load_xml(file)


def hashfun(*objects: bytes) -> bytes:
    m = hashlib.sha256()

    for obj in objects:
        m.update(obj)

    return m.digest()


def get_cell_hashes(
    cell_ids: Int[Array, " *batch"],
    mask: Bool[Array, "*batch num_path_candidates"],
) -> dict[int, bytes]:
    mask = mask.reshape(-1, mask.shape[-1])

    return {
        int(i): hashfun(mask[i, :].tobytes())
        for i in jnp.unique(cell_ids, return_index=True)[1]
    }


def merge_cell_ids_and_hashes(
    cell_ids: Int[Array, " *batch"],
    new_cell_ids: Int[Array, " *batch"],
    cell_hashes: dict[int, bytes],
    new_cell_hashes: dict[int, bytes],
) -> tuple[Int[Array, " *batch"], dict[int, bytes]]:
    ret_cell_ids = merge_cell_ids(cell_ids, new_cell_ids)

    ret_cell_hashes = {}

    for index in jnp.unique(ret_cell_ids, return_index=True)[1]:
        i = cell_ids.ravel()[index]
        j = new_cell_ids.ravel()[index]

        ret_cell_hashes[int(index)] = hashfun(
            cell_hashes[int(i)],
            new_cell_hashes[int(j)],
        )

    return ret_cell_ids, ret_cell_hashes


def random_rgb(cell_hash: bytes) -> str:
    rng = random.Random(cell_hash)  # noqa: S311
    r = rng.randint(0, 255)
    g = rng.randint(0, 255)
    b = rng.randint(0, 255)
    return f"rgb({r},{g},{b})"


def create_discrete_colorscale(
    cell_ids: Int[Array, " *batch"],
    cell_hashes: dict[int, bytes],
    first_is_multipath_cell: bool,
) -> list[list[float | str]]:
    unique_ids = jnp.unique(cell_ids).tolist()
    min_id = min(unique_ids)
    max_id = max(unique_ids)
    scale_factor = 1 + max_id - min_id

    def scale(id_: int) -> float:
        return (id_ - min_id) / scale_factor

    colorscale = [
        [scale(id_ + offset), random_rgb(cell_hashes[id_])]
        for id_ in unique_ids
        for offset in (0, 1)
    ]

    if first_is_multipath_cell:  # Let's hide the cell with no multipath
        colorscale[0][1] = colorscale[1][1] = "rgba(0,0,0,0)"

    return colorscale


# Let's put one transmitter and many receivers in our scene
scene = eqx.tree_at(lambda s: s.transmitters, scene, jnp.array([-33.0, 0.0, 32.0]))
# Our scene can be simplified to quadrilaterals,
# so informing the code of that matter will make it run faster
scene = scene.set_assume_quads()
batch = (
    1000,
    1000,
)  # Warning: a too large batch could easily cause OOM issues,
#    or you may want to reduce the 'chunk_size' value below.
z0 = 1.5  # The z coordinate of the receivers
scene_grid = scene.with_receivers_grid(*batch, height=z0)

# Only needed for plotting purposes
x, y, _ = jnp.unstack(scene_grid.receivers, axis=-1)


def draw_mesh_2d_mpl(mesh, ax) -> None:
    assert mesh.object_bounds is not None

    for i, j in mesh.object_bounds:
        sub_mesh = mesh[i:j]
        # Bounding box: xs, ys, zs
        (xs, ys, (_, z_max)) = sub_mesh.bounding_box.T

        zorder = 0 if z_max < 1e-6 else 1

        assert sub_mesh.face_colors is not None
        # Convert color to 0-1 range for Matplotlib
        color_255 = convert_to_RGB_255(sub_mesh.face_colors[0, :])
        color_normalized = [c / 255.0 for c in color_255]

        # Draw rectangle
        rect = patches.Rectangle(
            (xs[0], ys[0]),  # (x, y) bottom left
            xs[1] - xs[0],  # width
            ys[1] - ys[0],  # height
            linewidth=0,
            facecolor=color_normalized,
            zorder=zorder,
        )
        ax.add_patch(rect)


def create_mpl_cmap(cell_ids, cell_hashes, first_is_multipath_cell):
    unique_ids = np.unique(cell_ids)
    colors = []

    for id_ in sorted(unique_ids):
        # Extract RGB string "rgb(r,g,b)" and convert to [r,g,b] floats
        rgb_str = random_rgb(cell_hashes[int(id_)])
        rgb_vals = [
            int(c) / 255.0
            for c in rgb_str.replace("rgb(", "").replace(")", "").split(",")
        ]

        # Handle transparency for the "no multipath" cell
        if first_is_multipath_cell and id_ == unique_ids[0]:
            colors.append([0, 0, 0, 0])  # Fully transparent
        else:
            colors.append(rgb_vals + [1.0])  # Opaque

    return ListedColormap(colors)


for order_name, orders in tqdm([("2", [2])], desc="Orders"):
    for pos_name, x_pos in tqdm([("b", 0.0)], desc="Position", leave=False):
        fig, ax = plt.subplots()
        fig.set_figwidth(7.5 / 2.54)

        draw_mesh_2d_mpl(scene.mesh, ax)

        scene_grid = eqx.tree_at(
            lambda s: s.transmitters,
            scene_grid,
            scene_grid.transmitters.at[0].set(x_pos),
        )
        cell_ids = jnp.zeros(batch, dtype=jnp.int32)
        cell_hashes = {0: b""}
        has_multipath = jnp.zeros(batch, dtype=bool)

        for order in tqdm(orders, desc="Order", leave=False):
            for paths in tqdm(
                scene_grid.compute_paths(order=order, chunk_size=5),
                desc="Chunks",
                leave=False,
            ):
                new_cell_ids = paths.multipath_cells()
                new_cell_hashes = get_cell_hashes(new_cell_ids, paths.mask)
                has_multipath |= paths.mask.any(axis=-1)
                cell_ids, cell_hashes = merge_cell_ids_and_hashes(
                    cell_ids,
                    new_cell_ids,
                    cell_hashes,
                    new_cell_hashes,
                )

        if not has_multipath.all():
            cell_id = jnp.max(cell_ids, initial=0, where=~has_multipath)
            cell_hashes[-1] = cell_hashes.pop(int(cell_id))

        cell_ids = jnp.where(has_multipath, cell_ids, -1)
        unique_ids, renumbered_cell_ids = jnp.unique(cell_ids, return_inverse=True)
        renumbered_cell_ids = renumbered_cell_ids.reshape(cell_ids.shape)
        renumbered_cell_hashes = {
            i: cell_hashes[int(id_)] for i, id_ in enumerate(unique_ids)
        }

        cmap = create_mpl_cmap(
            renumbered_cell_ids,
            renumbered_cell_hashes,
            first_is_multipath_cell=bool(~has_multipath.all()),
        )
        im = ax.imshow(
            np.asarray(renumbered_cell_ids),
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=cmap,
            origin="lower",
            alpha=0.8,
            zorder=0.5,
            rasterized=True,
        )

        tx_x, tx_y, _ = scene_grid.transmitters.reshape(3, 1)
        ax.scatter(tx_x, tx_y, marker="x", c="black", s=25, zorder=5)

        ax.set_axis_off()
        ax.set_aspect("equal")
        plt.savefig(
            f"pgf/mlm-{order_name}-{pos_name}.pgf",
            dpi=500,
            bbox_inches="tight",
            pad_inches=0.0,
        )
