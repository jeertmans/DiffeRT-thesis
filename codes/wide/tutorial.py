# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "differt==0.8.1",
#     "jax[cuda]",
#     "kaleido",
#     "pillow>=12.1.1",
#     "plotly",
# ]
# ///
import io
from pathlib import Path

import differt.plotting as dplt
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from differt.geometry import (
  TriangleMesh,
  assemble_paths,
  triangles_contain_vertices_assuming_inside_same_plane,
)
from differt.rt import (
  consecutive_vertices_are_on_same_side_of_mirrors,
  generate_all_path_candidates,
  image_method,
  rays_intersect_triangles,
)
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DIR = Path(__file__).parent

dplt.set_defaults("plotly")  # Let's use the Plotly backend


def draw_tx_rx_on_plotly_fig(
  fig: go.Figure, tx: np.ndarray, rx: np.ndarray
) -> None:
  fig.add_trace(
    go.Scatter3d(
      x=[tx[0]],
      y=[tx[1]],
      z=[tx[2]],
      mode="markers+text",
      marker={"color": "black", "size": 6, "symbol": "x"},
      text=["TX"],
      textfont={
        "color": "black",
        "size": 22,
        "family": "Libertinus Serif",
      },
      showlegend=False,
    )
  )
  fig.add_trace(
    go.Scatter3d(
      x=[rx[0]],
      y=[rx[1]],
      z=[rx[2]],
      mode="markers+text",
      marker={"color": "black", "size": 6, "symbol": "x"},
      text=["RX"],
      textfont={
        "color": "black",
        "size": 22,
        "family": "Libertinus Serif",
      },
      showlegend=False,
    )
  )


def save_fig(
  fig: go.Figure,
  filename: str,
  *,
  azim: float = -100,
  elev: float = 30,
  dist: float = 2.0,
) -> None:
  fig.update_scenes(
    xaxis_visible=False,
    yaxis_visible=False,
    zaxis_visible=False,
  )
  azim_rad = np.deg2rad(azim)
  elev_rad = np.deg2rad(elev)
  camera = {
    "eye": {
      "x": float(dist * np.cos(elev_rad) * np.cos(azim_rad)),
      "y": float(dist * np.cos(elev_rad) * np.sin(azim_rad)),
      "z": float(dist * np.sin(elev_rad)),
    }
  }
  fig.update_layout(
    scene_camera=camera,
    width=1600,
    height=1200,
    margin={"t": 0, "r": 0, "l": 0, "b": 0},
    paper_bgcolor="rgba(0,0,0,0)",
  )
  fig_bytes = fig.to_image(format="png", scale=2)
  buf = io.BytesIO(fig_bytes)
  img = Image.open(buf)
  width, height = img.size
  img_array = np.asarray(img)
  y, x = img_array[
    :, :, 3
  ].nonzero()  # get the nonzero alpha coordinates
  minx = np.min(x)
  miny = np.min(y)
  maxx = np.max(x)
  maxy = np.max(y)
  cropped_img_array = img_array[
    max(0, miny - 3) : min(maxy + 3, height),
    max(0, minx - 3) : min(maxx + 3, width),
  ]
  img = Image.fromarray(cropped_img_array)
  img = img.resize(
    (width // 4, height // 4), resample=Image.LANCZOS
  )
  img.save(IMAGES_DIR / filename)
  print(f"Saved figure to {IMAGES_DIR / filename}")


# BEGIN_scene_loading
# Very simple scene with two buildings
mesh_file = DIR / "two_buildings.obj"
mesh = TriangleMesh.load_obj(str(mesh_file))

tx = jnp.array([0.0, 4.9352, 22.0])
rx = jnp.array([0.0, 10.034, 1.50])
# END_scene_loading

fig = mesh.plot(opacity=0.5)

draw_tx_rx_on_plotly_fig(fig, np.asarray(tx), np.asarray(rx))

save_fig(
  fig, "tutorial-simple-scene.png", azim=0, elev=35, dist=2.0
)

select = [
  8,  # Red
  9,  # Red
  22,  # Green
  23,  # Green
]  # In practice, you will never hard-code the primitive indices yourself

vertices = mesh.vertices
triangles = mesh.triangles[select, :]

dplt.draw_mesh(
  vertices, triangles[:2, :], figure=fig, color="red"
)
dplt.draw_mesh(
  vertices, triangles[2:, :], figure=fig, color="green"
)
# A list of colors to easily differentiate paths
color = ["black", "green", "orange", "yellow", "blue"]

# BEGIN_manual_image_method
select = [
  8,  # Red
  9,  # Red
  22,  # Green
  23,  # Green
]  # In practice, you will never hard-code the indices yourself

vertices = mesh.vertices
triangles = mesh.triangles[select, :]

select = jnp.array(
  select[::2],
  dtype=int,
)  # We actually only need one triangle per plane, so [8, 22]

# Iterate through path candidates
#                         ┌> order 0
#                         |           ┌> order 1
#                         |           |           ┌> order 2
for path_candidate in [select[:0], select[:1], select[:2]]:
  # 1 - Prepare input arrays
  mirror_vertices = mesh.vertices[
    mesh.triangles[path_candidate, 0], :
  ]
  mirror_normals = mesh.normals[path_candidate, :]

  # 2 - Trace paths
  path = image_method(tx, rx, mirror_vertices, mirror_normals)

  # 3 - ??

  # 4 - Obtain final valid paths
  full_path = jnp.concatenate(
    (
      tx[None, :],
      path,
      rx[None, :],
    ),
  )
  # END_manual_image_method

  # Then we plot it
  dplt.draw_paths(
    full_path,
    figure=fig,
    line={"color": color[len(path_candidate)], "width": 4},
    showlegend=False,
  )

save_fig(
  fig,
  "tutorial-image-method-orders-0-2.png",
  azim=0,
  elev=35,
  dist=2.0,
)

fig.data = fig.data[
  :2
]  # Keep only first 2 traces: geometry and TX/RX
# BEGIN_image_method_all_orders_a
# [num_triangles 3 3]
all_triangle_vertices = mesh.triangle_vertices

num_triangles = mesh.num_triangles

for order in range(5):
  # 1 - Prepare input arrays
  # [num_path_candidates order]
  path_candidates = generate_all_path_candidates(
    num_triangles, order
  )
  num_path_candidates = path_candidates.shape[0]

  # [num_path_candidates order 3]
  triangles = jnp.take(mesh.triangles, path_candidates, axis=0)

  # [num_path_candidates order 3 3]
  triangle_vertices = jnp.take(mesh.vertices, triangles, axis=0)

  # [num_path_candidates order 3]
  mirror_vertices = triangle_vertices[
    ...,
    0,  # Only one vertex per triangle is needed
    :,
  ]
  # [num_path_candidates order 3]
  mirror_normals = jnp.take(
    mesh.normals, path_candidates, axis=0
  )

  # 2 - Trace paths
  # [num_path_candidates order 3]
  paths = image_method(tx, rx, mirror_vertices, mirror_normals)
  # END_image_method_all_orders_a

  # BEGIN_image_method_all_orders_b
  # 3 - Remove invalid paths
  # 3.1 - Remove paths with vertices outside triangles
  # [num_path_candidates order]
  mask = triangles_contain_vertices_assuming_inside_same_plane(
    triangle_vertices,
    paths,
  )
  # [num_path_candidates]
  mask = jnp.all(mask, axis=-1)

  # [num_paths_inter order+2 3]
  full_paths = assemble_paths(
    tx,
    paths[mask, ...],
    rx,
  )
  # END_image_method_all_orders_b

  # BEGIN_image_method_all_orders_c
  # 3.2 - Remove paths with vertices not on the same side
  # [num_paths_inter order]
  mask = consecutive_vertices_are_on_same_side_of_mirrors(
    full_paths,
    mirror_vertices[mask, ...],
    mirror_normals[mask, ...],
  )

  # [num_paths_inter]
  mask = jnp.all(
    mask, axis=-1
  )  # We will actually remove them later

  # 3.3 - Remove paths that are obstructed by other objects
  # [num_paths_inter order+1 3]
  ray_origins = full_paths[..., :-1, :]
  # [num_paths_inter order+1 3]
  ray_directions = jnp.diff(full_paths, axis=-2)

  # [num_paths_inter order+1 num_triangles]
  t, hit = rays_intersect_triangles(
    ray_origins[..., None, :],
    ray_directions[..., None, :],
    all_triangle_vertices[None, None, ...],
  )
  # In theory, we could do t < 1.0
  # (because t == 1.0 means we are perfectly on a surface),
  # but numerical errors require us to use some tolerance
  tol = 1e-4
  # [num_paths_inter order+1 num_triangles]
  intersect = (t < (1.0 - tol)) & hit
  #  [num_paths_inter]
  intersect = jnp.any(intersect, axis=(-1, -2))
  #  [num_paths_inter]
  mask = mask & ~intersect

  # 4 - Obtain final valid paths and plot
  #  [num_paths_final]
  full_paths = full_paths[mask, ...]
  # END_image_method_all_orders_c

  dplt.draw_paths(
    full_paths,
    figure=fig,
    line={"color": color[order], "width": 4},
    showlegend=False,
  )

save_fig(
  fig,
  "tutorial-all-orders-valid-paths.png",
  azim=0,
  elev=35,
  dist=2.0,
)

mesh_file = DIR / "bruxelles.obj"
mesh = TriangleMesh.load_obj(str(mesh_file))

tx = jnp.array([-40.0, 75, 30.0])
rx = jnp.array([+20.0, 108.034, 1.50])

city_fig = mesh.plot(opacity=1.0)
city_fig = city_fig.update_layout(
  scene=dict(aspectmode="data"),
)
draw_tx_rx_on_plotly_fig(
  city_fig, np.asarray(tx), np.asarray(rx)
)
save_fig(
  city_fig,
  "tutorial-complex-scene-setup.png",
  azim=-90,
  elev=70,
  dist=1.2,
)

# This is the number of triangles
print(
  f"Number of primitives (triangles): {mesh.num_primitives:_}"
)

from differt_core.rt import CompleteGraph

graph = CompleteGraph(mesh.num_primitives)

from_ = graph.num_nodes  # Index of TX in the graph
to = from_ + 1  # Index of RX in the graph
order = 2  # Interaction order
depth = order + 2  # + 2 because we add TX and RX nodes

num_path_candidates = len(graph.all_paths(from_, to, depth))
print(f"Number of path candidates: {num_path_candidates:_}")


mesh = mesh.set_assume_quads(True)
# This is now the number of quadrilaterals, exactly half the number of triangles
print(
  f"Number of primitives (assume_quads=True): {mesh.num_primitives:_}"
)

graph = CompleteGraph(mesh.num_primitives)

from_ = graph.num_nodes
to = from_ + 1
order = 2

num_path_candidates = len(graph.all_paths(from_, to, depth))
# Roughly a quarter of the previous number
print(f"Number of path candidates: {num_path_candidates:_}")

from differt.rt import triangles_visible_from_vertices

# BEGIN_visibility_from_tx
tx = jnp.array([-40.0, 75, 30.0])

default_color = jnp.array([[0.2, 0.2, 0.2]])  # Hidden, black
visible_color = jnp.array([[1.0, 0.2, 0.2]])  # Visible, red
visible_triangles = triangles_visible_from_vertices(
  tx,
  mesh.triangle_vertices,
)
mesh = mesh.set_face_colors(default_color)
mesh = mesh.set_face_colors(
  mesh.face_colors.at[visible_triangles].set(visible_color)
)
# END_visibility_from_tx
visible_fig = mesh.plot(opacity=1.0)
visible_fig = visible_fig.update_layout(
  scene=dict(aspectmode="data"),
)
draw_tx_rx_on_plotly_fig(
  visible_fig, np.asarray(tx), np.asarray(rx)
)
save_fig(
  visible_fig,
  "tutorial-visibility-from-tx.png",
  azim=-20,
  elev=55,
  dist=1.9,
)

ratio = visible_triangles.sum() / mesh.num_triangles
print(f"Percentage of visible triangles: {100 * ratio:.2f}%")

visible_quads = visible_triangles.reshape(
  mesh.num_quads, 2
).any(axis=-1)
ratio = visible_quads.sum() / mesh.num_quads
print(
  f"Percentage of visible quadrilaterals: {100 * ratio:.2f}%"
)

from differt_core.rt import DiGraph

graph = DiGraph.from_complete_graph(
  CompleteGraph(mesh.num_quads)
)
from_, to = graph.insert_from_and_to_nodes(
  from_adjacency=np.asarray(visible_quads)
)

# DiGraph iterators are not sized, so we consume them to determine their size
num_path_candidates = graph.all_paths(from_, to, depth).count()
# Roughly 43% of the previous number
print(f"Number of path candidates: {num_path_candidates:_}")

import jax
from differt.geometry import fibonacci_lattice, viewing_frustum
from differt.rt import first_triangles_hit_by_rays

mesh = TriangleMesh.load_obj(
  str(mesh_file)
)  # Reload mesh to reset colors

# [num_triangles 3 3]
triangle_vertices = mesh.triangle_vertices

num_triangles = mesh.num_triangles

sbr_fig = mesh.plot(opacity=1.0)
sbr_fig = sbr_fig.update_layout(
  scene=dict(aspectmode="data"),
)
draw_tx_rx_on_plotly_fig(
  sbr_fig, np.asarray(tx), np.asarray(rx)
)

# BEGIN_sbr_a
num_rays = int(1e6)
max_dist = 0.5**2  # Squared distance (to avoid sqrt)
max_order = 2

# [num_path_candidates order 3]
frustum = viewing_frustum(
  tx, triangle_vertices.reshape(-1, 3)
)  # This avoids launching rays where there are no objects

# [num_rays 3]
ray_origins = jnp.broadcast_to(tx, (num_rays, 3))
ray_directions = fibonacci_lattice(num_rays, frustum=frustum)
# END_sbr_a


# BEGIN_sbr_b
def scan_fun(ray_origins_directions_and_valids, _unused_input):
  # Unpack values from previous iteration
  ray_origins, ray_directions, valid_rays = (
    ray_origins_directions_and_valids
  )

  # 1 - Compute next intersection with triangles

  # [num_rays]
  triangles, t_hit = first_triangles_hit_by_rays(
    ray_origins,
    ray_directions,
    triangle_vertices,
  )
  # The above may generate infinite values,
  # so we will need to be careful with those
  # END_sbr_b

  # BEGIN_sbr_c
  # 2 - Check if the rays pass near RX

  # [num_rays 3]
  ray_origins_to_rx = rx - ray_origins

  # [num_rays]
  # note: the fact that ray directions have unit length
  #       allows for some simplifications.
  ray_distances_to_rx = jnp.square(
    jnp.cross(ray_directions, ray_origins_to_rx)
  ).sum(axis=-1)  # Squared distance from rays to RX
  # Distance (scaled by ray directions) from
  # RX projected onto rays to ray origins
  t_rx = jnp.sum(ray_directions * ray_origins_to_rx, axis=-1)
  masks = jnp.where(
    (t_rx < t_hit) & (t_rx > 0) & valid_rays,
    # Check if RX is between origin and first triangle hit
    ray_distances_to_rx
    < max_dist,  # Check if RX is close enough
    False,
  )  # Whether rays pass near RX
  # END_sbr_c

  # BEGIN_sbr_d
  # 3 - Update rays

  # [num_rays 3]
  mirror_normals = jnp.take(mesh.normals, triangles, axis=0)

  ray_origins += t_hit[..., None] * ray_directions
  ray_directions = (
    ray_directions
    - 2.0
    * jnp.sum(
      ray_directions * mirror_normals, axis=-1, keepdims=True
    )
    * mirror_normals
  )
  # We mark rays that left the scene
  # i.e., when they no longer hit any object (t_hit is +inf.)
  valid_rays = valid_rays & jnp.isfinite(t_hit)

  return (ray_origins, ray_directions, valid_rays), (
    ray_origins,
    masks,
  )


# END_sbr_d


# BEGIN_sbr_e
# We mark rays that left the scene as invalid
valid_rays = jnp.ones(num_rays, dtype=bool)

# [max_order+1 num_rays 3], [max_order+1 num_rays]
_, (paths, masks) = jax.lax.scan(
  scan_fun,
  (ray_origins, ray_directions, valid_rays),
  length=max_order + 1,
)

# We swap 'max_order' and 'num_rays' axes
# [num_rays max_order+1 3], [num_rays max_order+1]
paths = jnp.moveaxis(paths, 0, 1)
masks = jnp.moveaxis(masks, 0, 1)

for order in range(max_order + 1):
  full_paths = assemble_paths(
    tx,
    # [num_valid_rays order 3]
    paths[masks[..., order], :order, :],
    rx,
  )
  # END_sbr_e

  dplt.draw_paths(
    full_paths,
    figure=sbr_fig,
    line={"width": 4, "color": color[order]},
    showlegend=False,
  )

save_fig(
  sbr_fig, "tutorial-sbr-paths.png", azim=+40, elev=10, dist=1.0
)
