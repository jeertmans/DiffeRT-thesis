#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jax[cpu]>=0.8.1",
#     "equinox>=0.13.5",
#     "matplotlib>=3.10.8",
#     "numpy>=1.26.0",
#     "plotly>=5.20.0",
#     "kaleido>=0.2.1",
#     "jaxtyping>=0.3.6",
#     "differt>=0.8.2",
#     "pillow>=10.0.0",
#     "sampling-paths @ git+https://github.com/jeertmans/sampling-paths.git",
# ]
# ///

import os
import subprocess
import urllib.request
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import plotly.graph_objects as go
from sampling_paths.model import Model
from sampling_paths.utils import geometric_transformation, random_scene, unpack_scene


def pearson_corr(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)


# Force CPU JAX
os.environ["JAX_PLATFORMS"] = "cpu"

# Resolve directories relative to script location
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
images_dir = repo_root / "images"
weights_dir = repo_root / "weights"
data_dir = repo_root / "data" / "study-model-embeddings"

os.makedirs(images_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Repository Root: {repo_root}")
print(f"Images Directory: {images_dir}")
print(f"Weights Directory: {weights_dir}")


def download_weights(order):
    url = f"https://github.com/jeertmans/sampling-paths/releases/download/npjwt2026/model_ws_{order}.eqx"
    dest_path = weights_dir / f"model_ws_{order}.eqx"
    if not dest_path.exists():
        print(f"Downloading pre-trained weights for order {order} from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved weights to {dest_path}")
    else:
        print(f"Weights for order {order} already exist at {dest_path}")
    return dest_path


def generate_scene_and_bounds():
    key = jr.key(1234)
    scene_unmasked = random_scene(key=key)
    original_mask = np.array(scene_unmasked.mesh.mask)
    scene_basic = eqx.tree_at(
        lambda s: s.mesh, scene_unmasked, scene_unmasked.mesh.masked()
    )
    scene_swapped = eqx.tree_at(
        lambda s: (s.transmitters, s.receivers),
        scene_basic,
        (scene_basic.receivers, scene_basic.transmitters),
    )

    base_bounds = np.array(
        [[0, 12], [12, 24], [24, 36], [36, 48], [48, 60], [60, 72], [72, 74]]
    )
    active_indices = np.where(original_mask)[0]
    triangle_object_ids = []
    for idx in active_indices:
        obj_id = np.where((base_bounds[:, 0] <= idx) & (idx < base_bounds[:, 1]))[0][0]
        triangle_object_ids.append(obj_id)
    triangle_object_ids = np.array(triangle_object_ids)

    masked_object_bounds = []
    current_obj = -1
    start_idx = 0
    for i, obj_id in enumerate(triangle_object_ids):
        if obj_id != current_obj:
            if current_obj != -1:
                masked_object_bounds.append((current_obj, start_idx, i))
            current_obj = obj_id
            start_idx = i
    if current_obj != -1:
        masked_object_bounds.append((current_obj, start_idx, len(triangle_object_ids)))

    y_ticks = []
    y_labels = []
    for obj_id, start, end in masked_object_bounds:
        center = (start + end - 1) / 2.0
        y_ticks.append(center)
        y_labels.append("Floor" if obj_id == 6 else f"Building {obj_id}")

    return scene_basic, scene_swapped, masked_object_bounds, y_ticks, y_labels


def save_scene_plot(scene_basic, masked_object_bounds):
    import io

    from PIL import Image

    print("Generating Plotly scene plot...")
    fig = scene_basic.mesh.plot(backend="plotly")

    # Hide legend, grid, and set background color to transparent
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(showlegend=False)

    # Get TX and RX coordinates
    tx = np.array(scene_basic.transmitters)
    rx = np.array(scene_basic.receivers)

    # Draw TX and RX with custom styles and fonts
    fig.add_trace(
        go.Scatter3d(
            x=[tx[0]],
            y=[tx[1]],
            z=[tx[2]],
            mode="markers+text",
            marker=dict(color="black", size=6, symbol="x"),
            text=["TX"],
            textposition="top center",
            textfont=dict(color="black", size=30, family="Libertinus Serif"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[rx[0]],
            y=[rx[1]],
            z=[rx[2]],
            mode="markers+text",
            marker=dict(color="black", size=6, symbol="x"),
            text=["RX"],
            textposition="top center",
            textfont=dict(color="black", size=30, family="Libertinus Serif"),
            showlegend=False,
        )
    )

    plotly_annotations = []
    vertices_np = np.array(scene_basic.mesh.triangle_vertices)
    for obj_id, start, end in masked_object_bounds:
        obj_vertices = vertices_np[start:end].reshape(-1, 3)
        if obj_id == 6:
            x_center = -40.0
            y_center = -24.0
        else:
            x_center = float(np.mean(obj_vertices[:, 0]))
            y_center = float(np.mean(obj_vertices[:, 1]))
        z_max = float(np.max(obj_vertices[:, 2]))

        label = "Floor" if obj_id == 6 else f"Building {obj_id}"
        plotly_annotations.append(
            dict(
                showarrow=True,
                x=x_center,
                y=y_center,
                z=z_max,
                text=label,
                font=dict(color="black", size=24, family="Libertinus Serif"),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                ax=0,
                ay=-30,
            )
        )

    fig.update_layout(scene=dict(annotations=plotly_annotations))

    azim = -110
    elev = 55.0
    dist = 2.4

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

    png_path = images_dir / "random-scene-annotated.png"
    # High quality supersampled rendering and transparency-based crop
    fig_bytes = fig.to_image(format="png", scale=4)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    width, height = img.size
    img_array = np.asarray(img)
    y, x = img_array[:, :, 3].nonzero()  # get the nonzero alpha coordinates
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)
    cropped_img_array = img_array[
        max(0, miny - 5) : min(maxy + 5, height),
        max(0, minx - 5) : min(maxx + 5, width),
    ]
    img = Image.fromarray(cropped_img_array)
    img = img.resize((width // 4, height // 4), resample=Image.LANCZOS)
    img.save(str(png_path))
    print(f"Saved annotated scene PNG to {png_path}")


def extract_edge_flows(model, scene, inference=True, key=jr.key(1234)):
    num_objects = scene.mesh.triangle_vertices.shape[0]
    pre_xyz, tx, rx = unpack_scene(scene)
    xyz = geometric_transformation(pre_xyz, tx, rx)

    objects_embeds = model.objects_encoder(xyz, active_objects=scene.mesh.mask)
    scene_embeds = model.scene_encoder(objects_embeds, active_objects=scene.mesh.mask)

    def compute_flows_mask(previous_object, interaction_index):
        mask = (
            scene.mesh.mask
            if scene.mesh.mask is not None
            else jnp.ones(num_objects, dtype=bool)
        )
        mask = mask.at[previous_object].set(False, wrap_negative_indices=False)
        if model.action_masking:
            is_second_interaction = interaction_index == 1
            v0, v1, v2 = xyz[previous_object, :, :]
            normal = jnp.cross(v1 - v0, v2 - v0)
            tx_dist = jnp.dot(normal, -v0)
            next_dists = jnp.sum((xyz - v0) * normal, axis=-1)
            same_side = (next_dists * tx_dist) >= jnp.finfo(xyz.dtype).eps
            visible = jnp.any(same_side, axis=-1)
            return jnp.where(
                is_second_interaction,
                jnp.where(visible, mask, False),
                mask,
            )
        return jax.lax.stop_gradient(mask)

    def compute_flows_weight(mask, previous_object, interaction_index):
        if not model.distance_based_weighting:
            return jnp.ones(num_objects, dtype=float)
        centers = xyz.mean(axis=1)
        previous_object_center = jnp.where(
            previous_object != -1,
            centers[previous_object],
            jnp.zeros(3),
        )
        d = jnp.linalg.norm(centers - previous_object_center, axis=-1)
        d += jnp.where(
            interaction_index == model.order - 1,
            jnp.linalg.norm(centers - jnp.array([0.0, 0.0, 1.0]), axis=-1),
            0.0,
        )
        zero_d = d == 0.0
        d = jnp.where(zero_d, 1.0, d)
        w = 1 / (d * d)
        w = jnp.where(mask & (~zero_d), w, 0.0)
        w = jnp.where(w.sum() == 0.0, 1.0, w)
        return w / w.sum()

    def apply_mask_and_weight(flows, previous_object, interaction_index):
        flows_mask = compute_flows_mask(previous_object, interaction_index)
        flows_weight = compute_flows_weight(
            flows_mask, previous_object, interaction_index
        )
        return jnp.where(flows_mask, flows, 0.0) * flows_weight

    init_edge_flows_key, scan_key = jr.split(key)
    init_state_embeds = jnp.zeros(model.state_encoder.out_size)

    init_edge_flows = model.flows(
        objects_embeds,
        scene_embeds,
        init_state_embeds,
        active_objects=scene.mesh.mask,
        inference=inference,
        key=init_edge_flows_key,
    )
    init_edge_flows = apply_mask_and_weight(
        init_edge_flows, jnp.array(-1), jnp.array(0)
    )

    partial_path_candidate = -jnp.ones(model.order, dtype=int)
    edge_flows = init_edge_flows

    all_flows = []
    all_policies = []

    keys = jr.split(scan_key, model.order)
    for interaction_index in range(model.order):
        policy = edge_flows / edge_flows.sum()
        all_flows.append(edge_flows)
        all_policies.append(policy)

        key_i = keys[interaction_index]
        next_object_key, next_edge_flows_key = jr.split(key_i)
        next_object = jr.choice(next_object_key, num_objects, p=policy)
        partial_path_candidate = partial_path_candidate.at[interaction_index].set(
            next_object
        )

        if interaction_index < model.order - 1:
            state_embeds = model.state_encoder(
                partial_path_candidate,
                objects_embeds,
                active_objects=scene.mesh.mask,
            )
            edge_flows = model.flows(
                objects_embeds,
                scene_embeds,
                state_embeds,
                active_objects=scene.mesh.mask,
                inference=inference,
                key=next_edge_flows_key,
            )
            edge_flows = apply_mask_and_weight(
                edge_flows, next_object, interaction_index + 1
            )

    return jnp.stack(all_flows), jnp.stack(all_policies), partial_path_candidate


def save_pgfplots_data(filename, data):
    filepath = data_dir / filename
    # If data has only 1 row, duplicate it to satisfy PGFPlots matrix plot requirement
    is_single_row = data.ndim == 2 and data.shape[0] == 1
    with open(filepath, "w") as f:
        f.write("col row val\n")
        rows = 2 if is_single_row else data.shape[0]
        for r in range(rows):
            data_r = 0 if is_single_row else r
            for c in range(data.shape[1]):
                f.write(f"{c} {r} {data[data_r, c]:.6f}\n")
            f.write("\n")
    print(f"Saved data table to {filepath}")


def compile_latex(name):
    print(f"Compiling standalone PDF for {name}.tex...")
    try:
        res = subprocess.run(
            ["lualatex", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
            cwd=images_dir,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        try:
            res = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"{name}.tex",
                ],
                cwd=images_dir,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("LaTeX compiler not found. Skipping compilation.")
            return False

    if res.returncode == 0:
        print(f"  -> SUCCESS: Compiled {name}.pdf successfully!")
        return True
    else:
        print(
            f"  -> FAILURE: Failed to compile {name}.tex (exit code: {res.returncode})"
        )
        print(res.stdout)
        return False


def generate_order_plots(
    order, scene_basic, scene_swapped, masked_object_bounds, y_ticks, y_labels
):
    print(f"\n================ Processing Order {order} ================")

    # Download and load weights
    weights_path = download_weights(order)
    model = Model.load_weights(
        weights_path, order=order, num_embeddings=128, width_size=256, depth=2
    )

    # 1. Extract Embeddings
    pre_xyz_b, tx_b, rx_b = unpack_scene(scene_basic)
    xyz_b = geometric_transformation(pre_xyz_b, tx_b, rx_b)
    objects_embeds_basic = model.objects_encoder(
        xyz_b, active_objects=scene_basic.mesh.mask
    )
    scene_embeds_basic = model.scene_encoder(
        objects_embeds_basic, active_objects=scene_basic.mesh.mask
    )

    pre_xyz_s, tx_s, rx_s = unpack_scene(scene_swapped)
    xyz_s = geometric_transformation(pre_xyz_s, tx_s, rx_s)
    objects_embeds_swapped = model.objects_encoder(
        xyz_s, active_objects=scene_swapped.mesh.mask
    )
    scene_embeds_swapped = model.scene_encoder(
        objects_embeds_swapped, active_objects=scene_swapped.mesh.mask
    )

    # Convert to Numpy
    embeds_basic_np = np.array(objects_embeds_basic)
    embeds_swapped_np = np.array(objects_embeds_swapped)
    scene_basic_np = np.array(scene_embeds_basic)
    scene_swapped_np = np.array(scene_embeds_swapped)

    scene_comparison = np.vstack([scene_basic_np, scene_swapped_np])

    # 2. Extract Flows & Policies
    _flows_b, policies_b, _path_b = extract_edge_flows(
        model, scene_basic, inference=True
    )
    _flows_s, policies_s, _path_s = extract_edge_flows(
        model, scene_swapped, inference=True
    )

    policies_basic_np = np.array(policies_b)
    policies_swapped_np = np.array(policies_s)

    # 3. Export Data tables
    save_pgfplots_data(f"objects_embeddings_basic_order{order}.txt", embeds_basic_np)
    save_pgfplots_data(
        f"objects_embeddings_swapped_order{order}.txt", embeds_swapped_np
    )
    save_pgfplots_data(f"scene_embeddings_order{order}.txt", scene_comparison)
    save_pgfplots_data(f"probability_vectors_basic_order{order}.txt", policies_basic_np)
    save_pgfplots_data(
        f"probability_vectors_swapped_order{order}.txt", policies_swapped_np
    )
    max_val = float(
        max(np.max(np.abs(embeds_basic_np)), np.max(np.abs(embeds_swapped_np)))
    )
    if max_val == 0.0:
        max_val = 1.0

    max_val_scene = float(
        max(np.max(np.abs(scene_basic_np)), np.max(np.abs(scene_swapped_np)))
    )
    if max_val_scene == 0.0:
        max_val_scene = 1.0

    print(f"\nStatistics for Order {order}:")
    print("  Object Embeddings:")
    print(
        f"    Basic: min={np.min(embeds_basic_np):.6f}, max={np.max(embeds_basic_np):.6f}"
    )
    print(
        f"    Swapped: min={np.min(embeds_swapped_np):.6f}, max={np.max(embeds_swapped_np):.6f}"
    )
    print(f"    Symmetric Max: {max_val:.6f}")
    print(
        f"    Pearson Correlation: {pearson_corr(embeds_basic_np.flatten(), embeds_swapped_np.flatten()):.6f}"
    )
    print("  Scene Embeddings:")
    print(
        f"    Basic: min={np.min(scene_basic_np):.6f}, max={np.max(scene_basic_np):.6f}"
    )
    print(
        f"    Swapped: min={np.min(scene_swapped_np):.6f}, max={np.max(scene_swapped_np):.6f}"
    )
    print(f"    Symmetric Max: {max_val_scene:.6f}")
    print(
        f"    Pearson Correlation: {pearson_corr(scene_basic_np, scene_swapped_np):.6f}"
    )

    return max_val, max_val_scene


def main():
    scene_basic, scene_swapped, masked_object_bounds, y_ticks, y_labels = (
        generate_scene_and_bounds()
    )
    save_scene_plot(scene_basic, masked_object_bounds)

    object_max_vals = {}
    scene_max_vals = {}
    for order in [1, 2, 3]:
        max_val, max_val_scene = generate_order_plots(
            order, scene_basic, scene_swapped, masked_object_bounds, y_ticks, y_labels
        )
        object_max_vals[order] = max_val
        scene_max_vals[order] = max_val_scene

    global_object_max = max(object_max_vals.values())
    global_scene_max = max(scene_max_vals.values())

    print("\nGlobal Colormap Limits across all orders:")
    print(f"  Object Embeddings: [-{global_object_max:.6f}, {global_object_max:.6f}]")
    print(f"  Scene Embeddings: [-{global_scene_max:.6f}, {global_scene_max:.6f}]")
    print("  Probability Vectors: [0.000000, 1.000000]")


if __name__ == "__main__":
    main()
