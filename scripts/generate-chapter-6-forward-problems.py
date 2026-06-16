# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt==0.8.1",
#     "jax[cuda]>=0.9.2",
#     "kaleido>=1.2.0",
#     "pillow>=12.1.1",
#     "plotly>=6.6.0",
#     "sionna-rt",
#     "sionna-vispy[recommended]>=1.2.0",
#     "tqdm>=4.67.3",
# ]
#
# [tool.uv.sources]
# sionna-rt = { git = "https://github.com/jeertmans/sionna-rt", branch = "fix-diffraction" }
# ///
import io

import drjit as dr
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import plotly.graph_objects as go
import sionna.rt
from differt.plotting import set_defaults
from differt.scene import (
    TriangleScene,
    download_sionna_scenes,
    get_sionna_scene,
)
from PIL import Image
from plotly.colors import convert_to_RGB_255
from sionna.rt import PolarizedAntennaPattern, r_hat, register_antenna_pattern
from tqdm import trange

mpl.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.size": 10,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "image.cmap": "plasma",
    },
)

download_sionna_scenes("v2.0.0")


file = get_sionna_scene("simple_street_canyon")
differt_scene = TriangleScene.load_xml(file)

set_defaults("plotly")


def compute_path_metrics(paths: sionna.rt.Paths):
    a_cir, tau_cir = paths.cir(normalize_delays=False, out_type="numpy")
    a = a_cir[0, 0, 0, 0, :, 0]
    tau = tau_cir[0, 0, 0, 0, :]
    amplitude = np.abs(a)
    power_linear = np.abs(a) ** 2
    power_db = 10 * np.log10(np.maximum(power_linear, np.finfo(float).tiny))
    return tau, amplitude, power_linear, power_db


def save_path_metrics_csv(path_type: str, paths: sionna.rt.Paths):
    tau, amplitude, power_linear, power_db = compute_path_metrics(paths)
    output = np.column_stack(
        (
            np.arange(len(amplitude), dtype=int),
            tau / 1e-9,
            amplitude,
            power_linear,
            power_db,
        )
    )
    output_path = f"data/{path_type}-path-metrics.txt"
    np.savetxt(
        output_path,
        output,
        delimiter=",",
        header="path_index,delay_ns,amplitude,power_linear,power_db",
        comments="",
    )
    print(f"Saved {len(amplitude)} {path_type} path metrics to {output_path}")


def draw_tx_rx_on_plotly_fig(fig: go.Figure, tx: np.ndarray, rx: np.ndarray):
    fig.add_trace(
        go.Scatter3d(
            x=[tx[0]],
            y=[tx[1]],
            z=[tx[2]],
            mode="markers+text",
            marker=dict(color="black", size=6, symbol="x"),
            text=["TX"],
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
            textfont=dict(color="black", size=30, family="Libertinus Serif"),
            showlegend=False,
        )
    )


def draw_sionna_paths_on_plotly_fig(
    fig: go.Figure, paths: sionna.rt.Paths, color="red", name="Path"
):
    vertices = paths.vertices.numpy()
    valid = paths.valid.numpy()
    types = paths.interactions.numpy()
    max_depth = vertices.shape[0]

    num_paths = vertices.shape[-2]
    if num_paths == 0:
        return  # Nothing to do

    # Build sources and targets
    src_positions, tgt_positions = paths.sources, paths.targets
    src_positions = src_positions.numpy().T
    tgt_positions = tgt_positions.numpy().T

    num_src = src_positions.shape[0]
    num_tgt = tgt_positions.shape[0]

    # Merge device and antenna dimensions if required
    if not paths.synthetic_array:
        # The dimension corresponding to the number of antenna patterns
        # is removed as it is a duplicate
        num_rx = paths.num_rx
        rx_array_size = paths.rx_array.array_size
        num_rx_patterns = len(paths.rx_array.antenna_pattern.patterns)
        #
        num_tx = paths.num_tx
        tx_array_size = paths.tx_array.array_size
        num_tx_patterns = len(paths.tx_array.antenna_pattern.patterns)
        #
        vertices = np.reshape(
            vertices,
            [
                max_depth,
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
                3,
            ],
        )
        valid = np.reshape(
            valid,
            [
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
            ],
        )
        types = np.reshape(
            types,
            [
                max_depth,
                num_rx,
                num_rx_patterns,
                rx_array_size,
                num_tx,
                num_tx_patterns,
                tx_array_size,
                -1,
            ],
        )
        vertices = vertices[:, :, 0, :, :, 0, :, :, :]
        types = types[:, :, 0, :, :, 0, :, :]
        valid = valid[:, 0, :, :, 0, :, :]
        vertices = np.reshape(vertices, [max_depth, num_tgt, num_src, -1, 3])
        valid = np.reshape(valid, [num_tgt, num_src, -1])
        types = np.reshape(types, [max_depth, num_tgt, num_src, -1])

    # Emit directly two lists of the beginnings and endings of line segments
    starts = []
    ends = []
    for rx in range(num_tgt):  # For each receiver
        for tx in range(num_src):  # For each transmitter
            for p in range(num_paths):  # For each path
                if not valid[rx, tx, p]:
                    continue
                start = src_positions[tx]
                i = 0
                while i < max_depth:
                    t = types[i, rx, tx, p]
                    if t == sionna.rt.constants.InteractionType.NONE:
                        break
                    end = vertices[i, rx, tx, p]
                    starts.append(start)
                    ends.append(end)
                    start = end
                    i += 1
                # Explicitly add the path endpoint
                starts.append(start)
                ends.append(tgt_positions[rx])

    starts = np.vstack(starts)
    ends = np.vstack(ends)
    sep = np.full_like(starts, np.nan)
    x, y, z = np.stack((starts, ends, sep), axis=1).reshape(-1, 3).T
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
        )
    )


def save_fig(
    fig: go.Figure, filename: str, azim: 45, elev: float = 90, dist: float = 1
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
    fig_bytes = fig.to_image(format="png", scale=2)
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
        max(0, miny - 3) : min(maxy + 3, height),
        max(0, minx - 3) : min(maxx + 3, width),
    ]
    img = Image.fromarray(cropped_img_array)
    img = img.resize((width // 4, height // 4), resample=Image.LANCZOS)
    img.save(filename)


fig = differt_scene.plot()
save_fig(fig, "images/street-canyon.png", azim=-100, elev=50.0, dist=2.03)

print("Scene visualization saved to images/street-canyon.png")

sionna_scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
sionna_scene.frequency = 28e9

sionna_scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

tx = [-33, 11, 32]
rx = [+25, 0, 1.5]
sionna_scene.rx_array = sionna_scene.tx_array
sionna_scene.add(sionna.rt.Transmitter(name="tx", position=tx, orientation=[0, 0, 0]))

sionna_scene.add(sionna.rt.Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

p_solver = sionna.rt.PathSolver()

# Compute propagation paths
paths = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=True,
    synthetic_array=False,
    seed=41,
)
refl_paths = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=False,
    synthetic_array=False,
    seed=41,
)
diffr_paths = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=False,
    specular_reflection=False,
    diffuse_reflection=False,
    refraction=False,
    diffraction=True,
    synthetic_array=False,
    seed=41,
)

draw_tx_rx_on_plotly_fig(fig, tx, rx)
refl_fig = go.Figure(fig)
draw_sionna_paths_on_plotly_fig(refl_fig, refl_paths, color="red")
# refl_fig.show()
diffr_fig = go.Figure(fig)
draw_sionna_paths_on_plotly_fig(diffr_fig, diffr_paths, color="orange")
# diffr_fig.show()

# ============================================================================
# From Paths to Channel Coefficients
# ============================================================================

save_path_metrics_csv("reflection", refl_paths)
save_path_metrics_csv("diffraction", diffr_paths)

save_fig(
    refl_fig, "images/street-canyon-reflection-paths.png", azim=0, elev=3.0, dist=0.8
)
save_fig(
    diffr_fig,
    "images/street-canyon-diffraction-paths.png",
    azim=-190,
    elev=50.0,
    dist=0.8,
)


def compute_principal_path_time_series(
    paths: sionna.rt.Paths, sampling_frequency: float, num_time_steps: int
):
    a_mob, _ = paths.cir(
        sampling_frequency=sampling_frequency,
        num_time_steps=num_time_steps,
        normalize_delays=False,
        out_type="numpy",
    )

    coefficients_flat = np.asarray(a_mob).reshape(-1, num_time_steps)
    amplitudes = np.abs(coefficients_flat)
    principal_index = int(np.argmax(np.max(amplitudes, axis=1)))
    principal_coefficients = coefficients_flat[principal_index]

    return principal_coefficients


print("\n=== Mobility Scenario: Principal Path Coefficients ===")

rx_velocity = [10.0, 0.0, 0.0]  # Moving at 10 m/s along x-axis

sampling_frequency = 1e1
dt = 1 / sampling_frequency
num_time_steps = 150
time_s, dt = np.linspace(0, 1, 101, retstep=True)

los_cir_mob, _ = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=True,
    specular_reflection=False,
    diffuse_reflection=False,
    refraction=False,
    diffraction=False,
    synthetic_array=True,
    seed=41,
).cir(
    normalize_delays=False,
    out_type="numpy",
)
coeff_los = [np.abs(los_cir_mob.squeeze())]
refl_cir_mob, _ = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=False,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=False,
    synthetic_array=True,
    seed=41,
).cir(
    normalize_delays=False,
    out_type="numpy",
)
refl_index = np.argmax(np.abs(refl_cir_mob.squeeze()))
coeff_refl = [np.abs(refl_cir_mob.squeeze())[refl_index]]
diffr_cir_mob, _ = p_solver(
    scene=sionna_scene,
    max_depth=3,
    los=False,
    specular_reflection=False,
    diffuse_reflection=False,
    refraction=False,
    diffraction=True,
    synthetic_array=True,
    seed=41,
).cir(
    normalize_delays=False,
    out_type="numpy",
)
diffr_index = np.argmax(np.abs(diffr_cir_mob.squeeze()))
coeff_diffr = [np.abs(diffr_cir_mob.squeeze())[diffr_index]]

print(f"{tx=}, {rx=}, {sionna_scene.get('rx').position=}")

print(
    f"Initial RX-TX distance: {np.linalg.norm(np.array(sionna_scene.get('rx').position) - np.array(tx)):.2f} m"
)

for _ in time_s[1:]:
    sionna_scene.get("rx").position += dt * np.array(rx_velocity)

    los_cir_mob, _ = p_solver(
        scene=sionna_scene,
        max_depth=3,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
        seed=41,
    ).cir(
        normalize_delays=False,
        out_type="numpy",
    )
    coeff_los.append(np.abs(los_cir_mob.squeeze()))
    refl_cir_mob, _ = p_solver(
        scene=sionna_scene,
        max_depth=3,
        los=False,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=True,
        seed=41,
    ).cir(
        normalize_delays=False,
        out_type="numpy",
    )
    coeff_refl.append(np.abs(refl_cir_mob.squeeze())[refl_index])
    diffr_cir_mob, _ = p_solver(
        scene=sionna_scene,
        max_depth=3,
        los=False,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=False,
        diffraction=True,
        synthetic_array=True,
        seed=41,
    ).cir(
        normalize_delays=False,
        out_type="numpy",
    )
    coeff_diffr.append(np.abs(diffr_cir_mob.squeeze())[diffr_index])

print(f"{tx=}, {rx=}, {sionna_scene.get('rx').position=}")
print(
    f"Final RX-TX distance: {np.linalg.norm(np.array(sionna_scene.get('rx').position) - np.array(tx)):.2f} m"
)

diversity_distance = time_s * rx_velocity[0]
power_los_db = 10 * np.log10(np.maximum(np.abs(coeff_los) ** 2, np.finfo(float).tiny))
power_refl_db = 10 * np.log10(np.maximum(np.abs(coeff_refl) ** 2, np.finfo(float).tiny))
power_diffr_db = 10 * np.log10(
    np.maximum(np.abs(coeff_diffr) ** 2, np.finfo(float).tiny)
)

output = np.column_stack(
    (
        diversity_distance,
        power_los_db,
        power_refl_db,
        power_diffr_db,
    )
)
output_path = "data/mobility-principal-path-power.txt"
np.savetxt(
    output_path,
    output,
    delimiter=",",
    header="distance_m,power_los_db,power_refl_db,power_diffr_db",
    comments="",
)
print(f"Saved mobility principal path power profiles to {output_path}")

# Undo mobility for next sections
sionna_scene.get("rx").position = rx

# ============================================================================
# Radio Map (Coverage Map)
# ============================================================================


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


print("\n=== Radio Map Computation ===")

# Compute radio map
rm_solver = sionna.rt.RadioMapSolver()
rm = rm_solver(
    scene=sionna_scene,
    los=True,
    specular_reflection=True,
    refraction=False,
    diffraction=True,
    max_depth=3,
    cell_size=[0.1, 0.1],
    samples_per_tx=int(1e8),
)
power_db = 10 * np.log10(rm.transmitter_radio_map("path_gain", 0).numpy())
vmin = np.min(power_db, where=np.isfinite(power_db), initial=np.inf)
vmax = np.max(power_db, where=np.isfinite(power_db), initial=-np.inf)
rm_refl = rm_solver(
    scene=sionna_scene,
    los=True,
    specular_reflection=True,
    refraction=False,
    diffraction=False,
    max_depth=3,
    cell_size=[0.1, 0.1],
    samples_per_tx=int(1e8),
)
rm_diffr = rm_solver(
    scene=sionna_scene,
    los=False,
    specular_reflection=False,
    refraction=False,
    diffraction=True,
    max_depth=3,
    cell_size=[0.1, 0.1],
    samples_per_tx=int(1e8),
)

mesh = differt_scene.mesh
((x_min, x_max), (y_min, y_max), _) = mesh.bounding_box.T
fig, ax = plt.subplots()
fig.set_figwidth(7.5 / 2.54)
draw_mesh_2d_mpl(mesh, ax)
im = ax.imshow(
    10 * np.log10(rm_refl.transmitter_radio_map("path_gain", 0).numpy()),
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    alpha=0.8,
    zorder=0.5,
    rasterized=True,
    vmin=vmin,
    vmax=vmax,
)
fig.colorbar(im, ax=ax, location="top", pad=0.05, label="Path gain (dB)")
ax.scatter(tx[0], tx[1], marker="x", c="black", s=25, zorder=5)
ax.set_axis_off()
ax.set_aspect("equal")
plt.savefig(
    "pgf/path-gain-reflection.pgf",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.0,
)
fig, ax = plt.subplots()
fig.set_figwidth(7.5 / 2.54)
draw_mesh_2d_mpl(mesh, ax)
im = ax.imshow(
    10 * np.log10(rm_diffr.transmitter_radio_map("path_gain", 0).numpy()),
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    alpha=0.8,
    zorder=0.5,
    rasterized=True,
    vmin=vmin,
    vmax=vmax,
)
fig.colorbar(im, ax=ax, location="top", pad=0.05, label="Path gain (dB)")
ax.scatter(tx[0], tx[1], marker="x", c="black", s=25, zorder=5)
ax.set_axis_off()
ax.set_aspect("equal")
plt.savefig(
    "pgf/path-gain-diffraction.pgf",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.0,
)

# ============================================================================
# Antenna Pattern Optimization (Gradient-based)
# ============================================================================

print("\n=== Antenna Pattern Optimization ===")


# Define a trainable antenna pattern with spherical Gaussian
class TrainableVPattern:
    """Trainable vertically polarized antenna pattern function

    Defined via a spherical Gaussian with trainable mean direction
    and sharpness parameter.
    """

    def __init__(self, opt: mi.ad.Optimizer):
        # Add trainable target directions to optimizer
        opt["theta_t"] = mi.Float(dr.pi / 2)
        opt["phi_t"] = mi.Float(0)

        # Add trainable sharpness to optimizer
        opt["lambda"] = mi.Float(0.5)

        self.opt = opt

    def __call__(self, theta, phi):
        mu = r_hat(self.opt["theta_t"], self.opt["phi_t"])
        v = r_hat(theta, phi)
        gain = (
            2
            * self.opt["lambda"]
            * dr.rcp(1 - dr.exp(-2 * self.opt["lambda"]))
            * dr.exp(self.opt["lambda"] * (dr.dot(mu, v) - 1))
        )
        c_theta_real = dr.sqrt(gain)
        return mi.Complex2f(c_theta_real, 0)


def trainable_pattern_factory(*, opt, polarization, polarization_model="tr38901_2"):
    """Factory method for trainable antenna pattern"""
    return PolarizedAntennaPattern(
        v_pattern=TrainableVPattern(opt),
        polarization=polarization,
        polarization_model=polarization_model,
    )


# Register the pattern
register_antenna_pattern("trainable", trainable_pattern_factory)

# Create a new scene for optimization
opt_scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
opt_scene.frequency = 28e9

# Create optimizer
opt = mi.ad.Adam(lr=5e-2)

# Set up transmitter with trainable antenna
opt_scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="trainable",
    opt=opt,
    polarization="V",
)

opt_scene.rx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

opt_scene.add(sionna.rt.Transmitter(name="tx", position=tx, orientation=[0, 0, 0]))

opt_scene.add(sionna.rt.Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

# Configure solver for gradient computation
p_solver = sionna.rt.PathSolver()
p_solver.loop_mode = "evaluated"

# Compute paths
opt_paths = p_solver(
    scene=opt_scene,
    max_depth=2,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    diffraction=False,
    synthetic_array=False,
    seed=41,
)

# Extract field components
a_r, a_i = opt_paths.a

# Compute total received power
power = dr.sum(a_r**2 + a_i**2)

# Compute gradients
dr.backward(-power)

print(f"{opt=}")

print(f"Initial received power: {float(power.numpy()):.4e}")

# Perform optimization steps and track power
num_steps = 10_000
powers_linear = [float(power.numpy())]
thetas = [opt["theta_t"].numpy()]
phis = [opt["phi_t"].numpy()]

for step in trange(num_steps):
    # Recompute paths with updated parameters
    opt_paths = p_solver(
        scene=opt_scene,
        max_depth=2,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=False,
        seed=41,
    )

    a_r, a_i = opt_paths.a
    power = dr.sum(a_r**2 + a_i**2)
    dr.backward(-power)
    opt.step()

    powers_linear.append(float(power.numpy()))
    thetas.append(opt["theta_t"].numpy())
    phis.append(opt["phi_t"].numpy())

    if (step + 1) % (num_steps // 10) == 0:
        print(f"Step {step + 1}: Power = {float(power.numpy()):.4e}")

steps = np.arange(len(powers_linear))
powers_db = 10 * np.log10(np.maximum(np.array(powers_linear), np.finfo(float).tiny))

antenna_optimization_data = np.column_stack(
    [
        steps,
        powers_db,
        np.rad2deg(thetas),
        np.rad2deg(phis),
    ]
)
np.savetxt(
    "data/antenna-pattern-optimization.txt",
    antenna_optimization_data,
    delimiter=",",
    header="step,power_db,theta_deg,phi_deg",
    comments="",
)

print(f"\nFinal received power: {powers_db[-1]:.2f} dB")
print(
    f"Power improvement: {(powers_linear[-1] - powers_linear[0]) / powers_linear[0] * 100:.1f}%"
)
print(f"Final theta: {np.rad2deg(thetas[-1])}°")
print(f"Final phi: {np.rad2deg(phis[-1])}°")

# ============================================================================
# Transmitter Orientation Optimization
# ============================================================================

print("\n\n=== Transmitter Orientation Optimization ===")

# Create a new scene for orientation optimization
orient_scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
orient_scene.frequency = 28e9

orient_scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V",
)

orient_scene.rx_array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

# Add transmitter and receiver
tx_orient = sionna.rt.Transmitter(name="tx", position=tx, orientation=[0, 0, 0])
orient_scene.add(tx_orient)

orient_scene.add(sionna.rt.Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

# Configure solver for gradient computation
p_solver_orient = sionna.rt.PathSolver()
p_solver_orient.loop_mode = "evaluated"

# Create optimizer for transmitter orientation
opt_orient = mi.ad.Adam(lr=0.05)
opt_orient["alpha"] = mi.Float(0.0)  # Roll angle
opt_orient["beta"] = mi.Float(0.0)  # Pitch angle
opt_orient["gamma"] = mi.Float(0.0)  # Yaw angle

print("\nOptimizing transmitter orientation (alpha, beta, gamma)...")
print("Initial orientation: [0.00°, 0.00°, 0.00°]")

num_steps_orient = 10_000
powers_orient_linear = []
alphas = []
betas = []
gammas = []

for step in trange(num_steps_orient):
    # Compute average received power over target region

    # Update transmitter orientation
    orient_scene.get("rx").orientation = [
        opt_orient["alpha"],
        opt_orient["beta"],
        opt_orient["gamma"],
    ]
    orient_paths = p_solver(
        scene=orient_scene,
        max_depth=2,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        diffraction=False,
        synthetic_array=False,
        seed=41,
    )

    a_r, a_i = orient_paths.a
    power = dr.sum(a_r**2 + a_i**2)
    dr.backward(-power)
    opt_orient.step()

    powers_orient_linear.append(float(power.numpy()))
    alphas.append(opt_orient["alpha"].numpy())
    betas.append(opt_orient["beta"].numpy())
    gammas.append(opt_orient["gamma"].numpy())

    if (step + 1) % (num_steps_orient // 10) == 0:
        print(
            f"Step {step + 1}: Avg Power = {float(power.numpy()):.4e}, "
            f"Angles = [{np.rad2deg((opt_orient['alpha']))}°, "
            f"{np.rad2deg((opt_orient['beta']))}°, "
            f"{np.rad2deg((opt_orient['gamma']))}°]"
        )

steps_array_orient = np.arange(len(powers_orient_linear))
powers_orient_db = 10 * np.log10(
    np.maximum(np.array(powers_orient_linear), np.finfo(float).tiny)
)

orientation_optimization_data = np.column_stack(
    [
        steps_array_orient,
        powers_orient_db,
        np.rad2deg(alphas),
        np.rad2deg(betas),
        np.rad2deg(gammas),
    ]
)
np.savetxt(
    "data/transmitter-orientation-optimization.txt",
    orientation_optimization_data,
    delimiter=",",
    header="step,power_db,alpha_deg,beta_deg,gamma_deg",
    comments="",
)

print(f"\nFinal average power: {powers_orient_db[-1]:.2f} dB")
print(
    f"Power improvement: {(powers_orient_linear[-1] - powers_orient_linear[0]) / powers_orient_linear[0] * 100:.1f}%"
)
print(
    f"Final orientation: α={np.rad2deg(alphas[-1])}°, β={np.rad2deg(betas[-1])}°, γ={np.rad2deg(gammas[-1])}°"
)
