# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "sionna-rt>=1,<1.2",
#     "tqdm>=4.67.3",
# ]
# ///

import drjit as dr
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import sionna.rt
from tqdm import trange


def linear_to_db(power_linear, eps=1e-30):
    power_linear = np.asarray(power_linear, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(power_linear, eps))


# ============================================================================
# Setup Scene for Inverse Problems
# ============================================================================

print("=== Inverse Problems: Localization and Calibration ===\n")


def build_localization_scene(tx_position, rx_position):
    scene = sionna.rt.load_scene(
        sionna.rt.scene.simple_street_canyon, merge_shapes=False
    )
    scene.frequency = 28e9
    scene.tx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = sionna.rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(
        sionna.rt.Transmitter(name="tx", position=tx_position, orientation=[0, 0, 0])
    )
    scene.add(
        sionna.rt.Receiver(name="rx", position=rx_position, orientation=[0, 0, 0])
    )
    return scene


# Ground truth receiver position
true_rx_position = np.array([10.0, 0.0, 1.5])
tx_primary = [-33, 11, 32]
tx_secondary = [33, -11, 32]
tx_tertiary = [0, 28, 32]

loc_scene_primary = build_localization_scene(tx_primary, true_rx_position.tolist())
loc_scene_secondary = build_localization_scene(tx_secondary, true_rx_position.tolist())

# ============================================================================
# Section 1: Inverse Localization
# ============================================================================

print("\n=== Inverse Localization ===")

p_solver = sionna.rt.PathSolver()
p_solver.loop_mode = "evaluated"

# Compute channels at ground truth position (simulated "measurements")
print(f"Ground truth receiver position: {true_rx_position}")

paths_gt = p_solver(
    scene=loc_scene_primary,
    max_depth=2,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    synthetic_array=False,
    seed=41,
)
paths_gt_secondary = p_solver(
    scene=loc_scene_secondary,
    max_depth=2,
    los=True,
    specular_reflection=True,
    diffuse_reflection=False,
    refraction=False,
    synthetic_array=False,
    seed=41,
)


def total_gain(paths):
    a_real, a_imag = paths.a
    g = dr.sum(dr.square(a_real) + dr.square(a_imag))
    return g


power_gt = dr.detach(total_gain(paths_gt))
power_gt_secondary = dr.detach(total_gain(paths_gt_secondary))

print(f"Ground truth received power (TX1): {float(power_gt.numpy()):.4e}")
print(f"Ground truth received power (TX2): {float(power_gt_secondary.numpy()):.4e}")

# Perform localization by optimizing receiver position
opt_loc = mi.ad.Adam(lr=0.2)
opt_loc["rx_x"] = mi.Float(0.0)
opt_loc["rx_y"] = mi.Float(5.0)

print("\nOptimizing receiver position...")
print("Initial guess: [0.0, 5.0, 1.5]")

num_steps_loc = 400
positions = []
losses = []
powers_loc = []

true_range_primary = np.linalg.norm(
    true_rx_position - np.asarray(tx_primary, dtype=np.float64)
)
true_range_secondary = np.linalg.norm(
    true_rx_position - np.asarray(tx_secondary, dtype=np.float64)
)
true_range_tertiary = np.linalg.norm(
    true_rx_position - np.asarray(tx_tertiary, dtype=np.float64)
)


def to_scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


for step in trange(num_steps_loc):
    candidate_position = [
        opt_loc["rx_x"],
        opt_loc["rx_y"],
        mi.Float(1.5),
    ]
    loc_scene_primary.get("rx").position = candidate_position
    loc_scene_secondary.get("rx").position = candidate_position

    # Compute paths
    paths_primary = p_solver(
        scene=loc_scene_primary,
        max_depth=2,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=False,
        seed=41,
    )
    paths_secondary = p_solver(
        scene=loc_scene_secondary,
        max_depth=2,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=False,
        seed=41,
    )
    power_primary = total_gain(paths_primary)
    power_secondary = total_gain(paths_secondary)

    rx_x = opt_loc["rx_x"]
    rx_y = opt_loc["rx_y"]
    rx_z = mi.Float(1.5)

    range_primary = dr.sqrt(
        dr.square(rx_x - tx_primary[0])
        + dr.square(rx_y - tx_primary[1])
        + dr.square(rx_z - tx_primary[2])
    )
    range_secondary = dr.sqrt(
        dr.square(rx_x - tx_secondary[0])
        + dr.square(rx_y - tx_secondary[1])
        + dr.square(rx_z - tx_secondary[2])
    )
    range_tertiary = dr.sqrt(
        dr.square(rx_x - tx_tertiary[0])
        + dr.square(rx_y - tx_tertiary[1])
        + dr.square(rx_z - tx_tertiary[2])
    )

    loss = (
        dr.square(range_primary - true_range_primary)
        + dr.square(range_secondary - true_range_secondary)
        + dr.square(range_tertiary - true_range_tertiary)
    )
    dr.backward(loss)

    opt_loc.step()

    positions.append(
        [to_scalar(opt_loc["rx_x"].numpy()), to_scalar(opt_loc["rx_y"].numpy()), 1.5]
    )
    losses.append(to_scalar(loss.numpy()))
    powers_loc.append(to_scalar(power_primary.numpy()))

    if (step + 1) % (num_steps_loc // 10) == 0:
        print(
            f"Step {step + 1}: Loss = {to_scalar(loss.numpy()):.4e}, "
            f"Position = [{to_scalar(opt_loc['rx_x'].numpy()):.4f}, {to_scalar(opt_loc['rx_y'].numpy()):.4f}, 1.5]"
        )

positions = np.array(positions)
final_position = positions[-1]
position_error = np.linalg.norm(final_position - true_rx_position)

print(f"\nFinal position: {final_position}")
print(f"Ground truth: {true_rx_position}")
print(f"Position error: {position_error:.4f} m")

# Save localization optimization trace
loc_steps = np.arange(1, num_steps_loc + 1)
powers_loc_db = linear_to_db(powers_loc)
power_gt_db = float(linear_to_db([float(power_gt.numpy())])[0])
loc_results = np.column_stack(
    (
        loc_steps,
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        np.asarray(losses),
        powers_loc_db,
        np.full_like(loc_steps, power_gt_db, dtype=np.float64),
    )
)
np.savetxt(
    "data/ch6-inverse-localization-optimization.txt",
    loc_results,
    delimiter=",",
    header="step,rx_x_m,rx_y_m,rx_z_m,loss,power_db,ground_truth_power_db",
    comments="",
)

# Plot localization results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
steps_array = np.arange(len(losses))

# Plot 1: Loss
axes[0, 0].plot(steps_array, losses, color="red", label="Loss")
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("Loss (log scale)")
axes[0, 0].set_title("Loss Convergence")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Received power
axes[0, 1].plot(steps_array, powers_loc, color="blue", label="Power")
axes[0, 1].axhline(
    y=float(power_gt.numpy()), color="green", linestyle="--", label="Ground truth"
)
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("Received Power")
axes[0, 1].set_title("Received Power")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: 2D trajectory
scatter = axes[1, 0].scatter(
    positions[:, 0],
    positions[:, 1],
    c=steps_array,
    cmap="viridis",
    s=25,
    label="Trajectory",
)
axes[1, 0].scatter(
    [true_rx_position[0]],
    [true_rx_position[1]],
    s=200,
    c="red",
    marker="*",
    label="Ground Truth",
)
axes[1, 0].scatter(
    [positions[0, 0]],
    [positions[0, 1]],
    s=120,
    c="orange",
    marker="D",
    label="Initial Guess",
)
axes[1, 0].set_xlabel("X position (m)")
axes[1, 0].set_ylabel("Y position (m)")
axes[1, 0].set_title("2D Position Trajectory (X-Y)")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
fig.colorbar(scatter, ax=axes[1, 0], label="Step")

# Plot 4: Height evolution
axes[1, 1].plot(
    steps_array,
    positions[:, 2],
    color="purple",
    marker="o",
    markersize=3,
    label="Height",
)
axes[1, 1].axhline(
    y=true_rx_position[2], color="red", linestyle="--", label="Ground truth"
)
axes[1, 1].set_xlabel("Step")
axes[1, 1].set_ylabel("Z position (m)")
axes[1, 1].set_title("Height Evolution")
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

fig.suptitle("Inverse Localization Results")
plt.tight_layout()
plt.savefig("images/inverse-localization.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# Section 2: Radio Materials Calibration
# ============================================================================

print("\n\n=== Radio Materials Calibration ===")

scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)

scene.add(sionna.rt.Transmitter("tx", position=(-2.0, 0.0, 1)))
scene.add(sionna.rt.Receiver("rx", position=(2.0, 0.0, 1)))

scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1, pattern="iso", polarization="V"
)
scene.rx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1, pattern="iso", polarization="V"
)

# Instantiate the radio material
my_mat = sionna.rt.RadioMaterial(
    "my-mat", thickness=0.1, relative_permittivity=5.0, conductivity=1
)
# Assign the radio material to the reflector
scene.objects["reflector"].radio_material = my_mat
# To avoid confusion, discard the radio material initially loaded with the scene
scene.remove("reflector-mat")

# Print the material to visualize its parameters
print(scene.radio_materials)

trainable_scene = sionna.rt.load_scene(
    sionna.rt.scene.simple_reflector, merge_shapes=False
)

trainable_scene.add(sionna.rt.Transmitter("tx", position=(-2.0, 0.0, 1)))
trainable_scene.add(sionna.rt.Receiver("rx", position=(2.0, 0.0, 1)))

trainable_scene.tx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1, pattern="iso", polarization="V"
)
trainable_scene.rx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1, pattern="iso", polarization="V"
)

# Adam optimizer
learning_rate = 3e-2
opt = mi.ad.Adam(lr=learning_rate)


def logit_2_conductivity(logit):
    max_conductivity = 10.0
    return sionna.rt.utils.sigmoid(logit) * max_conductivity


# Trainable variable is the logit of the conductivity
# It is initialized to an arbitrary value
opt["logit_conductivity"] = mi.Float(-5.0)

# Instantiate the radio material
# The conductivity is initialized using the trainable variable
trainable_mat = sionna.rt.RadioMaterial(
    "my-trainable-mat",
    thickness=0.1,
    relative_permittivity=5.0,
    conductivity=logit_2_conductivity(opt["logit_conductivity"]),
)  # Use the trainable variable
# Assign the radio material to the reflector
trainable_scene.objects["reflector"].radio_material = trainable_mat
# To avoid confusion, discard the radio material initially loaded with the scene
trainable_scene.remove("reflector-mat")

# Print the material to visualize its parameters
print(trainable_scene.radio_materials)

solver = sionna.rt.PathSolver()
solver.loop_mode = "evaluated"


def total_gain(paths):
    a_real, a_imag = paths.a
    g = dr.sum(dr.square(a_real) + dr.square(a_imag))
    return g


# Normalized absolute error
# The `dr.detach()` function stops gradient from
# propagating through its input
def nae(x, y):
    return dr.abs(x - y) * dr.detach(dr.rcp(y))


num_iterations = 200


# Ground-truth
scene.radio_materials["my-mat"].conductivity = 0.6260499535669882
ref_paths = solver(scene)
ref_gain = dr.detach(total_gain(ref_paths))

# Record the conductivity value
conductivity = []
losses = []
steps = []

# Optimization loop
for i in trange(num_iterations + 1):
    # Run simulation
    paths = solver(trainable_scene)

    # Compute loss on total gain and the gradients
    gain = total_gain(paths)
    loss = nae(gain, ref_gain)
    dr.backward(loss)
    conductivity.append(logit_2_conductivity(opt["logit_conductivity"]).numpy())
    losses.append(loss.numpy())
    steps.append(i)

    # Optimizer step
    opt.step()

    updated_conductivity = logit_2_conductivity(opt["logit_conductivity"])
    trainable_mat.conductivity = updated_conductivity

plt.figure()
plt.grid(True)
plt.plot(np.arange(0, num_iterations + 1), conductivity, label="Calibrated")
plt.hlines(
    [scene.get("my-mat").conductivity],
    1,
    num_iterations + 1,
    color="k",
    label="Ground-truth",
)
plt.xlabel("Iteration")
plt.ylabel("Conductivity [S/m]")
plt.legend()
plt.show()

# Save material calibration optimization trace
cal_results = np.column_stack((steps, conductivity, losses))
np.savetxt(
    "data/ch6-material-calibration-optimization.txt",
    cal_results,
    delimiter=",",
    header="step,estimated_conductivity_s_per_m,loss_nae",
    comments="",
)

print("\n=== All visualizations saved to images/ ===")
print("Generated files:")
print("  - inverse-localization.png")
print("  - materials-calibration.png")
print("\nSaved optimization traces to data/:")
print("  - ch6-inverse-localization-optimization.txt")
print("  - ch6-material-calibration-optimization.txt")
