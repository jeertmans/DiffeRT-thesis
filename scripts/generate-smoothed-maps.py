# ruff: noqa: F722
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "differt2d>=0.4.0",
#     "jax[cuda]>=0.9.2",
#     "jaxtyping>=0.3.9",
#     "matplotlib>=3.10.8",
# ]
# ///
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from differt2d.logic import sigmoid
from differt2d.scene import Scene
from differt2d.utils import P0, received_power
from matplotlib.colors import LogNorm

scene = Scene.basic_scene()
scene = scene.with_transmitters(Tx=scene.transmitters["tx"])

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

X, Y = scene.grid(500)

for grad in [False, True]:
    for approx in [False, True]:
        fig, ax = plt.subplots(tight_layout=True)
        fig.set_figwidth(6.0 / 2.54)
        scene.plot(
            ax,
            objects_kwargs=dict(color="red"),
            transmitters_kwargs=dict(marker="x", color="black", s=25),
            receivers=False,
            annotate=False,
        )

        P = scene.accumulate_on_receivers_grid_over_paths(
            X,
            Y,
            fun=received_power,
            reduce_all=True,
            grad=grad,
            approx=approx,
            function=sigmoid,
            alpha=50.0,
        )  # type: ignore

        if grad:
            dP = jnp.linalg.norm(P, axis=-1)
            dP = jnp.nan_to_num(dP)
            print("A", dP.min(), dP.max())
            im = ax.pcolormesh(
                X,
                Y,
                dP,
                norm=LogNorm(vmin=0.1, vmax=6486.8857),
                rasterized=True,
                # antialiased=True,
                zorder=-1,
            )
        else:
            PdB = 10.0 * jnp.log10(P / P0)
            print("B", PdB.min(), PdB.max())
            im = ax.pcolormesh(
                X,
                Y,
                PdB,
                vmin=-75.18792,
                vmax=0.83934015,
                rasterized=True,
                # antialiased=True,
                zorder=-1,
            )

        ax.set_axis_off()
        ax.set_aspect("equal")

        folder = Path(__file__).parent.parent / "pgf"
        folder.mkdir(exist_ok=True)

        suffix = "approx" if approx else "exact"

        if grad:
            fig.savefig(
                folder / f"power-gradient-{suffix}.pgf", dpi=500, bbox_inches="tight"
            )
        else:
            fig.savefig(
                folder / f"power-map-{suffix}.pgf", dpi=500, bbox_inches="tight"
            )
