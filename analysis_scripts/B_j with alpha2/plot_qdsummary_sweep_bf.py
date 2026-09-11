import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from plot_bf_independent_disorder import branch_overlap


SWEEPS = (
    {
        "output": "Analytic_Stability_pi4.png",
        "reference": "Stability_pi4.png",
        "x_kind": "p",
        "x_values": np.linspace(0.0, 0.5, 20),
        "theta": np.pi / 4,
        "x_label": r"$p$",
    },
    {
        "output": "Analytic_Stability_pi8.png",
        "reference": "Stability_pi8.png",
        "x_kind": "p",
        "x_values": np.linspace(0.0, 0.5, 20),
        "theta": np.pi / 8,
        "x_label": r"$p$",
    },
    {
        "output": "Analytic_Stability_pi0.png",
        "reference": "Stability_pi0.png",
        "x_kind": "p",
        "x_values": np.linspace(0.0, 0.5, 20),
        "theta": 0.0,
        "x_label": r"$p$",
    },
    {
        "output": "Analytic_DetailSweep2Pi8.png",
        "reference": "DetailSweep2Pi8.png",
        "x_kind": "p",
        "x_values": np.linspace(0.22, 0.56, 20),
        "theta": np.pi / 8,
        "alpha_values": np.linspace(0.0, 0.05, 20),
        "x_label": r"$p$",
    },
    {
        "output": "Analytic_DetailSweep2Pi0.png",
        "reference": "DetailSweep2Pi0.png",
        "x_kind": "p",
        "x_values": np.linspace(0.22, 0.56, 20),
        "theta": 0.0,
        "alpha_values": np.linspace(0.0, 0.05, 20),
        "x_label": r"$p$",
    },
    {
        "output": "Analytic_ThetaSweep.png",
        "reference": "ThetaSweep.png",
        "x_kind": "theta",
        "x_values": np.linspace(0.0, np.pi / 2, 20),
        "p": 0.5,
        "alpha_values": np.linspace(0.0, 0.07, 20),
        "x_label": r"$\theta$",
    },
)


def quadrature(h_se_j, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    normal_nodes = np.sqrt(2.0) * nodes
    normal_weights = weights / np.sqrt(np.pi)
    zg, zh = np.meshgrid(normal_nodes, normal_nodes, indexing="ij")
    wg, wh = np.meshgrid(normal_weights, normal_weights, indexing="ij")
    return (h_se_j * zg).ravel(), zh.ravel(), (wg * wh).ravel()


def bf_grid(sweep, *, t, fragment_size, g, z_h, weights):
    x_values = sweep["x_values"]
    alpha_values = sweep.get("alpha_values", np.linspace(0.0, 3.0, 20))
    bf = np.empty((len(alpha_values), len(x_values)), dtype=float)

    for ai, alpha in enumerate(alpha_values):
        h = alpha * z_h
        for xi, x in enumerate(x_values):
            if sweep["x_kind"] == "p":
                p, theta = x, sweep["theta"]
            else:
                p, theta = sweep["p"], x
            mean_overlap = np.sum(weights * branch_overlap(t, p, theta, g, h))
            bf[ai, xi] = np.clip(mean_overlap, 0.0, 1.0) ** fragment_size
    return alpha_values, bf


def render(summary_dir, sweep, alpha_values, bf, *, t, fragment_size, dpi):
    figure_dir = summary_dir / "Figs_1"
    reference_path = figure_dir / sweep["reference"]
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    x_values = sweep["x_values"]
    extent = [x_values[0], x_values[-1], alpha_values[0], alpha_values[-1]]
    with Image.open(reference_path) as image:
        reference = np.asarray(image).copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.65, 5.67),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )
    axes[0].imshow(reference)
    axes[0].axis("off")
    axes[0].set_title("Numerical QD slope", fontsize=16)

    image = axes[1].imshow(
        bf,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title(rf"$B_F$, $T={t:g}$, $|F|={fragment_size}$", fontsize=16)
    axes[1].set_xlabel(sweep["x_label"], fontsize=14)
    axes[1].set_ylabel(r"$\alpha_2$", fontsize=14)
    fig.colorbar(image, ax=axes[1], label=r"$B_F$")

    output_path = figure_dir / sweep["output"]
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the six QD-summary numerical/B_F comparison panels."
    )
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--t", type=float, default=60.0)
    parser.add_argument("--fragment-size", type=int, default=8)
    parser.add_argument("--h-se-j", type=float, default=0.1)
    parser.add_argument("--hermite-order", type=int, default=80)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    g, z_h, weights = quadrature(args.h_se_j, args.hermite_order)
    for sweep in SWEEPS:
        alpha_values, bf = bf_grid(
            sweep,
            t=args.t,
            fragment_size=args.fragment_size,
            g=g,
            z_h=z_h,
            weights=weights,
        )
        print(
            render(
                args.summary_dir,
                sweep,
                alpha_values,
                bf,
                t=args.t,
                fragment_size=args.fragment_size,
                dpi=args.dpi,
            )
        )


if __name__ == "__main__":
    main()
