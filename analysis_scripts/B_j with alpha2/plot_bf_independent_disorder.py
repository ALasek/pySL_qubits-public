import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_OUT_DIR = Path(r"D:\Github\QD-summary-Imperfect-CNOT\Figs_sweeps\Mironowicz_rand_bias035_theta_alpha2_sweep")


def clean_float(value):
    return f"{value:g}".replace(".", "p")


def plot_suffix(args):
    suffix = f"_T{clean_float(args.t)}_F{args.fragment_size}"
    if args.vmin != 0.0 or args.vmax != 1.0:
        suffix += f"_vmin{clean_float(args.vmin)}_vmax{clean_float(args.vmax)}"
    return suffix


def short_plot_suffix(args):
    return f"_T{clean_float(args.t)}_F{args.fragment_size}"


def branch_overlap(t, p, theta, g, h):
    c0 = np.sqrt(p)
    c1 = np.sqrt(1.0 - p)
    ax = -np.pi * g / 2.0 * np.cos(theta)
    az = h - np.pi * g / 2.0 * np.sin(theta)
    omega = np.sqrt(ax * ax + az * az)
    cos_ot = np.cos(omega * t)
    sin_over_omega = np.empty_like(omega)
    np.divide(np.sin(omega * t), omega, out=sin_over_omega, where=omega > 1e-14)
    sin_over_omega[omega <= 1e-14] = t

    v0 = cos_ot * c0 - 1j * sin_over_omega * (az * c0 + ax * c1)
    v1 = cos_ot * c1 - 1j * sin_over_omega * (ax * c0 - az * c1)
    amp = c0 * np.exp(1j * h * t) * v0 + c1 * np.exp(-1j * h * t) * v1
    return np.abs(amp)


def independent_bf_grid(
    theta_values,
    alpha_values,
    *,
    t,
    p,
    h_se_j,
    fragment_size,
    hermite_order,
):
    nodes, weights = np.polynomial.hermite.hermgauss(hermite_order)
    z = np.sqrt(2.0) * nodes
    w = weights / np.sqrt(np.pi)
    zg, zh = np.meshgrid(z, z, indexing="ij")
    wg, wh = np.meshgrid(w, w, indexing="ij")
    quad_weight = (wg * wh).ravel()
    g = (h_se_j * zg).ravel()
    z_h = zh.ravel()

    bf = np.empty((len(alpha_values), len(theta_values)), dtype=float)
    for ai, alpha in enumerate(alpha_values):
        h = alpha * z_h
        for ti, theta in enumerate(theta_values):
            b_single_mean = np.sum(quad_weight * branch_overlap(t, p, theta, g, h))
            bf[ai, ti] = np.clip(b_single_mean, 0.0, 1.0) ** fragment_size
    return bf


def load_qd_reference(out_dir):
    candidates = [
        out_dir / "qd_slope.jpg",
        out_dir / "qd_slope_2D_x-Mironowicz_theta_y-Mironowicz_alpha2_N-16_T--1.jpg",
    ]
    for path in candidates:
        if path.exists():
            return np.asarray(Image.open(path))
    return None


def plot_distinguishability(out_dir, theta_values, alpha_values, bf, qd_reference, args):
    df = np.sqrt(np.clip(1.0 - bf * bf, 0.0, 1.0))
    extent = [theta_values[0], theta_values[-1], alpha_values[0], alpha_values[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    if qd_reference is not None:
        axes[0].imshow(qd_reference)
        axes[0].axis("off")
    axes[0].set_title("Reference: QD slope")

    im1 = axes[1].imshow(df, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1].set_title(rf"Independent-disorder $D_F$, $T={args.t:g}$, $p={args.p:g}$, $|F|={args.fragment_size}$")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im1, ax=axes[1], label=r"$D_F=\sqrt{1-B_F^2}$")

    im2 = axes[2].imshow(bf, origin="lower", aspect="auto", extent=extent, vmin=args.vmin, vmax=args.vmax)
    axes[2].set_title("Poor-record proxy, independent field draws")
    axes[2].set_xlabel(r"$\theta$")
    axes[2].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im2, ax=axes[2], label=rf"$B_F$, $T={args.t:g}$")

    path = out_dir / f"distinguishability{plot_suffix(args)}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_bf_overlay(out_dir, theta_values, alpha_values, bf, qd_reference, args):
    extent = [theta_values[0], theta_values[-1], alpha_values[0], alpha_values[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.7), constrained_layout=True)
    if qd_reference is not None:
        axes[0].imshow(qd_reference)
        axes[0].axis("off")
    axes[0].set_title("Numerical QD slope")

    im = axes[1].imshow(bf, origin="lower", aspect="auto", extent=extent, vmin=args.vmin, vmax=args.vmax)
    axes[1].set_title(rf"Independent-disorder $B_F$, $T={args.t:g}$, $p={args.p:g}$, $|F|={args.fragment_size}$")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im, ax=axes[1], label=r"$B_F$")

    path = out_dir / f"bf{short_plot_suffix(args)}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot B_F for the Mironowicz theta/alpha2 sweep with independent g_j and h_j disorder."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--theta-points", type=int, default=160)
    parser.add_argument("--alpha-points", type=int, default=160)
    parser.add_argument("--theta-max", type=float, default=np.pi / 2)
    parser.add_argument("--alpha-max", type=float, default=2.5)
    parser.add_argument("--t", type=float, default=60.0)
    parser.add_argument("--p", type=float, default=0.35)
    parser.add_argument("--h-se-j", type=float, default=0.1)
    parser.add_argument("--fragment-size", type=int, default=8)
    parser.add_argument("--hermite-order", type=int, default=80)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    theta_values = np.linspace(0.0, args.theta_max, args.theta_points)
    alpha_values = np.linspace(0.0, args.alpha_max, args.alpha_points)
    bf = independent_bf_grid(
        theta_values,
        alpha_values,
        t=args.t,
        p=args.p,
        h_se_j=args.h_se_j,
        fragment_size=args.fragment_size,
        hermite_order=args.hermite_order,
    )

    qd_reference = load_qd_reference(args.out_dir)
    paths = [
        plot_distinguishability(args.out_dir, theta_values, alpha_values, bf, qd_reference, args),
        plot_bf_overlay(args.out_dir, theta_values, alpha_values, bf, qd_reference, args),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
