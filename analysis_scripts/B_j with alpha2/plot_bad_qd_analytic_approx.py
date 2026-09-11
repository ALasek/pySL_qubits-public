import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.special import erf


DEFAULT_FIG_DIR = Path(r"D:\Github\QD-summary-Imperfect-CNOT\Figs_sweeps\Mironowicz_rand_bias035_theta_alpha2_sweep")


def clean_float(value):
    return f"{value:g}".replace(".", "p")


def gaussian_one_minus_cos_over_h2(b, alpha):
    alpha = np.asarray(alpha, dtype=float)
    out = np.empty_like(alpha)
    small = np.abs(alpha) < 1e-12
    out[small] = 0.5 * b * b

    a = alpha[~small]
    out[~small] = (
        b * np.sqrt(np.pi / 2.0) / a * erf(a * b / np.sqrt(2.0))
        - (1.0 - np.exp(-0.5 * a * a * b * b)) / (a * a)
    )
    return out


def sinc_factor(alpha, t):
    return gaussian_one_minus_cos_over_h2(2.0 * t, alpha) / (2.0 * t * t)


def cos_integral_factor(alpha, t):
    alpha = np.asarray(alpha, dtype=float)
    out = np.empty_like(alpha)
    small = np.abs(alpha) < 1e-12
    out[small] = 1.0

    a = alpha[~small]
    out[~small] = np.sqrt(np.pi / 8.0) / (a * t) * erf(np.sqrt(2.0) * a * t)
    return out


def dressed_alignment(theta, alpha, *, p, t):
    rx = 2.0 * np.sqrt(p * (1.0 - p))
    rz = 2.0 * p - 1.0
    c = np.cos(theta)
    s = np.sin(theta)
    sh = sinc_factor(alpha, t)
    ch = cos_integral_factor(alpha, t)
    lam = (
        c * c * sh * (1.0 - rx * rx)
        + s * s * (1.0 - rz * rz)
        - 2.0 * rx * rz * c * s * ch
    )
    return np.clip(lam, 0.0, 1.0)


def proxies(theta_grid, alpha_grid, *, p, t, h_se_j, fragment_size):
    lam = dressed_alignment(theta_grid, alpha_grid, p=p, t=t)
    kappa = np.pi * h_se_j / 2.0

    chi = (kappa * t) ** 2 * lam
    bf_small_rotation = np.exp(-0.5 * fragment_size * chi)

    phase_factor = 0.5 * (1.0 - np.exp(-2.0 * (kappa * t) ** 2))
    bf_bounded = np.clip(1.0 - phase_factor * lam, 0.0, 1.0) ** (0.5 * fragment_size)
    return lam, bf_small_rotation, bf_bounded


def load_qd_reference(fig_dir):
    candidates = [
        fig_dir / "qd_slope.jpg",
        fig_dir / "qd_slope_2D_x-Mironowicz_theta_y-Mironowicz_alpha2_N-16_T--1.jpg",
    ]
    for path in candidates:
        if path.exists():
            return np.asarray(Image.open(path))
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot analytic bad-QD approximations for independent Mironowicz random fields."
    )
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--theta-points", type=int, default=240)
    parser.add_argument("--alpha-points", type=int, default=240)
    parser.add_argument("--theta-max", type=float, default=np.pi / 4)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--t", type=float, default=30.0)
    parser.add_argument("--p", type=float, default=0.35)
    parser.add_argument("--h-se-j", type=float, default=0.1)
    parser.add_argument("--fragment-size", type=int, default=8)
    parser.add_argument("--bounded-vmin", type=float, default=0.0)
    parser.add_argument("--bounded-vmax", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    theta_values = np.linspace(0.0, args.theta_max, args.theta_points)
    alpha_values = np.linspace(0.0, args.alpha_max, args.alpha_points)
    theta_grid, alpha_grid = np.meshgrid(theta_values, alpha_values)
    lam, bf_small_rotation, bf_bounded = proxies(
        theta_grid,
        alpha_grid,
        p=args.p,
        t=args.t,
        h_se_j=args.h_se_j,
        fragment_size=args.fragment_size,
    )

    extent = [theta_values[0], theta_values[-1], alpha_values[0], alpha_values[-1]]
    qd_reference = load_qd_reference(args.fig_dir)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), constrained_layout=True)
    if qd_reference is not None:
        axes[0].imshow(qd_reference)
        axes[0].axis("off")
    axes[0].set_title("Corrected simulation: QD slope")

    im1 = axes[1].imshow(lam, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[1].set_title(r"Dressed alignment $\Lambda_{\rm eff}$")
    axes[1].set_xlabel(r"$\theta$")
    axes[1].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        bf_bounded,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=args.bounded_vmin,
        vmax=args.bounded_vmax,
    )
    axes[2].set_title(r"Bounded proxy $B_F^{\rm approx}$")
    axes[2].set_xlabel(r"$\theta$")
    axes[2].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(bf_small_rotation, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[3].set_title(r"Small-rotation $\exp[-|F|\chi/2]$")
    axes[3].set_xlabel(r"$\theta$")
    axes[3].set_ylabel(r"$\alpha_2$")
    fig.colorbar(im3, ax=axes[3])

    suffix = f"_T{clean_float(args.t)}_F{args.fragment_size}"
    out_path = args.fig_dir / f"bad_qd_approx{suffix}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
