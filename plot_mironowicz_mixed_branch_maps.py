"""
CLI README

Standalone closed-form branch-distinguishability maps for the two-environment
qubit central-interaction model in Mironowicz et al.

This script does not run pySL and does not diagonalize density matrices. It
uses the SU(2) relative-branch rotation for Eq. (6) with alpha1=alpha3=0:

  H0 = alpha2 Z
  H1 = -(pi/2) cos(theta) X + (alpha2 - (pi/2) sin(theta)) Z

after dropping the branch-1 scalar phase. For the mixed environment state
rho = p|0><0| + (1-p)|1><1|, p in [0, 0.5], it plots:

  A = |q x r|^2
  B = sqrt(1 - A)
  |gamma| = sqrt(q0^2 + (q.r)^2)
  B^2 - |gamma|^2
  SBS bound = 2 |gamma|^R + B^M

where M is the observed fragment size and R the unobserved fragment size.

Common commands:
  python plot_mironowicz_mixed_branch_maps.py --theta 0

  python plot_mironowicz_mixed_branch_maps.py --theta-set fig2 \
      --alpha-points 301 --p-points 201

  python plot_mironowicz_mixed_branch_maps.py --theta-set fig2 \
      --observed-size 7 --unobserved-size 1

  python plot_mironowicz_mixed_branch_maps.py --replot-from-dir \
      data/mironowicz_mixed_branch_maps/fig2_M1_R1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def _as_float_expr(text: str) -> float:
    allowed = {"pi": math.pi, "tau": math.tau, "e": math.e}
    return float(eval(text, {"__builtins__": {}}, allowed))


def theta_label(theta: float) -> str:
    known = {
        0.0: "0",
        math.pi / 8.0: "pi8",
        math.pi / 4.0: "pi4",
        0.9 * math.pi / 2.0: "0p9pi2",
    }
    for value, label in known.items():
        if abs(theta - value) < 1e-12:
            return label
    return f"{theta:.6g}".replace("-", "m").replace(".", "p")


def relative_rotation_components(theta: float, alpha2: np.ndarray, time_value: float) -> tuple[np.ndarray, ...]:
    ax = -0.5 * math.pi * math.cos(theta)
    az = alpha2 - 0.5 * math.pi * math.sin(theta)
    omega = np.sqrt(ax * ax + az * az)

    mx = np.divide(ax, omega, out=np.zeros_like(omega, dtype=np.float64), where=omega > 0)
    mz = np.divide(az, omega, out=np.zeros_like(omega, dtype=np.float64), where=omega > 0)

    ch = np.cos(alpha2 * time_value)
    sh = np.sin(alpha2 * time_value)
    co = np.cos(omega * time_value)
    so = np.sin(omega * time_value)

    q0 = ch * co + sh * so * mz
    qx = ch * so * mx
    qy = -sh * so * mx
    qz = ch * so * mz - sh * co
    return q0, qx, qy, qz


def compute_branch_maps(
    theta: float,
    alphas: np.ndarray,
    ps: np.ndarray,
    *,
    time_value: float,
    observed_size: int,
    unobserved_size: int,
) -> dict[str, np.ndarray]:
    alpha_grid = alphas[None, :]
    p_grid = ps[:, None]
    rz = 2.0 * p_grid - 1.0

    q0, qx, qy, qz = relative_rotation_components(theta, alpha_grid, time_value)
    q_norm_sq = qx * qx + qy * qy + qz * qz
    dot = qz * rz

    record_strength = rz * rz * (qx * qx + qy * qy)
    branch_B_sq = np.clip(1.0 - record_strength, 0.0, 1.0)
    branch_B = np.sqrt(branch_B_sq)
    gamma_sq = np.clip(q0 * q0 + dot * dot, 0.0, 1.0)
    gamma = np.sqrt(gamma_sq)
    trace_distance = np.sqrt(np.clip(record_strength, 0.0, 1.0))
    gap = branch_B_sq - gamma_sq
    gap_identity = q_norm_sq * (1.0 - rz * rz)

    observed_branch_B = branch_B**observed_size
    unobserved_gamma = gamma**unobserved_size
    sbs_bound = 2.0 * unobserved_gamma + observed_branch_B

    return {
        "record_strength_A": record_strength,
        "branch_B": branch_B,
        "trace_distance_D": trace_distance,
        "gamma_abs": gamma,
        "branch_B_sq_minus_gamma_sq": gap,
        "gap_identity_check": gap_identity,
        "observed_branch_B": observed_branch_B,
        "unobserved_gamma": unobserved_gamma,
        "sbs_bound": sbs_bound,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _load_npz_maps(path: Path) -> tuple[float, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    legacy_keys = {
        "fidelity_F": "branch_B",
        "fidelity_sq_minus_gamma_sq": "branch_B_sq_minus_gamma_sq",
        "observed_fidelity": "observed_branch_B",
    }
    with np.load(path) as data:
        theta = float(np.asarray(data["theta"]).item())
        alphas = np.asarray(data["alpha2_values"])
        ps = np.asarray(data["p_values"])
        maps = {}
        for key in data.files:
            if key in {"theta", "alpha2_values", "p_values"}:
                continue
            maps[legacy_keys.get(key, key)] = np.asarray(data[key])
    return theta, alphas, ps, maps


def _theta_results_from_dir(source_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    paths = sorted(source_dir.glob("theta_*_branch_maps.npz"))
    if not paths:
        raise FileNotFoundError(f"No theta_*_branch_maps.npz files found in {source_dir}")

    theta_results = []
    alphas_ref = None
    ps_ref = None
    for path in paths:
        theta, alphas, ps, maps = _load_npz_maps(path)
        if alphas_ref is None:
            alphas_ref = alphas
            ps_ref = ps
        elif not (np.array_equal(alphas_ref, alphas) and np.array_equal(ps_ref, ps)):
            raise ValueError(f"Grid mismatch in {path}")
        label = path.name.removeprefix("theta_").removesuffix("_branch_maps.npz")
        theta_results.append({"theta": theta, "label": label, "maps": maps})

    theta_results.sort(key=lambda result: result["theta"])
    return theta_results, alphas_ref, ps_ref


def plot_single_theta(out_dir: Path, label: str, theta: float, alphas: np.ndarray, ps: np.ndarray, maps: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    specs = [
        ("record_strength_A", "record strength A", "A", "viridis"),
        ("branch_B", "branch fidelity B", "B", "magma_r"),
        ("gamma_abs", "decoherence factor |gamma|", "|gamma|", "magma_r"),
        ("branch_B_sq_minus_gamma_sq", "mixedness gap B^2 - |gamma|^2", "gap", "viridis"),
        ("observed_branch_B", "observed-fragment branch fidelity B^M", "B^M", "magma_r"),
        ("sbs_bound", "SBS bound 2|gamma|^R + B^M", "bound", "viridis"),
    ]
    extent = [float(alphas[0]), float(alphas[-1]), float(ps[0]), float(ps[-1])]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), constrained_layout=True)
    for ax, (key, title, cbar_label, cmap) in zip(axes.flat, specs):
        image = ax.imshow(
            maps[key],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
        )
        alpha_crit = 0.5 * math.pi * math.sin(theta)
        if alphas[0] <= alpha_crit <= alphas[-1]:
            ax.axvline(alpha_crit, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("alpha_2")
        ax.set_ylabel("mixedness p")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(cbar_label)
    fig.suptitle(f"Mironowicz mixed branch maps, theta={theta:.6g}, t=1")
    fig.savefig(out_dir / f"theta_{label}_branch_maps.png", dpi=180)
    plt.close(fig)

    for key, title, cbar_label, cmap in specs:
        fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
        image = ax.imshow(
            maps[key],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
        )
        alpha_crit = 0.5 * math.pi * math.sin(theta)
        if alphas[0] <= alpha_crit <= alphas[-1]:
            ax.axvline(alpha_crit, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(f"{title}, theta={theta:.6g}")
        ax.set_xlabel("alpha_2")
        ax.set_ylabel("mixedness p")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(cbar_label)
        fig.savefig(out_dir / f"theta_{label}_{key}.png", dpi=180)
        plt.close(fig)


def plot_theta_set_summary(out_dir: Path, theta_results: list[dict], alphas: np.ndarray, ps: np.ndarray, metric: str) -> None:
    import matplotlib.pyplot as plt

    if not theta_results:
        return

    metric_labels = {
        "sbs_bound": ("SBS bound 2|gamma|^R + B^M", "bound", "viridis"),
        "branch_B": ("branch fidelity B", "B", "magma_r"),
        "gamma_abs": ("decoherence factor |gamma|", "|gamma|", "magma_r"),
        "branch_B_sq_minus_gamma_sq": ("mixedness gap B^2 - |gamma|^2", "gap", "viridis"),
    }
    title, cbar_label, cmap = metric_labels.get(metric, (metric, metric, "viridis"))
    ncols = 2
    nrows = math.ceil(len(theta_results) / ncols)
    extent = [float(alphas[0]), float(alphas[-1]), float(ps[0]), float(ps[-1])]
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 4.4 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    values = [result["maps"][metric] for result in theta_results]
    vmin = min(float(np.nanmin(value)) for value in values)
    vmax = max(float(np.nanmax(value)) for value in values)
    for ax, result in zip(axes, theta_results):
        theta = result["theta"]
        image = ax.imshow(
            result["maps"][metric],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        alpha_crit = 0.5 * math.pi * math.sin(theta)
        if alphas[0] <= alpha_crit <= alphas[-1]:
            ax.axvline(alpha_crit, color="white", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(f"theta={theta:.6g}")
        ax.set_xlabel("alpha_2")
        ax.set_ylabel("mixedness p")
    for ax in axes[len(theta_results) :]:
        ax.axis("off")
    cbar = fig.colorbar(image, ax=axes[: len(theta_results)].tolist())
    cbar.set_label(cbar_label)
    fig.suptitle(f"{title} across theta values")
    fig.savefig(out_dir / f"theta_set_{metric}.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot closed-form mixed-state Mironowicz branch maps.")
    parser.add_argument("--out-dir", help="Output directory. Default: data/mironowicz_mixed_branch_maps/<timestamp>.")
    parser.add_argument("--replot-from-dir", help="Regenerate plots from existing theta_*_branch_maps.npz files without recomputing maps.")
    parser.add_argument("--theta", action="append", type=_as_float_expr, help="Theta value. Repeatable; accepts pi/8.")
    parser.add_argument(
        "--theta-set",
        choices=["fig1", "fig2"],
        default="fig1",
        help="fig1: theta=0. fig2: 0, pi/8, pi/4, 0.9*pi/2.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-points", type=int, default=301)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=0.5)
    parser.add_argument("--p-points", type=int, default=201)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--observed-size", type=int, default=1, help="Observed fragment size M. Use 1 for Figs. 1-2, 7 for Fig. 6-style bound.")
    parser.add_argument("--unobserved-size", type=int, default=1, help="Unobserved fragment size R.")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replot_from_dir:
        source_dir = Path(args.replot_from_dir)
        out_dir = Path(args.out_dir) if args.out_dir else source_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        theta_results, alphas, ps = _theta_results_from_dir(source_dir)
        if not args.no_plots:
            for result in theta_results:
                plot_single_theta(out_dir, result["label"], result["theta"], alphas, ps, result["maps"])
            if len(theta_results) > 1:
                for metric in ["sbs_bound", "branch_B", "gamma_abs", "branch_B_sq_minus_gamma_sq"]:
                    plot_theta_set_summary(out_dir, theta_results, alphas, ps, metric)
        manifest = {
            "script": Path(__file__).name,
            "replot_from_dir": str(source_dir),
            "outputs": [f"theta_{result['label']}_branch_maps.npz" for result in theta_results],
            "notation": "branch fidelity is B; legacy fidelity_F arrays are plotted as branch_B",
        }
        write_json(out_dir / "replot_manifest.json", manifest)
        print("Mironowicz mixed branch maps replotted from existing data")
        print(f"source_dir = {source_dir.resolve()}")
        print(f"out_dir = {out_dir.resolve()}")
        return

    if args.theta:
        theta_values = args.theta
    elif args.theta_set == "fig2":
        theta_values = [0.0, math.pi / 8.0, math.pi / 4.0, 0.9 * math.pi / 2.0]
    else:
        theta_values = [0.0]

    if args.observed_size < 1 or args.unobserved_size < 0:
        raise ValueError("--observed-size must be >=1 and --unobserved-size must be >=0")

    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "mironowicz_mixed_branch_maps" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    ps = np.linspace(args.p_min, args.p_max, args.p_points)

    theta_results = []
    for theta in theta_values:
        label = theta_label(theta)
        maps = compute_branch_maps(
            float(theta),
            alphas,
            ps,
            time_value=args.time,
            observed_size=args.observed_size,
            unobserved_size=args.unobserved_size,
        )
        np.savez_compressed(
            out_dir / f"theta_{label}_branch_maps.npz",
            theta=np.array(theta),
            alpha2_values=alphas,
            p_values=ps,
            **maps,
        )
        if not args.no_plots:
            plot_single_theta(out_dir, label, float(theta), alphas, ps, maps)
        theta_results.append({"theta": float(theta), "label": label, "maps": maps})

    if not args.no_plots and len(theta_results) > 1:
        for metric in ["sbs_bound", "branch_B", "gamma_abs", "branch_B_sq_minus_gamma_sq"]:
            plot_theta_set_summary(out_dir, theta_results, alphas, ps, metric)

    manifest = {
        "script": Path(__file__).name,
        "model": "Mironowicz Eq. (6), alpha1=alpha3=0, closed-form per-qubit branch maps",
        "args": vars(args),
        "theta_values": theta_values,
        "outputs": [f"theta_{result['label']}_branch_maps.npz" for result in theta_results],
    }
    write_json(out_dir / "manifest.json", manifest)

    print("Mironowicz mixed branch maps complete")
    print(f"out_dir = {out_dir.resolve()}")


if __name__ == "__main__":
    main()
