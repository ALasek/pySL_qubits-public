#!/usr/bin/env python3
"""Regenerate Supplemental Figs. S2 and S4 from their stored summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from paper_plot_style import apply_style, save_figure

apply_style()
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
S2_SOURCE = ROOT / "figures/paper_a/source_manifests/higher_n_validation_metrics.csv"
S4_SOURCE = ROOT / "figures/paper_a/source_manifests/stability_endpoints_compact.csv"
S2_OUTPUT = ROOT / "figures/paper_a/supplement/higher_n_validation.pdf"
S4_OUTPUT = ROOT / "figures/paper_a/supplement/stability_endpoints_compact.pdf"
DEFAULT_ANALYSIS_DIR = ROOT / "analysis/supplement_polish_2026_09_09/s2_s4"

COLORS = {16: "#0072B2", 20: "#D55E00"}
S2_CASES = (
    ("aligned_uniform_rescue", "uniform\n$0.25g$"),
    ("aligned_random_rescue", "Gaussian\n$0.5g$"),
    ("aligned_uniform_strong", "uniform\n$30g$"),
    ("aligned_random_strong", "Gaussian\n$30g$"),
)
S4_ENDPOINTS = (0.0, 0.5)
S4_COLUMNS = (
    ("holevo_fhalf", r"mean $\chi_z/H_Z$", "viridis", 0.0, 1.0, 0.9),
    ("discord_fhalf", r"mean $D_z/H_Z$", "magma_r", 0.0, 0.30, 0.1),
    ("holevo_fhalf_min", r"mean $\chi_{\min}$", "cividis", 0.0, 1.0, 0.9),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def row_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str, int], dict[str, str]]:
    lookup: dict[tuple[str, int, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["case"], int(row["n_environment"]), row["metric"], int(row["fragment_size"]))
        if key in lookup:
            raise ValueError(f"duplicate S2 summary row: {key}")
        lookup[key] = row
    return lookup


def estimate_interval(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(float(row[key]) for key in ("estimate", "ci_low", "ci_high"))


def plot_s2(rows: list[dict[str, str]], output: Path) -> dict:
    lookup = row_lookup(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), layout="constrained")

    for axis, metric, title, ylabel in (
        (axes[0, 0], "q10_holevo", "Equal fragment size", r"$q_{0.1}(\chi_z)/H_Z$"),
        (axes[0, 1], "q90_discord", "Equal fragment size", r"$q_{0.9}(D_z)/H_Z$"),
    ):
        for environment in (16, 20):
            members = [
                lookup[("no_field_lambda_half", environment, metric, fragment)]
                for fragment in range(1, environment // 2 + 1)
            ]
            x = np.asarray([int(member["fragment_size"]) for member in members])
            y, low, high = (np.asarray(values) for values in zip(*(estimate_interval(member) for member in members)))
            axis.plot(x, y, color=COLORS[environment], marker="o", markersize=3.2, linewidth=1.25)
            axis.fill_between(x, low, high, color=COLORS[environment], alpha=0.16, linewidth=0)
        axis.set(title=title, xlabel=r"fragment size $m$", ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].axhline(0.9, color="0.4", linestyle=":", linewidth=0.9)
    axes[0, 0].set_ylim(-0.02, 1.03)
    axes[0, 1].set_ylim(-0.005, 0.25)

    case_keys = [case for case, _ in S2_CASES]
    x = np.arange(len(case_keys), dtype=float)
    offsets = {16: -0.09, 20: 0.09}
    for axis, metric, title, ylabel in (
        (axes[1, 0], "q10_holevo_half", "Half environment", r"time median $q_{0.1}(\chi_z)/H_Z$"),
        (axes[1, 1], "holevo_fhalf_min", "Half environment", r"mean late-window min $q_{0.1}(\chi_z)/H_Z$"),
    ):
        for environment in (16, 20):
            members = [
                lookup[(case, environment, metric, environment // 2)] for case in case_keys
            ]
            y, low, high = (np.asarray(values) for values in zip(*(estimate_interval(member) for member in members)))
            axis.errorbar(
                x + offsets[environment],
                y,
                yerr=np.maximum(np.vstack((y - low, high - y)), 0.0),
                color=COLORS[environment],
                marker="o",
                markersize=4.0,
                linewidth=1.25,
                linestyle="none",
                capsize=2.0,
            )
        axis.axhline(0.9, color="0.4", linestyle=":", linewidth=0.9)
        axis.set(title=title, xticks=x, xticklabels=[label for _, label in S2_CASES], ylabel=ylabel)
        axis.set_ylim(-0.04, 1.05)
        axis.grid(axis="y", alpha=0.2)

    axes[0, 0].legend(
        handles=[
            Line2D([], [], color=COLORS[environment], marker="o", label=rf"$N_{{\mathcal{{E}}}}={environment}$")
            for environment in (16, 20)
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
    )
    for label, axis, y in zip("abcd", axes.flat, (0.94, 0.94, 0.84, 0.84)):
        axis.text(0.02, y, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output)
    plt.close(fig)
    return {
        "source_rows": len(rows),
        "finite_size_pairs": {
            "N_E_16": sum(row["n_environment"] == "16" for row in rows),
            "N_E_20": sum(row["n_environment"] == "20" for row in rows),
        },
        "estimators": {
            "a": "late-window median q0.1 Holevo at each absolute fragment size",
            "b": "late-window median q0.9 discord at each absolute fragment size",
            "c": "mean over realizations of the time median q0.1 Holevo at f=1/2",
            "d": "mean over realizations of the late-window minimum q0.1 Holevo at f=1/2",
        },
    }


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])]))


def s4_grid(rows: list[dict[str, str]], theta: float, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if math.isclose(float(row["theta_over_pi"]), theta, abs_tol=1e-12)]
    p_values = np.asarray(sorted({float(row["p"]) for row in selected}))
    widths = np.asarray(sorted({float(row["field_width_over_g"]) for row in selected}))
    if p_values.size != 11 or widths.size != 11 or len(selected) != 121:
        raise ValueError(f"S4 theta={theta}: expected a complete 11 by 11 grid, found {len(selected)} rows")
    values = np.full((widths.size, p_values.size), np.nan)
    row_index = {value: index for index, value in enumerate(widths)}
    column_index = {value: index for index, value in enumerate(p_values)}
    for row in selected:
        values[row_index[float(row["field_width_over_g"])], column_index[float(row["p"])]] = float(row[metric])
    if not np.isfinite(values).all():
        raise ValueError(f"S4 theta={theta}, metric={metric}: incomplete metric grid")
    return p_values, widths, values


def add_threshold_contour(axis, p_values: np.ndarray, row_centers: np.ndarray, values: np.ndarray, level: float) -> bool:
    if not (float(np.nanmin(values)) < level < float(np.nanmax(values))):
        return False
    contour = axis.contour(p_values, row_centers, values, levels=(level,), colors="black", linewidths=0.85)
    return bool(contour.allsegs[0])


def add_vector_colorbar(axis, cmap: str, lower: float, upper: float, threshold: float) -> None:
    levels = np.linspace(lower, upper, 257)
    centers = 0.5 * (levels[:-1] + levels[1:])
    axis.pcolormesh(
        (0.0, 1.0),
        levels,
        centers[:, None],
        shading="flat",
        cmap=cmap,
        norm=Normalize(lower, upper),
        edgecolors="face",
        linewidth=0.1,
    )
    axis.axhline(threshold, color="black", linewidth=0.8)
    axis.set(xticks=[], ylim=(lower, upper))
    axis.yaxis.tick_right()
    axis.set_yticks(np.linspace(lower, upper, 6))
    axis.tick_params(labelsize=7)


def plot_s4(rows: list[dict[str, str]], output: Path) -> dict:
    if len(rows) != 242:
        raise ValueError(f"expected 242 endpoint rows, found {len(rows)}")
    fig = plt.figure(figsize=(7.2, 5.2), layout="constrained")
    layout = fig.add_gridspec(2, 6, width_ratios=(1.0, 0.13, 1.0, 0.13, 1.0, 0.13))
    axes = np.asarray(
        [[fig.add_subplot(layout[row, 2 * column]) for column in range(3)] for row in range(2)]
    )
    colorbar_axes = [fig.add_subplot(layout[:, 2 * column + 1]) for column in range(3)]
    contours: dict[str, dict[str, bool]] = {}
    for row, theta in enumerate(S4_ENDPOINTS):
        for column, (metric, title, cmap, lower, upper, threshold) in enumerate(S4_COLUMNS):
            axis = axes[row, column]
            p_values, field_ratios, values = s4_grid(rows, theta, metric)
            row_centers = np.arange(field_ratios.size, dtype=float)
            axis.pcolormesh(
                centers_to_edges(p_values),
                np.arange(field_ratios.size + 1, dtype=float) - 0.5,
                values,
                shading="flat",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            contours[f"theta_{theta:g}_{metric}"] = {
                "level": threshold,
                "drawn": add_threshold_contour(axis, p_values, row_centers, values, threshold),
            }
            axis.set(xlim=(0.0, 0.5), ylim=(-0.5, field_ratios.size - 0.5))
            axis.set_xticks(np.arange(0.0, 0.51, 0.1))
            axis.set_yticks(row_centers, [f"{value:g}" for value in field_ratios])
            axis.tick_params(labelsize=7, labelleft=column == 0)
            axis.hlines(np.arange(field_ratios.size + 1) - 0.5, 0.0, 0.5, colors="white", alpha=0.18, linewidth=0.35)
            axis.text(0.035, 0.96, f"({chr(ord('a') + 3 * row + column)})", transform=axis.transAxes, va="top", fontweight="bold", fontsize=8)
            if row == 0:
                axis.set_title(title, fontsize=8.5)
            if row == 1:
                axis.set_xlabel(r"preparation bias $p$", fontsize=8)
            if column == 0:
                theta_label = r"$\theta=0$" if row == 0 else r"$\theta=\pi/2$"
                axis.set_ylabel(theta_label + "\n" + r"sampled $W/g$", fontsize=8)

    for axis, (_, _, cmap, lower, upper, threshold) in zip(colorbar_axes, S4_COLUMNS):
        add_vector_colorbar(axis, cmap, lower, upper, threshold)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output)
    plt.close(fig)
    return {
        "endpoint_rows": len(rows),
        "grid_shape": [11, 11],
        "field_axis": "categorical sampled W/g positions labelled at each row",
        "color_limits": {metric: [lower, upper] for metric, _, _, lower, upper, _ in S4_COLUMNS},
        "threshold_contours": contours,
        "contour_interpretation": "Each contour marks only its own displayed metric; their overlap is not a joint-passing criterion.",
    }


def write_provenance(path: Path, s2: dict, s4: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {str(S2_SOURCE): sha256(S2_SOURCE), str(S4_SOURCE): sha256(S4_SOURCE)},
        "outputs": {str(S2_OUTPUT): sha256(S2_OUTPUT), str(S4_OUTPUT): sha256(S4_OUTPUT)},
        "s2": s2,
        "s4": s4,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (path / "README.md").write_text(
        "# Supplemental S2 and S4 polish\n\n"
        "`polish_s2_s4.py` redraws the two supplemental PDFs from the stored CSV summaries. "
        "It does not simulate, recompute, or modify the source data.\n\n"
        "S2 retains the matched eight-seed N_E=16/N_E=20 estimators. S4 retains the 11 by 11 "
        "categorical W/g grid at each endpoint. Its contours are per-metric reference levels and do not "
        "define a joint-passing region. `manifest.json` records exact input/output hashes.\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    s2 = plot_s2(load_csv(S2_SOURCE), S2_OUTPUT)
    s4 = plot_s4(load_csv(S4_SOURCE), S4_OUTPUT)
    write_provenance(args.analysis_dir, s2, s4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
