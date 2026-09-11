"""Plot fragment-size Holevo and discord sweeps from the matched-Lambda batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BATCH_NAME = "Paper_A_final_SubmissionMatchedLambda"
EXPECTED_POINTS = 38
EXPECTED_LAMBDAS = 13
EXPECTED_ENVIRONMENT = 16
EXPECTED_REALIZATIONS = 24
LATE_WINDOW = (20.0, 40.0)


@dataclass(frozen=True)
class FragmentPoint:
    run_id: str
    lambda_value: float
    geometry: str
    fractions: np.ndarray
    holevo_q10: np.ndarray
    discord_q90: np.ndarray
    late_samples: int
    fragment_samples: int


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_batch_dir(path: Path) -> Path:
    path = path.resolve()
    candidate = path / BATCH_NAME
    if candidate.is_dir():
        path = candidate
    if path.name != BATCH_NAME or not (path / "runs").is_dir():
        raise ValueError(f"expected {BATCH_NAME} batch directory, got {path}")
    return path


def pointer_entropy(results, n_times: int) -> np.ndarray:
    if "rhoS_T" not in results:
        return np.full(n_times, math.log(2.0))
    rho = np.asarray(results["rhoS_T"])
    if rho.shape != (n_times, 2, 2):
        return np.full(n_times, math.log(2.0))
    probabilities = np.real(np.diagonal(rho, axis1=1, axis2=2))
    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        totals,
        out=np.zeros_like(probabilities, dtype=float),
        where=totals > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(probabilities > 0, -probabilities * np.log(probabilities), 0.0).sum(axis=1)


def quantile_array(results, key: str, target: float) -> np.ndarray:
    levels = np.asarray(results["fragment_quantile_levels"], dtype=float)
    index = int(np.argmin(np.abs(levels - target)))
    if not math.isclose(float(levels[index]), target, abs_tol=1e-12):
        raise ValueError(f"stored fragment quantiles do not contain q={target:g}")
    values = np.asarray(results[f"{key}_quantiles"], dtype=float)
    if values.ndim != 4 or values.shape[0] != levels.size:
        raise ValueError(f"unexpected {key}_quantiles shape {values.shape}")
    return values[index]


def validate_params(params: dict, run_dir: Path) -> None:
    expected = {
        "AverageOverRunsN": EXPECTED_REALIZATIONS,
        "H_EE_J": 0.0,
        "H_E_J": 0.0,
        "H_SE_J": 0.1,
        "H_SE_Special": "Mironowicz_rand",
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
        "compute_discord": True,
        "fragment_sample_count": 64,
        "nqubits_E": EXPECTED_ENVIRONMENT,
        "nqubits_S": 1,
        "store_fragment_samples": True,
    }
    for key, target in expected.items():
        value = params.get(key)
        matches = math.isclose(float(value), target, abs_tol=1e-12) if isinstance(target, float) else value == target
        if not matches:
            raise ValueError(f"{run_dir}: expected {key}={target!r}, got {value!r}")
    for key in ("lambda_geometry", "lambda_target"):
        if key not in params:
            raise ValueError(f"{run_dir}: missing {key}")


def load_point(run_dir: Path) -> FragmentPoint:
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    validate_params(params, run_dir)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        holevo = quantile_array(results, "Holevo_Z_S_Ef_fractionsT", 0.1).copy()
        discord = quantile_array(results, "Discord_Z_S_Ef_fractionsT", 0.9).copy()
        if holevo.shape != discord.shape or holevo.ndim != 3:
            raise ValueError(f"{run_dir}: incompatible Holevo and discord arrays")
        n_fragments, n_times, n_runs = holevo.shape
        if n_runs != EXPECTED_REALIZATIONS:
            raise ValueError(f"{run_dir}: expected {EXPECTED_REALIZATIONS} realizations, found {n_runs}")
        times = np.arange(n_times, dtype=float) * float(params["printT"])
        late_mask = (times >= LATE_WINDOW[0]) & (times <= LATE_WINDOW[1])
        if not np.any(late_mask):
            raise ValueError(f"{run_dir}: no samples in late window {LATE_WINDOW}")
        entropy = pointer_entropy(results, n_times)
        safe_entropy = np.where(entropy > 1e-12, entropy, np.nan)
        stored_sizes = [
            index
            for index in range(n_fragments)
            if np.any(np.isfinite(holevo[index])) and np.any(np.isfinite(discord[index]))
        ]
        expected_stored_sizes = list(range(1, EXPECTED_ENVIRONMENT // 2 + 1))
        if stored_sizes != expected_stored_sizes:
            raise ValueError(
                f"{run_dir}: expected stored fragment quantiles for sizes {expected_stored_sizes}, found {stored_sizes}"
            )
        finite_sizes = [0, *stored_sizes]
        holevo[0] = 0.0
        discord[0] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            normalized_holevo = holevo[finite_sizes] / safe_entropy[None, :, None]
            normalized_discord = discord[finite_sizes] / safe_entropy[None, :, None]
            holevo_late = np.nanmedian(normalized_holevo[:, late_mask, :], axis=1)
            discord_late = np.nanmedian(normalized_discord[:, late_mask, :], axis=1)
    return FragmentPoint(
        run_id=run_dir.name,
        lambda_value=float(params["lambda_target"]),
        geometry=str(params["lambda_geometry"]),
        fractions=np.asarray(finite_sizes, dtype=float) / EXPECTED_ENVIRONMENT,
        holevo_q10=holevo_late,
        discord_q90=discord_late,
        late_samples=int(np.sum(late_mask)),
        fragment_samples=int(params["fragment_sample_count"]),
    )


def load_points(batch_dir: Path) -> list[FragmentPoint]:
    points = [load_point(run_dir) for run_dir in sorted((batch_dir / "runs").iterdir()) if run_dir.is_dir()]
    if len(points) != EXPECTED_POINTS:
        raise ValueError(f"expected {EXPECTED_POINTS} completed runs, found {len(points)}")
    return points


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estimate = np.nanmean(values, axis=1)
    if draws == 0:
        empty = np.full_like(estimate, np.nan)
        return estimate, empty, empty
    indices = rng.integers(0, values.shape[1], size=(draws, values.shape[1]))
    sampled = np.nanmean(values[:, indices], axis=2)
    low, high = np.nanquantile(sampled, [0.025, 0.975], axis=1)
    return estimate, low, high


def aggregate(points: list[FragmentPoint], bootstrap: int, seed: int) -> tuple[list[dict], dict]:
    groups: dict[float, list[FragmentPoint]] = defaultdict(list)
    for point in points:
        groups[round(point.lambda_value, 12)].append(point)
    if len(groups) != EXPECTED_LAMBDAS:
        raise ValueError(f"expected {EXPECTED_LAMBDAS} Lambda targets, found {len(groups)}")
    rng = np.random.default_rng(seed)
    rows = []
    max_geometry_range = {"holevo_q10": 0.0, "discord_q90": 0.0}
    for lambda_value, members in sorted(groups.items()):
        fractions = members[0].fractions
        if any(not np.array_equal(member.fractions, fractions) for member in members[1:]):
            raise ValueError(f"Lambda={lambda_value:g}: inconsistent fragment grids")
        metric_outputs = {}
        for name in ("holevo_q10", "discord_q90"):
            geometry_values = np.stack([getattr(member, name) for member in members])
            paired_values = np.nanmean(geometry_values, axis=0)
            estimate, low, high = bootstrap_mean(paired_values, rng, bootstrap)
            geometry_means = np.nanmean(geometry_values, axis=2)
            geometry_range = np.ptp(geometry_means, axis=0) if len(members) > 1 else np.zeros(fractions.size)
            max_geometry_range[name] = max(max_geometry_range[name], float(np.nanmax(geometry_range)))
            metric_outputs[name] = (estimate, low, high, geometry_range)
        for index, fraction in enumerate(fractions):
            row = {
                "lambda": lambda_value,
                "fragment_size": int(round(fraction * EXPECTED_ENVIRONMENT)),
                "fragment_fraction": float(fraction),
                "geometry_count": len(members),
            }
            for name, values in metric_outputs.items():
                estimate, low, high, geometry_range = values
                row[name] = float(estimate[index])
                row[f"{name}_ci_low"] = float(low[index])
                row[f"{name}_ci_high"] = float(high[index])
                row[f"{name}_geometry_range"] = float(geometry_range[index])
            row["joint_record_threshold"] = bool(row["holevo_q10"] >= 0.9 and row["discord_q90"] <= 0.1)
            rows.append(row)
    metadata = {
        "lambda_count": len(groups),
        "fragment_fractions": members[0].fractions.tolist(),
        "maximum_fixed_lambda_geometry_range": max_geometry_range,
    }
    return rows, metadata


def grid(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambdas = np.asarray(sorted({row["lambda"] for row in rows}), dtype=float)
    fractions = np.asarray(sorted({row["fragment_fraction"] for row in rows}), dtype=float)
    lookup = {(row["lambda"], row["fragment_fraction"]): row[key] for row in rows}
    values = np.asarray([[lookup[(lambda_value, fraction)] for fraction in fractions] for lambda_value in lambdas])
    return fractions, lambdas, values


def centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        raise ValueError("at least two grid centers are required")
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def add_contour(axis, fractions: np.ndarray, lambdas: np.ndarray, values: np.ndarray, level: float, color: str) -> None:
    finite = values[np.isfinite(values)]
    if finite.size and float(np.min(finite)) <= level <= float(np.max(finite)):
        contour = axis.contour(fractions, lambdas, values, levels=[level], colors=[color], linewidths=1.3)
        axis.clabel(contour, fmt={level: f"{level:g}"}, fontsize=7, inline=True)


def plot(rows: list[dict], output_dir: Path, dpi: int) -> list[Path]:
    fractions, lambdas, holevo = grid(rows, "holevo_q10")
    _, _, discord = grid(rows, "discord_q90")
    x_edges = centers_to_edges(fractions)
    y_edges = centers_to_edges(lambdas)
    discord_max = min(1.0, max(0.25, math.ceil(float(np.nanmax(discord)) / 0.05) * 0.05))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    panels = (
        (axes[0], holevo, "viridis", 0.0, 1.0, r"10th-percentile $\chi_z/H_Z(S)$", 0.9, "white"),
        (axes[1], discord, "magma_r", 0.0, discord_max, r"90th-percentile $D_z/H_Z(S)$", 0.1, "black"),
    )
    for label, (axis, values, cmap, lower, upper, color_label, threshold, contour_color) in zip("ab", panels):
        image = axis.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="flat",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            rasterized=True,
        )
        add_contour(axis, fractions, lambdas, values, threshold, contour_color)
        axis.set_xlim(0.0, 0.5)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(r"fragment fraction $f=m/N_E$")
        axis.set_ylabel(r"record-writing invariant $\Lambda$")
        axis.set_xticks([0.0, 0.125, 0.25, 0.375, 0.5])
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.text(
            0.025,
            0.965,
            f"({label})",
            transform=axis.transAxes,
            va="top",
            color="white" if label == "a" else "black",
            fontweight="bold",
        )
        colorbar = fig.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(color_label)

    axes[0].set_title("Conservative pointer information", fontsize=9)
    axes[1].set_title("Conservative discord-like remainder", fontsize=9)
    output_pdf = output_dir / "matched_lambda_fragment_sweep.pdf"
    output_png = output_dir / "matched_lambda_fragment_sweep.png"
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [output_pdf, output_png]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot matched-Lambda fragment-size Holevo and discord maps.")
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    batch_dir = resolve_batch_dir(args.batch_dir)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(batch_dir)
    rows, aggregation = aggregate(points, args.bootstrap, args.seed)
    csv_path = output_dir / "matched_lambda_fragment_sweep.csv"
    write_csv(csv_path, rows)
    figures = plot(rows, output_dir, args.dpi)

    batch_manifest_path = batch_dir / "batch_manifest.json"
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    script_path = Path(__file__).resolve()
    outputs = [csv_path, *figures]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_batch": BATCH_NAME,
        "source_batch_dir": str(batch_dir),
        "source_batch_manifest_sha256": file_hash(batch_manifest_path),
        "simulation_source_provenance": batch_manifest.get("source_provenance"),
        "analysis_script": str(script_path),
        "analysis_script_sha256": file_hash(script_path),
        "point_count": len(points),
        "n_environment": EXPECTED_ENVIRONMENT,
        "realizations_per_point": EXPECTED_REALIZATIONS,
        "fragment_samples_per_size": points[0].fragment_samples,
        "late_window": list(LATE_WINDOW),
        "late_window_samples": points[0].late_samples,
        "aggregation": (
            "For each geometry and realization, take the late-window median of the stored fragment q10 Holevo "
            "or q90 pointer-basis discord after normalization by H_Z(S); average paired geometries at fixed "
            "Lambda, then average across realizations. Confidence intervals bootstrap the 24 paired realizations."
        ),
        "thresholds": {"holevo_q10": 0.9, "discord_q90": 0.1},
        **aggregation,
        "outputs": {path.name: file_hash(path) for path in outputs},
    }
    manifest_path = output_dir / "matched_lambda_fragment_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote matched-Lambda fragment sweep to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
