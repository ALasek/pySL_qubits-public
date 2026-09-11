#!/usr/bin/env python3
"""Build Paper-A publication diagnostics from completed v2 batches without rerunning them."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


FAMILY_BATCHES = {
    "lambda": ("Paper_A_final_LambdaCollapse", "Paper_A_final_SubmissionMatchedLambda"),
    "uniform": ("Paper_A_final_UniformField",),
    "disorder": ("Paper_A_final_DisorderField",),
    "stability": ("Paper_A_final_StabilityMap",),
}
WINDOWS = {
    "lambda": (20.0, 40.0),
    "uniform": (30.0, 60.0),
    "disorder": (30.0, 60.0),
    "stability": (20.0, 40.0),
}
METRICS = (
    "holevo_fhalf",
    "discord_fhalf",
    "redundancy_q10",
    "kappa_typ",
    "kappa_ann",
    "holevo_fhalf_min",
)
LABELS = {
    "holevo_fhalf": r"$\chi_z(f=1/2)/H_Z(S)$",
    "discord_fhalf": r"$D_z(f=1/2)/H_Z(S)$",
    "redundancy_q10": r"$R_{0.1}^{(q_{0.1})}$",
    "kappa_typ": r"$\kappa_{\mathrm{typ}}$",
    "kappa_ann": r"$\kappa_{\mathrm{ann}}$",
    "holevo_fhalf_min": r"late-window min of $q_{0.1}[\chi_z(f=1/2)]/H_Z(S)$",
}


@dataclass
class Point:
    family: str
    batch: str
    run_id: str
    params: dict
    window: tuple[float, float]
    window_points: int
    metrics: dict[str, tuple[float, float, float, int]]
    formation: tuple[float, float, float, int]
    lifetime: tuple[float, float, float, int]
    lambda_value: float
    q10_source: str


def alignment_lambda(p: float, theta: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"psi_bias must lie in [0, 1], got {p}")
    dot = 2.0 * math.sqrt(p * (1.0 - p)) * math.cos(theta)
    dot += (2.0 * p - 1.0) * math.sin(theta)
    return float(np.clip(1.0 - dot * dot, 0.0, 1.0))


def _validate_model(params: dict, run_dir: Path):
    expected = {
        "nqubits_S": 1,
        "H_SE_J": 0.1,
        "H_S_J": 0.0,
        "H_SE_Special": "Mironowicz_rand",
        "H_SE_bonds": "S_to_all",
        "H_SE_p": "Norm",
        "psi_S_spec": "x+",
        "compute_discord": True,
    }
    for key, target in expected.items():
        if key not in params:
            raise ValueError(f"{run_dir}: missing required Paper-A field {key}")
        value = params[key]
        if isinstance(target, float):
            matches = math.isclose(float(value), target, abs_tol=1e-12)
        else:
            matches = value == target
        if not matches:
            raise ValueError(f"{run_dir}: expected {key}={target!r}, got {value!r}")


def _pointer_entropy(results, n_times: int) -> np.ndarray:
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


def _bootstrap(values: np.ndarray, rng: np.random.Generator, count: int):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan, math.nan, 0
    estimate = float(np.mean(finite))
    if finite.size == 1 or count == 0:
        return estimate, math.nan, math.nan, int(finite.size)
    indices = rng.integers(0, finite.size, size=(count, finite.size))
    draws = np.mean(finite[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high), int(finite.size)


def _late_values(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(series[mask], axis=0)


def _late_minimum(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-realization minimum over the late window; a threshold-free persistence measure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(series[mask], axis=0)


def _q10_holevo(results, n_environment: int):
    raw = np.asarray(results["Holevo_Z_S_Ef_fractionsT"], dtype=float)
    quantiles = results.get("Holevo_Z_S_Ef_fractionsT_quantiles")
    levels = results.get("fragment_quantile_levels")
    if quantiles is None or levels is None:
        return raw, "mean_fallback"
    quantiles = np.asarray(quantiles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if quantiles.ndim != 4 or not levels.size:
        return raw, "mean_fallback"
    index = int(np.argmin(np.abs(levels - 0.1)))
    return quantiles[index, : n_environment + 1], f"stored_q{levels[index]:g}"


def _redundancy(q_holevo: np.ndarray, entropy: np.ndarray, n_environment: int) -> np.ndarray:
    n_times, n_runs = q_holevo.shape[1:]
    values = np.full((n_times, n_runs), np.nan)
    upper = min(q_holevo.shape[0], n_environment + 1)
    for time_index in range(n_times):
        threshold = 0.9 * entropy[time_index]
        if not math.isfinite(float(threshold)) or threshold <= 1e-12:
            continue
        for run_index in range(n_runs):
            for fragment_size in range(1, upper):
                if q_holevo[fragment_size, time_index, run_index] >= threshold:
                    values[time_index, run_index] = n_environment / fragment_size
                    break
    return values


def _record_windows(
    times: np.ndarray,
    q10_half: np.ndarray,
    discord_half: np.ndarray,
    entropy: np.ndarray,
    late_mask: np.ndarray,
    sustained: int,
    formation_min_time: float,
):
    # Record criterion: conservative half-environment Holevo information only; the
    # pointer-basis remainder is reported separately rather than folded in.
    qualifying = q10_half >= 0.9 * entropy[:, None]
    qualifying[times < formation_min_time] = False
    formation = np.full(qualifying.shape[1], np.nan)
    lifetime = np.full(qualifying.shape[1], np.nan)
    occupancy = np.mean(qualifying[late_mask], axis=0)
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    for run_index in range(qualifying.shape[1]):
        flags = qualifying[:, run_index]
        starts = [
            index
            for index in range(0, max(0, flags.size - sustained + 1))
            if np.all(flags[index : index + sustained])
        ]
        if not starts:
            continue
        start = starts[0]
        stop = start
        while stop < flags.size and flags[stop]:
            stop += 1
        formation[run_index] = times[start]
        lifetime[run_index] = (stop - start) * dt
    return formation, lifetime, occupancy


def _load_point(
    family: str,
    batch: Path,
    run_dir: Path,
    window: tuple[float, float],
    rng: np.random.Generator,
    bootstrap: int,
    sustained: int,
) -> Point:
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    _validate_model(params, run_dir)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        holevo = np.asarray(results["Holevo_Z_S_Ef_fractionsT"], dtype=float)
        discord = np.asarray(results["Discord_Z_S_Ef_fractionsT"], dtype=float)
        if holevo.ndim != 3 or discord.shape != holevo.shape:
            raise ValueError(f"{run_dir}: unexpected Holevo/discord shapes")
        n_environment = int(params.get("nqubits_E", params.get("nqubitsE")))
        half = int(np.argmin(np.abs(np.arange(holevo.shape[0]) / n_environment - 0.5)))
        times = np.arange(holevo.shape[1], dtype=float) * float(params["printT"])
        late_mask = (times >= window[0]) & (times <= window[1])
        if not np.any(late_mask):
            raise ValueError(
                f"{run_dir}: no stored samples in late window {window}; "
                f"stored range is [{times[0]}, {times[-1]}]"
            )
        entropy = _pointer_entropy(results, times.size)
        safe_entropy = np.where(entropy > 1e-12, entropy, np.nan)
        holevo_half = holevo[half] / safe_entropy[:, None]
        discord_half_raw = discord[half]
        discord_half = discord_half_raw / safe_entropy[:, None]
        q_holevo, q10_source = _q10_holevo(results, n_environment)
        q10_half = q_holevo[half]
        q10_half_norm = q10_half / safe_entropy[:, None]
        redundancy = _redundancy(q_holevo, entropy, n_environment)

        fidelity_key = "SBS_fid_runs_1" if "SBS_fid_runs_1" in results else "SBS_fid_1"
        fidelity = np.clip(np.asarray(results[fidelity_key], dtype=float), 1e-300, 1.0)
        if fidelity.ndim == 2:
            fidelity = fidelity[:, :, None]
        kappa_typ = -0.5 * np.nanmean(np.log(fidelity), axis=0)
        kappa_ann = -np.log(np.nanmean(np.sqrt(fidelity), axis=0))
        if np.nanmin(kappa_typ - kappa_ann) < -1e-10:
            raise ValueError(f"{run_dir}: kappa_typ < kappa_ann; check fidelity convention")

        formation, lifetime, occupancy = _record_windows(
            times,
            q10_half,
            discord_half_raw,
            entropy,
            late_mask,
            sustained,
            5.0,
        )
        per_run = {
            "holevo_fhalf": _late_values(holevo_half, late_mask),
            "discord_fhalf": _late_values(discord_half, late_mask),
            "redundancy_q10": _late_values(redundancy, late_mask),
            "kappa_typ": _late_values(kappa_typ, late_mask),
            "kappa_ann": _late_values(kappa_ann, late_mask),
            "holevo_fhalf_min": _late_minimum(q10_half_norm, late_mask),
        }
    p = float(params["psi_bias"])
    theta = float(params["Mironowicz_theta"])
    return Point(
        family=family,
        batch=batch.name,
        run_id=run_dir.name,
        params=params,
        window=window,
        window_points=int(np.sum(late_mask)),
        metrics={key: _bootstrap(value, rng, bootstrap) for key, value in per_run.items()},
        formation=_bootstrap(formation, rng, bootstrap),
        lifetime=_bootstrap(lifetime, rng, bootstrap),
        lambda_value=float(params.get("lambda_target", alignment_lambda(p, theta))),
        q10_source=q10_source,
    )


def _discover(data_root: Path, family: str, include_incomplete: bool = False):
    found = []
    missing = []
    completeness = []
    for name in FAMILY_BATCHES[family]:
        batch = data_root / name
        if not (batch / "runs").is_dir():
            missing.append(str(batch))
            continue
        run_dirs = sorted(path.parent for path in (batch / "runs").glob("*/params.json"))
        complete = [path for path in run_dirs if (path / "results.npz").is_file()]
        manifest_path = batch / "batch_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        expected = manifest.get("combo_count")
        incomplete = expected is not None and len(complete) < int(expected)
        completeness.append(
            {
                "batch": str(batch.resolve()),
                "manifest_status": manifest.get("status"),
                "expected_points": expected,
                "complete_points": len(complete),
                "incomplete": incomplete,
                "included": bool(complete) and (include_incomplete or not incomplete),
            }
        )
        if complete and (include_incomplete or not incomplete):
            found.extend((batch, path) for path in complete)
        elif not complete and expected is None:
            missing.append(f"{batch} (no complete v2 runs)")
    return found, missing, completeness


def _row(point: Point):
    params = point.params
    row = {
        "family": point.family,
        "batch": point.batch,
        "run_id": point.run_id,
        "n_environment": int(params.get("nqubits_E", params.get("nqubitsE"))),
        "realizations": int(params["AverageOverRunsN"]),
        "psi_bias": float(params["psi_bias"]),
        "theta": float(params["Mironowicz_theta"]),
        "lambda": point.lambda_value,
        "uniform_field": float(params.get("Mironowicz_h0", 0.0)),
        "disorder_strength": float(params.get("Mironowicz_alpha2", 0.0)),
        "window_start": point.window[0],
        "window_end": point.window[1],
        "window_points": point.window_points,
        "q10_source": point.q10_source,
    }
    for metric, (value, low, high, count) in point.metrics.items():
        row[metric] = value
        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high
        row[f"{metric}_n"] = count
    for name, summary in (("formation_time", point.formation), ("record_lifetime", point.lifetime)):
        row[name], row[f"{name}_ci_low"], row[f"{name}_ci_high"], row[f"{name}_n"] = summary
    return row


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _lambda_group_rows(points: list[Point]):
    groups = {}
    for point in points:
        key = (int(point.params["nqubits_E"]), round(point.lambda_value, 9))
        groups.setdefault(key, []).append(point)
    rows = []
    for (n_environment, lambda_value), members in sorted(groups.items()):
        row = {
            "n_environment": n_environment,
            "lambda": lambda_value,
            "geometry_count": len(members),
        }
        for metric in METRICS:
            values = np.asarray([point.metrics[metric][0] for point in members], dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            row[f"{metric}_geometry_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else math.nan
            )
            row[f"{metric}_geometry_range"] = float(np.ptp(values)) if values.size else math.nan
        rows.append(row)
    return rows


def _plot_lines(points: list[Point], output: Path):
    family = points[0].family
    x_key = {
        "lambda": lambda point: point.lambda_value,
        "uniform": lambda point: float(point.params["Mironowicz_h0"]),
        "disorder": lambda point: float(point.params["Mironowicz_alpha2"]),
    }[family]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)
    groups = {}
    for point in points:
        if family == "lambda":
            key = (int(point.params["nqubits_E"]), point.params.get("lambda_geometry", "legacy"))
        else:
            key = (float(point.params["psi_bias"]), float(point.params["Mironowicz_theta"]))
        groups.setdefault(key, []).append(point)

    def label(key):
        if family == "lambda":
            n_environment, geometry = key
            suffix = "" if geometry == "legacy" else f", {geometry.replace('_', ' ')}"
            return rf"$N_E={n_environment}$" + suffix
        p, theta = key
        return rf"$p={p:g},\ \theta/\pi={theta / math.pi:g}$"

    for axis, metric in zip(axes.flat, METRICS):
        for key, members in sorted(groups.items(), key=lambda item: str(item[0])):
            members = sorted(members, key=x_key)
            x = np.asarray([x_key(point) for point in members])
            y = np.asarray([point.metrics[metric][0] for point in members])
            low = np.asarray([point.metrics[metric][1] for point in members])
            high = np.asarray([point.metrics[metric][2] for point in members])
            errors = np.vstack((y - low, high - y))
            errors = None if not np.any(np.isfinite(errors)) else np.maximum(errors, 0.0)
            axis.errorbar(
                x,
                y,
                yerr=errors,
                marker="o",
                markersize=3.5,
                linewidth=1.0,
                label=label(key),
            )
        axis.set_ylabel(LABELS[metric])
        axis.grid(alpha=0.2)
        axis.set_xlabel(
            r"$\Lambda(p,\theta)$"
            if family == "lambda"
            else (r"$h_0$" if family == "uniform" else r"$W$")
        )
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"Paper A: {family} late-window diagnostics")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _grid(points: list[Point], metric: str, theta: float):
    selected = [point for point in points if math.isclose(float(point.params["Mironowicz_theta"]), theta)]
    p_values = sorted({float(point.params["psi_bias"]) for point in selected})
    w_values = sorted({float(point.params["Mironowicz_alpha2"]) for point in selected})
    grid = np.full((len(w_values), len(p_values)), np.nan)
    for point in selected:
        row = w_values.index(float(point.params["Mironowicz_alpha2"]))
        column = p_values.index(float(point.params["psi_bias"]))
        grid[row, column] = point.metrics[metric][0]
    return p_values, w_values, grid


def _edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    midpoints = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [values[-1] + (values[-1] - midpoints[-1])])
    )


def _stability_rows(points: list[Point], theta: float, coupling: float) -> list[dict]:
    rows = []
    for point in sorted(
        (item for item in points if math.isclose(float(item.params["Mironowicz_theta"]), theta)),
        key=lambda item: (float(item.params["Mironowicz_alpha2"]), float(item.params["psi_bias"])),
    ):
        row = {
            "p": float(point.params["psi_bias"]),
            "field_width": float(point.params["Mironowicz_alpha2"]),
            "field_width_over_g": float(point.params["Mironowicz_alpha2"]) / coupling,
        }
        row.update({metric: point.metrics[metric][0] for metric in METRICS})
        rows.append(row)
    return rows


def _plot_stability(points: list[Point], output_dir: Path):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.9")
    for theta in sorted({float(point.params["Mironowicz_theta"]) for point in points}):
        selected = [
            point
            for point in points
            if math.isclose(float(point.params["Mironowicz_theta"]), theta)
        ]
        couplings = {float(point.params["H_SE_J"]) for point in selected}
        if len(couplings) != 1:
            raise ValueError(f"stability slice has inconsistent H_SE_J values: {couplings}")
        coupling = couplings.pop()
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)
        for panel_index, (axis, metric) in enumerate(zip(axes.flat, METRICS)):
            p_values, w_values, grid = _grid(points, metric, theta)
            field_ratios = np.asarray(w_values, dtype=float) / coupling
            row_centers = np.arange(field_ratios.size, dtype=float)
            image = axis.pcolormesh(
                _edges(p_values),
                _edges(row_centers),
                grid,
                shading="flat",
                cmap=cmap,
            )
            fig.colorbar(image, ax=axis)
            axis.set_xlabel(r"$p$")
            axis.set_ylabel(r"sampled random-field width $W/g$")
            axis.set_yticks(row_centers)
            axis.set_yticklabels([f"{value:g}" for value in field_ratios])
            axis.tick_params(axis="y", labelsize=8)
            axis.set_title(LABELS[metric])
            # Panels whose top-left cell is dark get a white label for contrast.
            axis.text(
                0.02,
                0.97,
                f"({chr(ord('a') + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                color="white" if panel_index in (0, 3, 4, 5) else "black",
            )
        stem = f"stability_theta_{theta / math.pi:.3f}"
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        _write_csv(output_dir / f"{stem}.csv", _stability_rows(points, theta, coupling))


def build(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    points = []
    missing = {}
    batch_completeness = []
    failures = []
    for family in args.families:
        discovered, missing_paths, family_completeness = _discover(
            args.data_root,
            family,
            include_incomplete=args.include_incomplete,
        )
        missing[family] = missing_paths
        batch_completeness.extend(family_completeness)
        for batch, run_dir in discovered:
            try:
                points.append(
                    _load_point(
                        family,
                        batch,
                        run_dir,
                        WINDOWS[family],
                        rng,
                        args.bootstrap,
                        args.sustained,
                    )
                )
            except (KeyError, ValueError, OSError) as exc:
                failures.append({"run": str(run_dir), "error": str(exc)})
    rows = [_row(point) for point in points]
    _write_csv(args.output_dir / "paper_a_late_window_points.csv", rows)
    lambda_points = [point for point in points if point.family == "lambda"]
    _write_csv(args.output_dir / "paper_a_lambda_groups.csv", _lambda_group_rows(lambda_points))
    for family in args.families:
        selected = [point for point in points if point.family == family]
        if family in ("lambda", "uniform", "disorder") and selected:
            _plot_lines(selected, args.output_dir / f"{family}_late_window.png")
        elif family == "stability" and selected:
            _plot_stability(selected, args.output_dir)
    source_batches = sorted({str((args.data_root / point.batch).resolve()) for point in points})
    metric_finite_counts = {
        metric: sum(math.isfinite(point.metrics[metric][0]) for point in points) for metric in METRICS
    }
    provenance = {}
    try:
        provenance = {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).resolve().parents[1],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        }
    except (OSError, subprocess.SubprocessError):
        pass
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "requested_families": args.families,
        "loaded_point_count": len(points),
        "metric_finite_counts": metric_finite_counts,
        "source_batch_paths": source_batches,
        "analysis_code_provenance": provenance,
        "loaded_by_family": {
            family: sum(point.family == family for point in points) for family in args.families
        },
        "missing_batch_paths": missing,
        "batch_completeness": batch_completeness,
        "incomplete_batches_included": args.include_incomplete,
        "failed_runs": failures,
        "late_windows": {family: WINDOWS[family] for family in args.families},
        "estimator": "mean across realization-level late-window medians",
        "confidence_interval": f"95% percentile bootstrap over realizations ({args.bootstrap} draws)",
        "normalization": (
            "Holevo and discord divided by H_Z(S) from the stored rhoS_T diagonal; "
            "ln(2) fallback is allowed only after Paper-A model validation"
        ),
        "kappa_typ": "-0.5 * mean_site(log(F)); stored fidelity is F=B^2",
        "kappa_ann": "-log(mean_site(sqrt(F)))",
        "redundancy": "N_E / smallest m with stored q10 Holevo >= 0.9 H_Z(S)",
        "record_rule": (
            "q10 Holevo(f=1/2) >= 0.9 H_Z(S); "
            f"formation is searched from t=5 and requires {args.sustained} consecutive stored samples"
        ),
        "holevo_fhalf_min": (
            "mean across realizations of the late-window minimum of q10 Holevo(f=1/2)/H_Z(S); "
            "threshold-free persistence measure replacing record occupancy"
        ),
        "bootstrap_level": "realization (stored fragment samples are not resampled)",
    }
    stability_points = [point for point in points if point.family == "stability"]
    if stability_points:
        couplings = {float(point.params["H_SE_J"]) for point in stability_points}
        if len(couplings) != 1:
            raise ValueError(f"stability data have inconsistent H_SE_J values: {couplings}")
        coupling = couplings.pop()
        payload["stability_display"] = {
            "field_axis": "equally spaced sampled rows labeled by W/g",
            "sampled_W_over_g": sorted(
                {float(point.params["Mironowicz_alpha2"]) / coupling for point in stability_points}
            ),
            "slice_csvs": [
                f"stability_theta_{theta / math.pi:.3f}.csv"
                for theta in sorted(
                    {float(point.params["Mironowicz_theta"]) for point in stability_points}
                )
            ],
        }
    (args.output_dir / "paper_a_analysis_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"loaded {len(points)} parameter points")
    for family in args.families:
        count = sum(point.family == family for point in points)
        print(f"{family}: {count} points; {len(missing[family])} missing batch path(s)")
    incomplete = [entry for entry in batch_completeness if entry["incomplete"]]
    for entry in incomplete:
        action = "included" if entry["included"] else "excluded"
        print(
            f"incomplete ({action}): {entry['batch']} "
            f"{entry['complete_points']}/{entry['expected_points']} points"
        )
    if failures:
        print(f"skipped {len(failures)} incomplete/incompatible run(s); see manifest")
    if args.strict_missing and (any(missing.values()) or failures or incomplete):
        raise SystemExit("requested raw batches are missing or incomplete; see analysis manifest")
    if not points:
        raise SystemExit("no compatible Paper-A v2 runs found")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/figs/Paper_A_submission"))
    parser.add_argument(
        "--families",
        nargs="+",
        choices=tuple(FAMILY_BATCHES),
        default=list(FAMILY_BATCHES),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--sustained", type=int, default=3)
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="include available points from batches whose manifest count is not yet complete",
    )
    parser.add_argument("--strict-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    if args.sustained < 1:
        parser.error("--sustained must be positive")
    return args


def main(argv=None):
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
