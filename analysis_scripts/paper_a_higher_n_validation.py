#!/usr/bin/env python3
"""Analyze the focused Paper-A higher-size validation batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


CASE_ORDER = (
    "aligned_uniform_rescue",
    "aligned_random_rescue",
    "aligned_uniform_strong",
    "aligned_random_strong",
)
CASE_LABELS = {
    "aligned_uniform_rescue": "uniform\n$0.25g$",
    "aligned_random_rescue": "Gaussian\n$0.5g$",
    "aligned_uniform_strong": "uniform\n$30g$",
    "aligned_random_strong": "Gaussian\n$30g$",
}
COLORS = {16: "#0072B2", 20: "#D55E00"}


@dataclass
class Point:
    source_batch: str
    run_id: str
    case: str
    params: dict
    window: tuple[float, float]
    entropy: np.ndarray
    q10_holevo: np.ndarray
    q90_discord: np.ndarray
    mean_discord: np.ndarray
    redundancy: np.ndarray
    holevo_min: np.ndarray
    health: dict[str, float]

    @property
    def n_environment(self) -> int:
        return int(self.params["nqubits_E"])

    @property
    def half(self) -> int:
        return self.n_environment // 2

    @property
    def realization_count(self) -> int:
        return int(self.q10_holevo.shape[-1])

    @property
    def source_realization_count(self) -> int:
        return int(self.params["AverageOverRunsN"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info(root: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def _run_dirs(batch_dir: Path) -> list[tuple[Path, dict]]:
    complete = []
    for params_path in sorted((batch_dir / "runs").glob("*/params.json")):
        run_dir = params_path.parent
        if not (run_dir / "results.npz").is_file():
            continue
        complete.append((run_dir, json.loads(params_path.read_text(encoding="utf-8"))))
    return complete


def _close(left, right) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _match_reference(case: str, params: dict, matched_dir: Path, field_dir: Path):
    if case == "no_field_lambda_half":
        candidates = _run_dirs(matched_dir)

        def matches(candidate):
            return (
                _close(candidate.get("psi_bias", -1), 0.5)
                and _close(candidate.get("Mironowicz_theta", -1), math.pi / 4)
                and _close(candidate.get("Mironowicz_h0", 0), 0)
                and _close(candidate.get("Mironowicz_alpha2", 0), 0)
            )

    else:
        candidates = _run_dirs(field_dir)

        def matches(candidate):
            return (
                candidate.get("field_geometry") == "aligned"
                and candidate.get("field_profile") == params.get("field_profile")
                and _close(
                    candidate.get("field_strength_ratio", -1),
                    params.get("field_strength_ratio", -2),
                )
            )

    selected = [(run_dir, candidate) for run_dir, candidate in candidates if matches(candidate)]
    if len(selected) != 1:
        raise ValueError(f"{case}: expected one N_E=16 reference, found {len(selected)}")
    return selected[0]


def _pointer_entropy(results, time_count: int) -> np.ndarray:
    rho = np.asarray(results["rhoS_T"])
    if rho.shape != (time_count, 2, 2):
        raise ValueError(f"unexpected rhoS_T shape {rho.shape}")
    probabilities = np.real(np.diagonal(rho, axis1=1, axis2=2)).copy()
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            probabilities > 0,
            -probabilities * np.log(probabilities),
            0.0,
        ).sum(axis=1)


def _redundancy(q10_holevo, entropy, n_environment):
    time_count, realization_count = q10_holevo.shape[1:]
    values = np.full((time_count, realization_count), np.nan)
    for time_index in range(time_count):
        threshold = 0.9 * entropy[time_index]
        for realization_index in range(realization_count):
            for size in range(1, n_environment // 2 + 1):
                if q10_holevo[size, time_index, realization_index] >= threshold:
                    values[time_index, realization_index] = n_environment / size
                    break
    return values


def _validate_params(params: dict, expected_n: int, expected_runs: int, run_dir: Path):
    expected = {
        "nqubits_S": 1,
        "nqubits_E": expected_n,
        "AverageOverRunsN": expected_runs,
        "H_SE_Special": "Mironowicz_rand",
        "H_SE_J": 0.1,
        "H_EE_J": 0.0,
        "compute_discord": True,
        "fragment_half_only": True,
        "psi_E_spec": "bias",
        "psi_S_spec": "x+",
        "psi_bias": 0.5,
        "store_fragment_samples": True,
    }
    for key, target in expected.items():
        value = params.get(key)
        if isinstance(target, float):
            valid = value is not None and _close(value, target)
        else:
            valid = value == target
        if not valid:
            raise ValueError(f"{run_dir}: expected {key}={target!r}, got {value!r}")


def _load_point(
    source_batch: str,
    run_dir: Path,
    params: dict,
    case: str,
    window: tuple[float, float],
    expected_n: int,
    expected_runs: int,
    analysis_runs: int | None = None,
) -> Point:
    _validate_params(params, expected_n, expected_runs, run_dir)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        holevo_samples = np.asarray(results["Holevo_Z_S_Ef_fractionsT_samples"], dtype=float)
        discord_samples = np.asarray(results["Discord_Z_S_Ef_fractionsT_samples"], dtype=float)
        expected_times = int(round(float(params["T"]) / float(params["printT"]))) + 1
        expected_prefix = (expected_n + 1, expected_times, expected_runs)
        if holevo_samples.shape[:3] != expected_prefix or discord_samples.shape != holevo_samples.shape:
            raise ValueError(
                f"{run_dir}: unexpected stored-fragment shapes "
                f"{holevo_samples.shape}/{discord_samples.shape}"
            )
        sample_count = int(params["fragment_sample_count"])
        if holevo_samples.shape[3] != sample_count:
            raise ValueError(f"{run_dir}: fragment sample count does not match params")

        levels = np.asarray(results["fragment_quantile_levels"], dtype=float)
        q10_index = int(np.argmin(np.abs(levels - 0.1)))
        q90_index = int(np.argmin(np.abs(levels - 0.9)))
        if not _close(levels[q10_index], 0.1) or not _close(levels[q90_index], 0.9):
            raise ValueError(f"{run_dir}: stored quantile levels do not contain 0.1 and 0.9")

        valid = slice(1, expected_n // 2 + 1)
        reduced_shape = holevo_samples.shape[:3]
        q10_holevo = np.full(reduced_shape, np.nan)
        q90_discord = np.full(reduced_shape, np.nan)
        mean_discord = np.full(reduced_shape, np.nan)
        q10_holevo[valid] = np.nanquantile(holevo_samples[valid], 0.1, axis=3)
        q90_discord[valid] = np.nanquantile(discord_samples[valid], 0.9, axis=3)
        mean_discord[valid] = np.nanmean(discord_samples[valid], axis=3)
        stored_q10 = np.asarray(results["Holevo_Z_S_Ef_fractionsT_quantiles"])[q10_index]
        stored_q90 = np.asarray(results["Discord_Z_S_Ef_fractionsT_quantiles"])[q90_index]
        stored_mean_discord = np.asarray(results["Discord_Z_S_Ef_fractionsT"])
        if not np.allclose(q10_holevo[valid], stored_q10[valid], equal_nan=True, atol=1e-12):
            raise ValueError(f"{run_dir}: stored q10 Holevo values do not match raw samples")
        if not np.allclose(q90_discord[valid], stored_q90[valid], equal_nan=True, atol=1e-12):
            raise ValueError(f"{run_dir}: stored q90 discord values do not match raw samples")
        if not np.allclose(
            mean_discord[valid], stored_mean_discord[valid], equal_nan=True, atol=1e-12
        ):
            raise ValueError(f"{run_dir}: stored mean discord does not match raw samples")

        times = np.arange(holevo_samples.shape[1], dtype=float) * float(params["printT"])
        late = (times >= window[0]) & (times <= window[1])
        if int(np.count_nonzero(late)) != int(window[1] - window[0]) + 1:
            raise ValueError(f"{run_dir}: incomplete late window {window}")
        entropy = _pointer_entropy(results, times.size)
        safe_entropy = np.where(entropy > 1e-12, entropy, np.nan)
        q10_holevo_norm = q10_holevo / safe_entropy[None, :, None]
        q90_discord_norm = q90_discord / safe_entropy[None, :, None]
        mean_discord_norm = mean_discord / safe_entropy[None, :, None]
        redundancy = _redundancy(q10_holevo, entropy, expected_n)
        half = expected_n // 2
        # Late-window minimum of the conservative half-environment Holevo information.
        holevo_min = np.nanmin(q10_holevo_norm[half][late], axis=0)

        norms = np.asarray(results["norms"], dtype=float)
        rho = np.asarray(results["rhoS_T"])
        health = {
            "max_norm_error": float(np.max(np.abs(norms - 1.0))),
            "max_rho_trace_error": float(
                np.max(np.abs(np.trace(rho, axis1=1, axis2=2) - 1.0))
            ),
            "max_rho_hermiticity_error": float(
                np.max(np.abs(rho - np.swapaxes(rho.conj(), 1, 2)))
            ),
        }

    selected = slice(None, analysis_runs)
    return Point(
        source_batch=source_batch,
        run_id=run_dir.name,
        case=case,
        params=params,
        window=window,
        entropy=entropy,
        q10_holevo=q10_holevo_norm[..., selected],
        q90_discord=q90_discord_norm[..., selected],
        mean_discord=mean_discord_norm[..., selected],
        redundancy=redundancy[..., selected],
        holevo_min=holevo_min[selected],
        health=health,
    )


def _late_values(values: np.ndarray, point: Point) -> np.ndarray:
    times = np.arange(values.shape[-2], dtype=float) * float(point.params["printT"])
    late = (times >= point.window[0]) & (times <= point.window[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values[..., late, :], axis=-2)


def _bootstrap_mean(values, rng, draws):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan, math.nan, 0
    estimate = float(np.mean(finite))
    if finite.size == 1 or draws == 0:
        return estimate, math.nan, math.nan, int(finite.size)
    indices = rng.integers(0, finite.size, size=(draws, finite.size))
    samples = np.mean(finite[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return estimate, float(low), float(high), int(finite.size)


def _summary_metrics(point: Point, rng, draws):
    half = point.half
    redundancy = _late_values(point.redundancy, point)
    metrics = {
        "q10_holevo_half": _late_values(point.q10_holevo[half], point),
        "q90_discord_half": _late_values(point.q90_discord[half], point),
        "mean_discord_half": _late_values(point.mean_discord[half], point),
        "redundancy_q10_conditional": redundancy,
        "holevo_fhalf_min": point.holevo_min,
    }
    summary = {name: _bootstrap_mean(values, rng, draws) for name, values in metrics.items()}
    summary["redundancy_q10_defined_fraction"] = (
        float(np.mean(np.isfinite(redundancy))),
        math.nan,
        math.nan,
        int(redundancy.size),
    )
    return summary


def _curve_rows(point: Point, rng, draws):
    rows = []
    for metric, values in (
        ("q10_holevo", point.q10_holevo),
        ("q90_discord", point.q90_discord),
    ):
        late = _late_values(values, point)
        for size in range(1, point.half + 1):
            estimate, low, high, count = _bootstrap_mean(late[size], rng, draws)
            rows.append(
                {
                    "source_batch": point.source_batch,
                    "run_id": point.run_id,
                    "case": point.case,
                    "n_environment": point.n_environment,
                    "realizations": point.realization_count,
                    "source_realizations": point.source_realization_count,
                    "fragment_samples": int(point.params["fragment_sample_count"]),
                    "window_start": point.window[0],
                    "window_end": point.window[1],
                    "metric": metric,
                    "fragment_size": size,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "finite_count": count,
                }
            )
    return rows


def _summary_rows(point: Point, rng, draws):
    rows = []
    for metric, (estimate, low, high, count) in _summary_metrics(point, rng, draws).items():
        rows.append(
            {
                "source_batch": point.source_batch,
                "run_id": point.run_id,
                "case": point.case,
                "n_environment": point.n_environment,
                "realizations": point.realization_count,
                "source_realizations": point.source_realization_count,
                "fragment_samples": int(point.params["fragment_sample_count"]),
                "window_start": point.window[0],
                "window_end": point.window[1],
                "metric": metric,
                "fragment_size": point.half,
                "estimate": estimate,
                "ci_low": low,
                "ci_high": high,
                "finite_count": count,
            }
        )
    return rows


def _row_map(rows):
    return {
        (row["case"], int(row["n_environment"]), row["metric"], int(row["fragment_size"])): row
        for row in rows
    }


def _errorbar(axis, x, rows, color, label):
    y = np.asarray([row["estimate"] for row in rows])
    low = np.asarray([row["ci_low"] for row in rows])
    high = np.asarray([row["ci_high"] for row in rows])
    axis.errorbar(
        x,
        y,
        yerr=np.maximum(np.vstack((y - low, high - y)), 0.0),
        color=color,
        marker="o",
        markersize=4.0,
        linewidth=1.35,
        linestyle="none",
        capsize=2.0,
        label=label,
    )


def _plot(rows, output_dir: Path):
    lookup = _row_map(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)

    for axis, metric, ylabel in (
        (axes[0, 0], "q10_holevo", r"$q_{0.1}(\chi_z)/H_Z(\mathcal{S})$"),
        (axes[0, 1], "q90_discord", r"$q_{0.9}(D_z)/H_Z(\mathcal{S})$"),
    ):
        for n_environment in (16, 20):
            members = [
                lookup[("no_field_lambda_half", n_environment, metric, size)]
                for size in range(1, n_environment // 2 + 1)
            ]
            x = np.asarray([member["fragment_size"] for member in members])
            y = np.asarray([member["estimate"] for member in members])
            low = np.asarray([member["ci_low"] for member in members])
            high = np.asarray([member["ci_high"] for member in members])
            axis.plot(x, y, color=COLORS[n_environment], marker="o", markersize=3.2)
            axis.fill_between(x, low, high, color=COLORS[n_environment], alpha=0.16)
        axis.set_xlabel(r"fragment size $m$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].axhline(0.9, color="0.4", linestyle=":", linewidth=0.9)
    axes[0, 0].set_ylim(-0.02, 1.03)
    axes[0, 1].set_ylim(-0.005, 0.25)

    x = np.arange(len(CASE_ORDER), dtype=float)
    offsets = {16: -0.09, 20: 0.09}
    for axis, metric, ylabel in (
        (axes[1, 0], "q10_holevo_half", r"$q_{0.1}(\chi_z)/H_Z(\mathcal{S})$ at $f=1/2$"),
        (axes[1, 1], "holevo_fhalf_min", r"late-window min of $q_{0.1}(\chi_z)/H_Z(\mathcal{S})$ at $f=1/2$"),
    ):
        for n_environment in (16, 20):
            members = [
                lookup[(case, n_environment, metric, n_environment // 2)]
                for case in CASE_ORDER
            ]
            _errorbar(
                axis,
                x + offsets[n_environment],
                members,
                COLORS[n_environment],
                rf"$N_{{\mathcal{{E}}}}={n_environment}$",
            )
        axis.axhline(0.9, color="0.4", linestyle=":", linewidth=0.9)
        axis.set_xticks(x)
        axis.set_xticklabels([CASE_LABELS[case] for case in CASE_ORDER])
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.04, 1.05)
        axis.grid(axis="y", alpha=0.2)

    handles = [
        plt.Line2D([], [], color=COLORS[n], marker="o", label=rf"$N_{{\mathcal{{E}}}}={n}$")
        for n in (16, 20)
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    label_positions = ((0.02, 0.94), (0.02, 0.94), (0.02, 0.84), (0.02, 0.84))
    for label, axis, position in zip("abcd", axes.flat, label_positions):
        axis.text(
            *position,
            f"({label})",
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"higher_n_validation.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    high_manifest_path = args.high_n_batch_dir / "batch_manifest.json"
    high_manifest = json.loads(high_manifest_path.read_text(encoding="utf-8"))
    if high_manifest.get("status") != "completed" or high_manifest.get("combo_count") != 5:
        raise ValueError("higher-N batch is not a complete five-case run")
    if high_manifest.get("source_provenance", {}).get("git_dirty"):
        raise ValueError("higher-N batch records a dirty source tree")

    high_runs = _run_dirs(args.high_n_batch_dir)
    if len(high_runs) != 5:
        raise ValueError(f"expected five higher-N run artifacts, found {len(high_runs)}")

    points = []
    pairings = []
    for run_dir, params in high_runs:
        case = params["validation_case"]
        window = (20.0, 40.0) if case == "no_field_lambda_half" else (30.0, 60.0)
        high_point = _load_point(
            args.high_n_batch_dir.name,
            run_dir,
            params,
            case,
            window,
            expected_n=20,
            expected_runs=8,
            analysis_runs=8,
        )
        reference_dir, reference_params = _match_reference(
            case,
            params,
            args.matched_lambda_dir,
            args.field_control_dir,
        )
        reference_point = _load_point(
            reference_dir.parents[1].name,
            reference_dir,
            reference_params,
            case,
            window,
            expected_n=16,
            expected_runs=24,
            analysis_runs=8,
        )
        if reference_params["run_seeds"][:8] != params["run_seeds"]:
            raise ValueError(f"{case}: N_E=16 and N_E=20 realization seeds are not paired")
        points.extend((reference_point, high_point))
        pairings.append(
            {
                "case": case,
                "reference_run_id": reference_point.run_id,
                "higher_n_run_id": high_point.run_id,
                "first_eight_run_seeds_match": True,
            }
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    for point in points:
        rows.extend(_summary_rows(point, rng, args.bootstrap))
        if point.case == "no_field_lambda_half":
            rows.extend(_curve_rows(point, rng, args.bootstrap))
    rows.sort(
        key=lambda row: (
            row["case"],
            int(row["n_environment"]),
            row["metric"],
            int(row["fragment_size"]),
        )
    )

    csv_path = args.output_dir / "higher_n_validation_metrics.csv"
    _write_csv(csv_path, rows)
    _plot(rows, args.output_dir)

    lookup = _row_map(rows)
    highlights = {}
    for case in ("no_field_lambda_half", *CASE_ORDER):
        highlights[case] = {}
        for n_environment in (16, 20):
            highlights[case][str(n_environment)] = {
                metric: lookup[(case, n_environment, metric, n_environment // 2)]["estimate"]
                for metric in (
                    "q10_holevo_half",
                    "q90_discord_half",
                    "mean_discord_half",
                    "holevo_fhalf_min",
                    "redundancy_q10_defined_fraction",
                )
            }

    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    batch_inputs = []
    for batch_dir in (
        args.high_n_batch_dir,
        args.matched_lambda_dir,
        args.field_control_dir,
    ):
        manifest_path = batch_dir / "batch_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batch_inputs.append(
            {
                "batch": batch_dir.name,
                "manifest_sha256": _sha256(manifest_path),
                "source_provenance": manifest.get("source_provenance"),
            }
        )
    manifest_payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "targeted N_E=20 full-state validation against matching N_E=16 production runs",
        "analysis_source": {
            **_git_info(repo_root),
            "script": str(script_path.relative_to(repo_root)).replace("\\", "/"),
            "script_sha256": _sha256(script_path),
        },
        "inputs": batch_inputs,
        "estimators": {
            "windows": {
                "no_field_lambda_half": [20.0, 40.0],
                "field_profile_cases": [30.0, 60.0],
            },
            "cohort": "the same first eight run seeds at N_E=16 and N_E=20",
            "central_value": "mean over paired realization-level late-window medians",
            "interval": f"95% percentile bootstrap over the eight paired realizations, {args.bootstrap} draws",
            "holevo": "stored-fragment q10, normalized by pointer entropy H_Z(S)",
            "discord_curve": "stored-fragment q90, normalized by H_Z(S)",
            "holevo_fhalf_min": "mean over realizations of the late-window minimum of q10 Holevo at f=1/2, normalized by H_Z(S)",
        },
        "pairings": sorted(pairings, key=lambda item: item["case"]),
        "highlights": highlights,
        "health": {
            "max_norm_error": max(point.health["max_norm_error"] for point in points),
            "max_rho_trace_error": max(point.health["max_rho_trace_error"] for point in points),
            "max_rho_hermiticity_error": max(
                point.health["max_rho_hermiticity_error"] for point in points
            ),
        },
        "validation": {
            "complete_higher_n_cases": 5,
            "higher_n_failures": 0,
            "raw_fragment_samples_reproduce_stored_means_and_quantiles": True,
            "reference_and_higher_n_first_eight_seeds_match": True,
            "claim_boundary": "finite-size implementation cross-check, not thermodynamic scaling",
        },
        "outputs": {
            "figure_pdf_sha256": _sha256(args.output_dir / "higher_n_validation.pdf"),
            "figure_png_sha256": _sha256(args.output_dir / "higher_n_validation.png"),
            "metrics_csv_sha256": _sha256(csv_path),
        },
    }
    (args.output_dir / "higher_n_validation_analysis_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("processed 5/5 N_E=20 cases and 5 matching N_E=16 references")
    print(f"max norm error: {manifest_payload['health']['max_norm_error']:.3e}")
    print(f"max rho trace error: {manifest_payload['health']['max_rho_trace_error']:.3e}")
    for case, values in highlights.items():
        print(case)
        for n_environment, metrics in values.items():
            print(
                f"  N_E={n_environment}: q10 Holevo={metrics['q10_holevo_half']:.4f}, "
                f"late-window min q10 Holevo={metrics['holevo_fhalf_min']:.4f}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-n-batch-dir", type=Path, required=True)
    parser.add_argument("--matched-lambda-dir", type=Path, required=True)
    parser.add_argument("--field-control-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=918273)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
