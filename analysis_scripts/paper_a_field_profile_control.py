#!/usr/bin/env python3
"""Analyze the focused Paper-A matched-strength field-profile batch."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from analysis_scripts.paper_a_submission_panels import (
        _bootstrap,
        _late_minimum,
        _late_values,
        _pointer_entropy,
        _q10_holevo,
        _redundancy,
    )
except ModuleNotFoundError:  # Direct script execution.
    from paper_a_submission_panels import (
        _bootstrap,
        _late_minimum,
        _late_values,
        _pointer_entropy,
        _q10_holevo,
        _redundancy,
    )


METRICS = ("holevo_fhalf", "discord_fhalf", "redundancy_q10", "holevo_fhalf_min")
LABELS = {
    "holevo_fhalf": r"$\chi_z(f=1/2)/H_Z(S)$",
    "discord_fhalf": r"$D_z(f=1/2)/H_Z(S)$",
    "redundancy_q10": r"conditional $R_{0.1}^{(q_{0.1})}$",
    "holevo_fhalf_min": r"late-window min of $q_{0.1}[\chi_z(f=1/2)]/H_Z(S)$",
    "kappa_typ": r"$\kappa_{\rm typ}$",
    "kappa_ann": r"$\kappa_{\rm ann}$",
}
PROFILE_LEGEND = (
    ("uniform", r"uniform ($h_{\rm rms}=h_0$)"),
    ("random", r"Gaussian ($h_{\rm rms}=W$)"),
)
EXPECTED_RATIOS = {
    "aligned": (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    "longitudinal": (0.0, 0.5, 2.0, 10.0, 30.0),
}
PROFILE_COLORS = {"uniform": "#0072B2", "random": "#D55E00"}


@dataclass
class FieldPoint:
    run_id: str
    params: dict
    per_run: dict[str, np.ndarray]
    metrics: dict[str, tuple[float, float, float, int]]
    fragment_curves: dict[str, np.ndarray]
    q10_source: str


def _validate_params(params: dict, run_dir: Path):
    expected = {
        "AverageOverRunsN": 24,
        "H_EE_J": 0.0,
        "H_SE_J": 0.1,
        "H_SE_Special": "Mironowicz_rand",
        "H_SE_bonds": "S_to_all",
        "Mironowicz_ZZ_to_ZX_epsilon": 0.0,
        "Mironowicz_epsilon": 0.0,
        "Mironowicz_epsilon2": 0.0,
        "compute_discord": True,
        "fragment_half_only": True,
        "fragment_sample_count": 64,
        "nqubits_E": 16,
        "nqubits_S": 1,
        "psi_E_spec": "bias",
        "psi_S_spec": "x+",
        "psi_bias": 0.5,
        "store_fragment_samples": True,
    }
    for key, target in expected.items():
        value = params.get(key)
        if isinstance(target, float):
            matches = value is not None and math.isclose(float(value), target, abs_tol=1e-12)
        else:
            matches = value == target
        if not matches:
            raise ValueError(f"{run_dir}: expected {key}={target!r}, got {value!r}")

    geometry = params.get("field_geometry")
    profile = params.get("field_profile")
    ratio = float(params.get("field_strength_ratio", math.nan))
    if geometry not in EXPECTED_RATIOS or ratio not in EXPECTED_RATIOS[geometry]:
        raise ValueError(f"{run_dir}: unexpected geometry/strength {geometry!r}, {ratio!r}")
    if ratio == 0.0:
        if profile != "none":
            raise ValueError(f"{run_dir}: zero-strength case must use profile='none'")
    elif profile not in PROFILE_COLORS:
        raise ValueError(f"{run_dir}: unexpected nonzero field profile {profile!r}")
    theta = float(params["Mironowicz_theta"])
    target_theta = 0.0 if geometry == "aligned" else math.pi / 2.0
    if not math.isclose(theta, target_theta, abs_tol=1e-12):
        raise ValueError(f"{run_dir}: {geometry} case has theta={theta}")
    strength = 0.1 * ratio
    expected_uniform = strength if profile == "uniform" else 0.0
    expected_random = strength if profile == "random" else 0.0
    if not math.isclose(float(params["Mironowicz_h0"]), expected_uniform, abs_tol=1e-12):
        raise ValueError(f"{run_dir}: uniform field does not match declared strength ratio")
    if not math.isclose(float(params["Mironowicz_alpha2"]), expected_random, abs_tol=1e-12):
        raise ValueError(f"{run_dir}: random field does not match declared strength ratio")


def _load_point(
    run_dir: Path,
    window: tuple[float, float],
    rng: np.random.Generator,
    bootstrap: int,
    sustained: int,
) -> tuple[FieldPoint, dict]:
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    _validate_params(params, run_dir)
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        holevo = np.asarray(results["Holevo_Z_S_Ef_fractionsT"], dtype=float)
        discord = np.asarray(results["Discord_Z_S_Ef_fractionsT"], dtype=float)
        expected_shape = (17, 61, 24)
        if holevo.shape != expected_shape or discord.shape != expected_shape:
            raise ValueError(
                f"{run_dir}: expected Holevo/discord shape {expected_shape}, "
                f"got {holevo.shape}/{discord.shape}"
            )
        if not np.all(np.isfinite(holevo[:9])) or not np.all(np.isfinite(discord[:9])):
            raise ValueError(f"{run_dir}: nonfinite Holevo/discord value in evaluated m<=8 region")
        n_environment = 16
        half = n_environment // 2
        times = np.arange(holevo.shape[1], dtype=float) * float(params["printT"])
        late_mask = (times >= window[0]) & (times <= window[1])
        if not np.any(late_mask):
            raise ValueError(f"{run_dir}: no stored times in late window {window}")
        entropy = _pointer_entropy(results, times.size)
        safe_entropy = np.where(entropy > 1e-12, entropy, np.nan)
        holevo_norm = holevo / safe_entropy[None, :, None]
        discord_norm = discord / safe_entropy[None, :, None]
        q_holevo, q10_source = _q10_holevo(results, n_environment)
        q10_norm = q_holevo / safe_entropy[None, :, None]
        redundancy = _redundancy(q_holevo, entropy, n_environment)

        fidelity = np.clip(np.asarray(results["SBS_fid_runs_1"], dtype=float), 1e-300, 1.0)
        kappa_typ = -0.5 * np.nanmean(np.log(fidelity), axis=0)
        kappa_ann = -np.log(np.nanmean(np.sqrt(fidelity), axis=0))
        if np.nanmin(kappa_typ - kappa_ann) < -1e-10:
            raise ValueError(f"{run_dir}: kappa_typ < kappa_ann")

        per_run = {
            "holevo_fhalf": _late_values(holevo_norm[half], late_mask),
            "discord_fhalf": _late_values(discord_norm[half], late_mask),
            "redundancy_q10": _late_values(redundancy, late_mask),
            "holevo_fhalf_min": _late_minimum(q10_norm[half], late_mask),
            "kappa_typ": _late_values(kappa_typ, late_mask),
            "kappa_ann": _late_values(kappa_ann, late_mask),
        }
        q10_curve = np.full((half + 1, holevo.shape[2]), np.nan)
        q10_curve[1:] = np.nanmedian(q10_norm[1 : half + 1, late_mask], axis=1)
        curves = {
            "holevo_mean": np.nanmedian(holevo_norm[: half + 1, late_mask], axis=1),
            "holevo_q10": q10_curve,
            "discord_mean": np.nanmedian(discord_norm[: half + 1, late_mask], axis=1),
        }
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
    return (
        FieldPoint(
            run_id=run_dir.name,
            params=params,
            per_run=per_run,
            metrics={key: _bootstrap(values, rng, bootstrap) for key, values in per_run.items()},
            fragment_curves=curves,
            q10_source=q10_source,
        ),
        health,
    )


def _case_map(points: list[FieldPoint]):
    return {
        (
            point.params["field_geometry"],
            point.params["field_profile"],
            float(point.params["field_strength_ratio"]),
        ): point
        for point in points
    }


def _series(points: list[FieldPoint], geometry: str, profile: str):
    baseline = next(
        point
        for point in points
        if point.params["field_geometry"] == geometry
        and point.params["field_profile"] == "none"
    )
    selected = [
        point
        for point in points
        if point.params["field_geometry"] == geometry
        and point.params["field_profile"] == profile
    ]
    return sorted([baseline, *selected], key=lambda point: point.params["field_strength_ratio"])


def _metric_rows(points: list[FieldPoint]):
    rows = []
    for point in sorted(
        points,
        key=lambda item: (
            item.params["field_geometry"],
            float(item.params["field_strength_ratio"]),
            item.params["field_profile"],
        ),
    ):
        row = {
            "run_id": point.run_id,
            "field_geometry": point.params["field_geometry"],
            "field_profile": point.params["field_profile"],
            "field_strength_ratio": float(point.params["field_strength_ratio"]),
            "uniform_field_h0": float(point.params["Mironowicz_h0"]),
            "random_field_width_W": float(point.params["Mironowicz_alpha2"]),
            "realizations": int(point.params["AverageOverRunsN"]),
            "q10_source": point.q10_source,
        }
        for metric, (value, low, high, count) in point.metrics.items():
            row[metric] = value
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
            row[f"{metric}_n"] = count
        rows.append(row)
    return rows


def _paired_rows(
    points: list[FieldPoint], rng: np.random.Generator, bootstrap: int
) -> list[dict]:
    cases = _case_map(points)
    rows = []
    for geometry, ratios in EXPECTED_RATIOS.items():
        for ratio in ratios[1:]:
            if (geometry, "uniform", ratio) not in cases or (geometry, "random", ratio) not in cases:
                continue
            uniform = cases[(geometry, "uniform", ratio)]
            random = cases[(geometry, "random", ratio)]
            if uniform.params["run_seeds"] != random.params["run_seeds"]:
                raise ValueError(f"{geometry}, ratio={ratio}: uniform/random seeds are not paired")
            row = {"field_geometry": geometry, "field_strength_ratio": ratio}
            for metric in (*METRICS, "kappa_typ", "kappa_ann"):
                difference = random.per_run[metric] - uniform.per_run[metric]
                value, low, high, count = _bootstrap(difference, rng, bootstrap)
                row[f"delta_{metric}"] = value
                row[f"delta_{metric}_ci_low"] = low
                row[f"delta_{metric}_ci_high"] = high
                row[f"delta_{metric}_n"] = count
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _set_strength_axis(axis, ratios):
    axis.set_xscale("symlog", linthresh=0.25, linscale=0.8, base=10)
    axis.set_xticks(ratios)
    axis.set_xticklabels([f"{ratio:g}" for ratio in ratios])
    axis.set_xlabel(r"$h_{\rm rms}/g$", labelpad=2)
    axis.grid(alpha=0.2)


def _plot_metric_series(axis, members: list[FieldPoint], metric: str, label: str, color: str):
    x = np.asarray([point.params["field_strength_ratio"] for point in members], dtype=float)
    y = np.asarray([point.metrics[metric][0] for point in members])
    low = np.asarray([point.metrics[metric][1] for point in members])
    high = np.asarray([point.metrics[metric][2] for point in members])
    errors = np.maximum(np.vstack((y - low, high - y)), 0.0)
    axis.errorbar(
        x,
        y,
        yerr=errors,
        color=color,
        label=label,
        linewidth=1.5,
        marker="o",
        markersize=4.2,
        capsize=2.2,
    )
    # Open symbols mark estimates that rest on fewer realizations than were simulated
    # (a conditional statistic with an undefined value in some realizations).
    counts = np.asarray([point.metrics[metric][3] for point in members])
    totals = np.asarray([int(point.params["AverageOverRunsN"]) for point in members])
    partial = np.isfinite(y) & (counts < totals)
    if np.any(partial):
        axis.plot(
            x[partial],
            y[partial],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            zorder=4,
        )


def _save(fig, output_dir: Path, stem: str):
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=260 if suffix == "png" else None)
    plt.close(fig)


def _plot_aligned(points: list[FieldPoint], output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for profile, label in PROFILE_LEGEND:
            _plot_metric_series(
                axis,
                _series(points, "aligned", profile),
                metric,
                label,
                PROFILE_COLORS[profile],
            )
        _set_strength_axis(axis, EXPECTED_RATIOS["aligned"])
        axis.set_ylabel(LABELS[metric])
    axes[0, 0].axhline(0.9, color="0.45", linestyle=":", linewidth=0.9)
    axes[0, 1].axhline(0.1, color="0.45", linestyle=":", linewidth=0.9)
    axes[0, 0].set_ylim(-0.03, 1.05)
    axes[0, 1].set_ylim(-0.03, 0.3)
    axes[1, 1].set_ylim(-0.03, 1.05)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for label, axis in zip("abcd", axes.flat):
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
    fig.suptitle(r"Aligned recorder ($p=1/2$, $\theta=0$), $N_E=16$")
    _save(fig, output_dir, "field_profile_aligned_records")


def _plot_contrasts(rows: list[dict], output_dir: Path):
    selected = [row for row in rows if row["field_geometry"] == "aligned"]
    metrics = ("holevo_fhalf", "discord_fhalf", "holevo_fhalf_min")
    contrast_labels = (
        r"$\Delta[\chi_z(f=1/2)/H_Z(S)]$",
        r"$\Delta[D_z(f=1/2)/H_Z(S)]$",
        r"$\Delta$ late-window min of $q_{0.1}(\chi_z)/H_Z(S)$",
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for label, axis, metric, y_label in zip("abc", axes, metrics, contrast_labels):
        x = np.asarray([row["field_strength_ratio"] for row in selected])
        y = np.asarray([row[f"delta_{metric}"] for row in selected])
        low = np.asarray([row[f"delta_{metric}_ci_low"] for row in selected])
        high = np.asarray([row[f"delta_{metric}_ci_high"] for row in selected])
        axis.errorbar(
            x,
            y,
            yerr=np.maximum(np.vstack((y - low, high - y)), 0.0),
            color="#7B3294",
            marker="o",
            linewidth=1.4,
            capsize=2.2,
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set_xscale("log")
        axis.set_xticks(EXPECTED_RATIOS["aligned"][1:])
        axis.set_xticklabels([f"{x:g}" for x in EXPECTED_RATIOS["aligned"][1:]])
        axis.grid(alpha=0.2)
        axis.set_xlabel(r"$h_{\rm rms}/g$")
        axis.set_ylabel(y_label)
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
    fig.suptitle(r"Paired profile contrast: $\Delta=$ Gaussian $-$ uniform")
    _save(fig, output_dir, "field_profile_paired_contrast")


def _plot_longitudinal(points: list[FieldPoint], output_dir: Path):
    metrics = ("holevo_fhalf", "discord_fhalf", "holevo_fhalf_min")
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for label, axis, metric in zip("abc", axes, metrics):
        for profile, profile_label in PROFILE_LEGEND:
            _plot_metric_series(
                axis,
                _series(points, "longitudinal", profile),
                metric,
                profile_label,
                PROFILE_COLORS[profile],
            )
        _set_strength_axis(axis, EXPECTED_RATIOS["longitudinal"])
        axis.set_ylabel(LABELS[metric])
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
    axes[0].set_ylim(0.97, 1.003)
    axes[1].set_ylim(-0.0002, 0.0012)
    axes[2].set_ylim(0.93, 1.003)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(r"Commuting-axis control ($p=1/2$, $\theta=\pi/2$)")
    _save(fig, output_dir, "field_profile_longitudinal_control")


def _plot_fragment_curves(points: list[FieldPoint], output_dir: Path):
    cases = _case_map(points)
    ratios = (0.5, 30.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharey=True, constrained_layout=True)
    for label, axis, ratio in zip("ab", axes, ratios):
        for profile, profile_label in (("uniform", "uniform"), ("random", "Gaussian")):
            point = cases[("aligned", profile, ratio)]
            fraction = np.arange(1, 9) / 16.0
            color = PROFILE_COLORS[profile]
            axis.plot(
                fraction,
                np.nanmean(point.fragment_curves["holevo_mean"][1:9], axis=1),
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3.5,
                label=profile_label + r" mean $\chi_z$",
            )
            axis.plot(
                fraction,
                np.nanmean(point.fragment_curves["holevo_q10"][1:9], axis=1),
                color=color,
                linewidth=1.3,
                linestyle=":",
                label=profile_label + r" q$_{0.1}(\chi_z)$",
            )
            axis.plot(
                fraction,
                np.nanmean(point.fragment_curves["discord_mean"][1:9], axis=1),
                color=color,
                linewidth=1.3,
                linestyle="--",
                label=profile_label + r" mean $D_z$",
            )
        axis.axhline(0.9, color="0.45", linestyle=":", linewidth=0.8)
        axis.set_xlabel(r"fragment fraction $f=m/N_E$")
        axis.set_title(rf"$h_{{\rm rms}}/g={ratio:g}$")
        axis.grid(alpha=0.2)
        axis.text(0.02, 0.96, f"({label})", transform=axis.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel(r"late-window information / $H_Z(S)$")
    axes[0].set_ylim(-0.03, 1.05)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(r"Aligned-recorder fragment scaling ($p=1/2$, $\theta=0$)")
    _save(fig, output_dir, "field_profile_fragment_scaling")


def _fragment_rows(
    points: list[FieldPoint], rng: np.random.Generator, bootstrap: int
):
    rows = []
    for point in points:
        if point.params["field_geometry"] != "aligned":
            continue
        for size in range(1, 9):
            row = {
                "field_profile": point.params["field_profile"],
                "field_strength_ratio": point.params["field_strength_ratio"],
                "fragment_size": size,
                "fragment_fraction": size / 16.0,
            }
            for metric, values in point.fragment_curves.items():
                estimate, low, high, count = _bootstrap(values[size], rng, bootstrap)
                row[metric] = estimate
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                row[f"{metric}_n"] = count
            rows.append(row)
    return rows


def build(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.batch_dir / "batch_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dirs = sorted(path.parent for path in (args.batch_dir / "runs").glob("*/params.json"))
    complete = [run_dir for run_dir in run_dirs if (run_dir / "results.npz").is_file()]
    if manifest.get("status") != "completed" or manifest.get("combo_count") != 26 or len(complete) != 26:
        raise ValueError(
            f"batch is not the complete 26-case suite: status={manifest.get('status')!r}, "
            f"expected={manifest.get('combo_count')!r}, complete={len(complete)}"
        )
    window = tuple(float(value) for value in manifest["analysis"]["late_window"])
    sustained = int(manifest["analysis"]["record_sustained_samples"])
    rng = np.random.default_rng(args.seed)
    loaded = [_load_point(run_dir, window, rng, args.bootstrap, sustained) for run_dir in complete]
    points = [item[0] for item in loaded]
    health = [item[1] for item in loaded]
    cases = _case_map(points)
    expected_case_count = sum(1 + 2 * (len(ratios) - 1) for ratios in EXPECTED_RATIOS.values())
    if len(cases) != expected_case_count:
        raise ValueError(f"expected {expected_case_count} unique cases, found {len(cases)}")

    metric_rows = _metric_rows(points)
    paired_rows = _paired_rows(points, rng, args.bootstrap)
    fragment_rows = _fragment_rows(points, rng, args.bootstrap)
    _write_csv(args.output_dir / "field_profile_points.csv", metric_rows)
    _write_csv(args.output_dir / "field_profile_paired_contrasts.csv", paired_rows)
    _write_csv(args.output_dir / "field_profile_fragment_curves.csv", fragment_rows)
    _plot_aligned(points, args.output_dir)
    _plot_contrasts(paired_rows, args.output_dir)
    _plot_longitudinal(points, args.output_dir)
    _plot_fragment_curves(points, args.output_dir)

    longitudinal = [point for point in points if point.params["field_geometry"] == "longitudinal"]
    longitudinal_ranges = {
        metric: float(np.ptp([point.metrics[metric][0] for point in longitudinal]))
        for metric in METRICS
    }
    analysis_provenance = {}
    try:
        repo = Path(__file__).resolve().parents[1]
        analysis_provenance = {
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repo,
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
        "batch_dir": str(args.batch_dir.resolve()),
        "batch_status": manifest["status"],
        "expected_cases": manifest["combo_count"],
        "loaded_cases": len(points),
        "simulation_source_provenance": manifest.get("source_provenance", {}),
        "simulation_parameter_contract_matches_committed_suite": True,
        "analysis_code_provenance": analysis_provenance,
        "late_window": window,
        "late_window_samples": int(window[1] - window[0]) + 1,
        "central_estimator": "mean across realization-level late-window medians",
        "confidence_interval": f"95% percentile bootstrap over paired or unpaired realizations ({args.bootstrap} draws)",
        "bootstrap_scope": "realizations only; stored 64-fragment samples determine means and q10 thresholds but are not resampled",
        "normalization": "Holevo and discord divided by stored pointer entropy H_Z(S)",
        "redundancy": "N_E / smallest evaluated m with per-realization stored q10 Holevo >= 0.9 H_Z(S); late-window medians condition on times with a crossing; open plot symbols mark cases where fewer than all realizations have a defined value",
        "holevo_fhalf_min": "mean across realizations of the late-window minimum of q10 Holevo(f=1/2)/H_Z(S); threshold-free persistence measure replacing record occupancy",
        "record_rule": "q10 Holevo(f=1/2) >= 0.9 H_Z(S)",
        "pairing": "uniform and Gaussian cases at matched strength use identical ordered run_seeds",
        "max_norm_error": max(item["max_norm_error"] for item in health),
        "max_rho_trace_error": max(item["max_rho_trace_error"] for item in health),
        "max_rho_hermiticity_error": max(item["max_rho_hermiticity_error"] for item in health),
        "longitudinal_central_estimator_ranges": longitudinal_ranges,
        "expected_nonfinite_region": "fragment sizes m>8 were intentionally not evaluated because fragment_half_only=true",
    }
    (args.output_dir / "field_profile_analysis_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"loaded {len(points)}/26 complete cases")
    print(f"max norm error: {payload['max_norm_error']:.3e}")
    print("longitudinal ranges:")
    for metric, value in longitudinal_ranges.items():
        print(f"  {metric}: {value:.6g}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=48271)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
