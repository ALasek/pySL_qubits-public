"""Plot a representative time-resolved Holevo plateau for Paper A."""

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
import numpy as np


BATCH_NAME = "Paper_A_final_HolevoPlateauFullFragments"
RUN_ID = "f2e1c686636f27ea143451a8a15ebdcafc4ce03c771c90983a40247234f4b001"
LAMBDA_TARGET = 0.75
SLICE_TIME = 36.0
THRESHOLD = 0.9
BOOTSTRAP_SEED = 4171


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pointer_entropy(results: np.lib.npyio.NpzFile, n_times: int) -> np.ndarray:
    rho = np.asarray(results["rhoS_T"])
    if rho.shape != (n_times, 2, 2):
        raise ValueError(f"unexpected rhoS_T shape {rho.shape}")
    probabilities = np.real(np.diagonal(rho, axis1=1, axis2=2)).copy()
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(probabilities > 0, -probabilities * np.log(probabilities), 0.0).sum(axis=1)


def bootstrap_mean(values: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimate = np.mean(values, axis=1)
    indices = rng.integers(0, values.shape[1], size=(draws, values.shape[1]))
    resampled = np.mean(values[:, indices], axis=2)
    low, high = np.quantile(resampled, [0.025, 0.975], axis=1)
    return estimate, low, high


def centers_to_edges(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    edges = np.empty(values.size + 1)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = lower
    edges[-1] = upper
    return edges


def validate_params(params: dict) -> None:
    expected = {
        "AverageOverRunsN": 24,
        "H_EE_J": 0.0,
        "H_E_J": 0.0,
        "H_SE_J": 0.1,
        "H_SE_Special": "Mironowicz_rand",
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
        "Mironowicz_theta": math.pi / 3.0,
        "fragment_half_only": False,
        "fragment_sample_count": 64,
        "lambda_geometry": "p_half",
        "lambda_target": LAMBDA_TARGET,
        "nqubits_E": 16,
        "nqubits_S": 1,
        "psi_bias": 0.5,
        "store_fragment_samples": True,
    }
    for key, target in expected.items():
        value = params.get(key)
        if isinstance(target, float):
            matches = math.isclose(float(value), target, abs_tol=1e-12)
        else:
            matches = value == target
        if not matches:
            raise ValueError(f"expected {key}={target!r}, got {value!r}")


def load_data(batch_dir: Path, slice_time: float):
    run_dir = batch_dir / "runs" / RUN_ID
    params_path = run_dir / "params.json"
    metadata_path = run_dir / "metadata.json"
    results_path = run_dir / "results.npz"
    for path in (params_path, metadata_path, results_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    params = json.loads(params_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_params(params)
    if metadata.get("status") != "completed" or metadata.get("run_id") != RUN_ID:
        raise ValueError("selected run is not a completed matching v2 result")

    with np.load(results_path, allow_pickle=False) as results:
        samples = np.asarray(results["Holevo_Z_S_Ef_fractionsT_samples"], dtype=float)
        sample_counts = np.asarray(results["Holevo_Z_S_Ef_fractionsT_Nsamples"], dtype=int)
        if samples.shape[:3] != (17, 41, 24) or samples.shape[3] != 64:
            raise ValueError(f"unexpected Holevo sample shape {samples.shape}")
        if sample_counts.shape != samples.shape[:3]:
            raise ValueError(f"unexpected sample-count shape {sample_counts.shape}")
        n_times = samples.shape[1]
        times = np.arange(n_times, dtype=float) * float(params["printT"])
        entropy = pointer_entropy(results, n_times)
        norm_drift = float(np.max(np.abs(np.asarray(results["norms"], dtype=float) - 1.0)))
        modified_norm_drift = float(np.max(np.abs(np.asarray(results["modified_norms"], dtype=float) - 1.0)))

    slice_index = int(np.argmin(np.abs(times - slice_time)))
    if not math.isclose(float(times[slice_index]), slice_time, abs_tol=1e-12):
        raise ValueError(f"requested time {slice_time:g} is not stored")

    direct_sizes = np.arange(1, params["nqubits_E"] + 1)
    expected_counts = np.minimum(
        [math.comb(params["nqubits_E"], int(size)) for size in direct_sizes],
        params["fragment_sample_count"],
    )
    for size, expected in zip(direct_sizes, expected_counts, strict=True):
        actual_finite = np.sum(np.isfinite(samples[size]), axis=2)
        if not np.all(actual_finite == expected) or not np.all(sample_counts[size] == expected):
            raise ValueError(f"incomplete direct fragment samples at m={size}")

    stored = samples[direct_sizes]
    run_fragment_means = np.nanmean(stored, axis=3)
    run_fragment_q10 = np.nanquantile(stored, 0.1, axis=3)
    surface_direct_count = params["nqubits_E"] // 2
    surface = np.nanmean(run_fragment_means[:surface_direct_count], axis=2) / entropy[None, :]

    slice_entropy = float(entropy[slice_index])
    mean_by_run = run_fragment_means[:, slice_index, :] / slice_entropy
    q10_by_run = run_fragment_q10[:, slice_index, :] / slice_entropy
    mean, mean_low, mean_high = bootstrap_mean(mean_by_run, 2000)
    q10, q10_low, q10_high = bootstrap_mean(q10_by_run, 2000)

    slice_sizes = np.insert(direct_sizes, 0, 0)
    surface_sizes = np.arange(surface_direct_count + 1)
    surface = np.vstack((np.zeros(n_times), surface))
    zero = np.zeros(1)
    mean = np.concatenate((zero, mean))
    mean_low = np.concatenate((zero, mean_low))
    mean_high = np.concatenate((zero, mean_high))
    q10 = np.concatenate((zero, q10))
    q10_low = np.concatenate((zero, q10_low))
    q10_high = np.concatenate((zero, q10_high))
    qualifying = np.flatnonzero(q10 >= THRESHOLD)
    threshold_size = int(slice_sizes[qualifying[0]]) if qualifying.size else None

    return {
        "batch_dir": batch_dir,
        "run_dir": run_dir,
        "params": params,
        "metadata": metadata,
        "paths": {
            "batch_manifest": batch_dir / "batch_manifest.json",
            "params": params_path,
            "metadata": metadata_path,
            "results": results_path,
        },
        "times": times,
        "surface_sizes": surface_sizes,
        "surface_fractions": surface_sizes / params["nqubits_E"],
        "slice_sizes": slice_sizes,
        "slice_fractions": slice_sizes / params["nqubits_E"],
        "slice_sample_counts": np.insert(expected_counts, 0, 0),
        "surface": surface,
        "slice_index": slice_index,
        "slice_entropy": slice_entropy,
        "mean": mean,
        "mean_low": mean_low,
        "mean_high": mean_high,
        "q10": q10,
        "q10_low": q10_low,
        "q10_high": q10_high,
        "threshold_size": threshold_size,
        "norm_drift": norm_drift,
        "modified_norm_drift": modified_norm_drift,
    }


def plot(data: dict, output_dir: Path, dpi: int) -> list[Path]:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 4.75),
        gridspec_kw={"height_ratios": (1.05, 1.0)},
        constrained_layout=True,
    )

    time_edges = centers_to_edges(data["times"], 0.0, float(data["times"][-1]))
    fraction_edges = centers_to_edges(data["surface_fractions"], 0.0, 0.5)
    image = axes[0].pcolormesh(
        time_edges,
        fraction_edges,
        data["surface"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=True,
    )
    axes[0].axvline(SLICE_TIME, color="white", linestyle="--", linewidth=1.0)
    axes[0].set(xlim=(0, 40), ylim=(0, 0.5), xlabel=r"time $t$", ylabel=r"fragment fraction $f=m/N_{\mathcal{E}}$")
    axes[0].set_title(r"Mean pointer information, $\Lambda=3/4$")
    colorbar = fig.colorbar(image, ax=axes[0], pad=0.02, fraction=0.075)
    colorbar.set_label(r"$\chi_z(\mathcal{S}{:}\mathcal{F})/H_Z(\mathcal{S})$")
    axes[0].text(0.02, 0.95, "(a)", transform=axes[0].transAxes, va="top", fontweight="bold")

    sizes = data["slice_sizes"]
    axes[1].fill_between(sizes, data["mean_low"], data["mean_high"], color="#2468a2", alpha=0.18, linewidth=0)
    axes[1].plot(sizes, data["mean"], color="#2468a2", marker="o", markersize=3, label="fragment mean")
    axes[1].fill_between(sizes, data["q10_low"], data["q10_high"], color="#d67b16", alpha=0.18, linewidth=0)
    axes[1].plot(sizes, data["q10"], color="#d67b16", marker="s", markersize=3, label="10th percentile")
    axes[1].axhline(THRESHOLD, color="0.25", linestyle="--", linewidth=1.0, label=r"$0.9H_Z$ threshold")
    if data["threshold_size"] is not None:
        size = data["threshold_size"]
        axes[1].axvline(size, color="#d67b16", linestyle=":", linewidth=1.0)
        axes[1].annotate(
            rf"$m_{{0.1}}^{{(q_{{0.1}})}}={size}$",
            xy=(size, data["q10"][size]),
            xytext=(-46, -25),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.25"},
        )
    axes[1].set(
        xlim=(0, 16),
        ylim=(0, 1.03),
        xticks=(0, 4, 8, 12, 16),
        xlabel=r"fragment size $m=|\mathcal{F}|$",
        ylabel=r"normalized $\chi_z$",
    )
    axes[1].set_title(rf"Fragment-size cut at $t={SLICE_TIME:g}$")
    axes[1].legend(loc="lower right", frameon=False)
    axes[1].text(0.02, 0.95, "(b)", transform=axes[1].transAxes, va="top", fontweight="bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"holevo_plateau_overview.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_tables(data: dict, output_dir: Path) -> list[Path]:
    surface_path = output_dir / "holevo_plateau_surface.csv"
    with surface_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("time", "fragment_size", "fragment_fraction", "mean_holevo_norm"))
        writer.writeheader()
        for size_index, size in enumerate(data["surface_sizes"]):
            for time_index, time in enumerate(data["times"]):
                writer.writerow(
                    {
                        "time": float(time),
                        "fragment_size": int(size),
                        "fragment_fraction": float(data["surface_fractions"][size_index]),
                        "mean_holevo_norm": float(data["surface"][size_index, time_index]),
                    }
                )

    slice_path = output_dir / "holevo_plateau_slice.csv"
    fields = (
        "time",
        "fragment_size",
        "fragment_fraction",
        "mean_holevo_norm",
        "mean_ci_low",
        "mean_ci_high",
        "q10_holevo_norm",
        "q10_ci_low",
        "q10_ci_high",
        "fragment_samples_per_realization",
        "crosses_q10_threshold",
    )
    with slice_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, size in enumerate(data["slice_sizes"]):
            writer.writerow(
                {
                    "time": SLICE_TIME,
                    "fragment_size": int(size),
                    "fragment_fraction": float(data["slice_fractions"][index]),
                    "mean_holevo_norm": float(data["mean"][index]),
                    "mean_ci_low": float(data["mean_low"][index]),
                    "mean_ci_high": float(data["mean_high"][index]),
                    "q10_holevo_norm": float(data["q10"][index]),
                    "q10_ci_low": float(data["q10_low"][index]),
                    "q10_ci_high": float(data["q10_high"][index]),
                    "fragment_samples_per_realization": int(data["slice_sample_counts"][index]),
                    "crosses_q10_threshold": bool(data["q10"][index] >= THRESHOLD),
                }
            )
    return [surface_path, slice_path]


def write_manifest(data: dict, outputs: list[Path], script_path: Path, output_dir: Path) -> Path:
    batch_manifest = json.loads(data["paths"]["batch_manifest"].read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(script_path.resolve()),
        "analysis_script_sha256": file_hash(script_path),
        "source_batch": BATCH_NAME,
        "source_batch_dir": str(data["batch_dir"].resolve()),
        "source_batch_manifest_sha256": file_hash(data["paths"]["batch_manifest"]),
        "simulation_source_provenance": batch_manifest.get("source_provenance"),
        "source_run_id": RUN_ID,
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": file_hash(path)}
            for name, path in data["paths"].items()
        },
        "parameters": {
            "n_environment": data["params"]["nqubits_E"],
            "p": data["params"]["psi_bias"],
            "theta": data["params"]["Mironowicz_theta"],
            "lambda": data["params"]["lambda_target"],
            "h0": data["params"]["Mironowicz_h0"],
            "W": data["params"]["Mironowicz_alpha2"],
            "H_EE_J": data["params"]["H_EE_J"],
            "realizations": data["params"]["AverageOverRunsN"],
            "fragment_samples_per_size": data["params"]["fragment_sample_count"],
            "time_step": data["params"]["dT"],
            "stored_time_step": data["params"]["printT"],
        },
        "aggregation": {
            "surface": "For m=1,...,8, average stored fragments within each realization, then average the 24 realization means.",
            "slice": "At t=36, calculate fragment means and q10 values directly for every m=1,...,16 within each realization, then average across realizations.",
            "confidence_intervals": "Percentile interval from 2000 bootstrap resamples of the 24 realization clusters; fragments are not treated as independent realizations.",
            "normalization": "Pointer-basis entropy H_Z(S) calculated from the stored reduced system state at each time.",
        },
        "fragment_sampling": {
            "direct_fragment_sizes": data["slice_sizes"][1:].tolist(),
            "samples_per_realization": {
                str(int(size)): int(count)
                for size, count in zip(data["slice_sizes"][1:], data["slice_sample_counts"][1:], strict=True)
            },
            "reconstruction_or_extrapolation": None,
        },
        "numerical_checks": {
            "maximum_norm_drift": data["norm_drift"],
            "maximum_modified_norm_drift": data["modified_norm_drift"],
            "all_declared_direct_samples_finite": True,
        },
        "provenance_caveat": "The batch records the exact source commit but marks the launch tree dirty; no launch-time diff snapshot accompanies the upload.",
        "slice_time": SLICE_TIME,
        "q10_threshold": THRESHOLD,
        "threshold_fragment_size": data["threshold_size"],
        "redundancy_estimate": data["params"]["nqubits_E"] / data["threshold_size"] if data["threshold_size"] else None,
        "outputs": {path.name: file_hash(path) for path in outputs},
    }
    path = output_dir / "holevo_plateau_overview_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=Path("data") / BATCH_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("generated") / "paper_a" / "holevo_plateau")
    parser.add_argument("--slice-time", type=float, default=SLICE_TIME)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    if batch_dir.name != BATCH_NAME:
        candidate = batch_dir / BATCH_NAME
        if candidate.is_dir():
            batch_dir = candidate
    if batch_dir.name != BATCH_NAME:
        raise ValueError(f"expected {BATCH_NAME} batch directory, got {batch_dir}")
    data = load_data(batch_dir, args.slice_time)
    outputs = plot(data, args.output_dir, args.dpi)
    tables = write_tables(data, args.output_dir)
    manifest = write_manifest(data, [*outputs, *tables], Path(__file__), args.output_dir)
    print(f"wrote {outputs[0]}")
    print(f"wrote {outputs[1]}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
