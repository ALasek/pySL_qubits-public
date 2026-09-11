#!/usr/bin/env python3
"""
Generate post-hoc figures for QD_summary single-run replacement JSONs.

This script is CPU-only: it reads data/nonbatch/runs/<run_id>/results.npz and
does not import pySL.py or CuPy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.run_store import RESULTS_FILENAME, compute_run_id
from src.input.param_expressions import resolve_parameter_expressions
from src.input.run_seeds import resolve_run_seed_plan_from_params
from src.input.validation import validate_params

DEFAULT_MANIFEST = REPO_ROOT / "qdsummary_replacements" / "plot_rebuild_manifest.json"
DEFAULT_OUT_ROOT = Path(get_data_dir("figs")) / "qdsummary_single_replacements"


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def single_entries(manifest, replacement_paths=None):
    wanted = set(replacement_paths or [])
    entries = []
    for entry in manifest["replacements"]:
        if entry["replacement_type"] != "run_params_json":
            continue
        if wanted and entry["replacement_path"] not in wanted:
            continue
        entries.append(entry)
    return entries


def load_source_params(replacement_path):
    path = REPO_ROOT / replacement_path
    params = json.loads(path.read_text(encoding="utf-8"))
    params = resolve_parameter_expressions(params)
    validate_params(params)
    stored = dict(params)
    stored.update(resolve_run_seed_plan_from_params(stored))
    return path, params, stored


def run_dir_for(entry):
    _, _params, stored = load_source_params(entry["replacement_path"])
    run_id = compute_run_id(stored)
    return Path(get_data_dir(DEFAULT_RUN_SUBDIR)) / "runs" / run_id


def scalar(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return value


def data_value(data, key, default=None):
    return data[key] if key in data.files else default


def run_payload(entry):
    params_path, source_params, stored = load_source_params(entry["replacement_path"])
    run_dir = run_dir_for(entry)
    results_path = run_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(f"missing results: {results_path}")

    params_path_in_run = run_dir / "params.json"
    params = json.loads(params_path_in_run.read_text(encoding="utf-8")) if params_path_in_run.exists() else stored
    data = np.load(results_path, allow_pickle=True)
    return {
        "entry": entry,
        "params_path": params_path,
        "source_params": source_params,
        "params": params,
        "run_dir": run_dir,
        "results_path": results_path,
        "data": data,
    }


def theta_suffix(params):
    theta = params.get("Mironowicz_theta", params.get("theta", None))
    if isinstance(theta, (int, float)):
        theta = f"{theta:.2f}"
    return theta


def plot_suffix(params):
    return f"_p={params.get('psi_bias', 0)}_theta={theta_suffix(params)}"


def time_axis(data, params, sample_count):
    T = scalar(data_value(data, "T"), params.get("T", sample_count - 1))
    return np.linspace(0, T, sample_count)


def fragment_axis(data, params, fragment_count):
    nqubits_e = int(scalar(data_value(data, "nqubits_E"), params.get("nqubits_E", fragment_count - 1)))
    return np.arange(fragment_count) / nqubits_e


def total_qubits(data, params):
    if "nqubits" in data.files:
        return int(scalar(data["nqubits"]))
    return int(params.get("nqubits_S", 1)) + int(params["nqubits_E"])


def save_figure(fig, filename, output_dirs):
    paths = []
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def max_redundancy_slope(ise, cutoff=0.5, min_time=0):
    slopes = []
    times = []
    polyfits = []
    xdata = np.linspace(0, 1, ise.shape[0])
    max_ise = float(np.max(ise))
    for it in range(ise.shape[1]):
        curve = ise[:, it]
        if max(curve) > cutoff * max_ise and it >= min_time:
            degree = min(6, max(1, ise.shape[0] - 1))
            coeffs = np.polyfit(xdata, curve, degree)
            derivative = np.polyder(np.poly1d(coeffs))
            slopes.append(abs(derivative(0.5)) / max(curve))
            times.append(it)
            polyfits.append(coeffs)
    if not slopes:
        return None
    best = int(np.argmin(slopes))
    return {
        "slope": float(slopes[best]),
        "time_index": int(times[best]),
        "coeffs": [float(value) for value in polyfits[best]],
    }


def generate_figures(payload, out_root, cutoff=0.5, min_time=0):
    entry = payload["entry"]
    params = payload["params"]
    data = payload["data"]
    run_dir = payload["run_dir"]

    ise = np.asarray(data["I_S_Ef_fractionsT_runAv"])
    ise_std = np.asarray(data["I_S_Ef_fractionsT_STD_runAv"]) if "I_S_Ef_fractionsT_STD_runAv" in data else None
    t_axis = time_axis(data, params, ise.shape[1])
    f_axis = fragment_axis(data, params, ise.shape[0])
    nqubits = total_qubits(data, params)
    suffix = plot_suffix(params)

    central_dir = out_root / Path(entry["replacement_path"]).stem
    run_fig_dir = run_dir / "figs"
    output_dirs = [central_dir, run_fig_dir]
    saved = []

    if "S_vn_runAv" in data:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(t_axis, np.asarray(data["S_vn_runAv"]))
        ax.set_title(f"Ent_VnS_N={nqubits}")
        ax.set_xlabel("T")
        ax.set_ylabel("Ent_VnS")
        saved.extend(save_figure(fig, f"Ent_VnS_N={nqubits}{suffix}.jpg", output_dirs))

    X, Y = np.meshgrid(t_axis, f_axis)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, ise, cmap="viridis", edgecolor="none")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title(f"ISE_3Dplot_N={nqubits}")
    ax.set_xlabel("T")
    ax.set_ylabel("Frac")
    ax.set_zlabel("I(S:E)")
    saved.extend(save_figure(fig, f"ISE_3Dplot_N={nqubits}{suffix}.jpg", output_dirs))

    slope = max_redundancy_slope(ise, cutoff=cutoff, min_time=min_time)
    if slope is not None:
        time_index = slope["time_index"]
        print_t = float(params.get("printT", 1))
        physical_t = time_index * print_t
        fig, ax = plt.subplots(figsize=(7, 4.5))
        yerr = ise_std[:, time_index] if ise_std is not None else None
        ax.errorbar(f_axis, ise[:, time_index], yerr=yerr, fmt="-o", capsize=5)
        poly = np.poly1d(slope["coeffs"])
        x_fit = np.linspace(0, 1, 200)
        ax.plot(x_fit, poly(x_fit), color="k")
        ax.set_title(f"ISEplat_T={physical_t:g}_N={nqubits}")
        ax.set_xlabel(f"Fraction of E at T={physical_t:g}")
        ax.set_ylabel("I(S:Ef)")
        saved.extend(save_figure(fig, f"ISEplat_T={physical_t:g}_N={nqubits}{suffix}.jpg", output_dirs))

    data.close()
    record = {
        "labels": entry.get("labels", []),
        "replacement_path": entry["replacement_path"],
        "run_dir": str(run_dir),
        "central_dir": str(central_dir),
        "saved": [str(path) for path in saved],
        "qd_slope": slope,
    }
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "figure_manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return record


def main():
    parser = argparse.ArgumentParser(description="Generate QD_summary single-run replacement figures.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Replacement manifest JSON.")
    parser.add_argument(
        "--replacement-path",
        action="append",
        help="Limit to a manifest replacement_path. Repeatable.",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Central figure output root.")
    parser.add_argument("--cutoff", type=float, default=0.5, help="QD-slope cutoff.")
    parser.add_argument("--min-time", type=int, default=0, help="Minimum printed-time index for plateau selection.")
    parser.add_argument("--dry-run", action="store_true", help="Report target run dirs without writing figures.")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    entries = single_entries(manifest, replacement_paths=args.replacement_path)
    if not entries:
        raise SystemExit("no matching single-run replacement entries")

    records = []
    for entry in entries:
        run_dir = run_dir_for(entry)
        results_path = run_dir / RESULTS_FILENAME
        if args.dry_run:
            print(f"{entry['replacement_path']} -> {run_dir} results_exists={results_path.exists()}")
            continue
        payload = run_payload(entry)
        record = generate_figures(
            payload,
            Path(args.out_root),
            cutoff=args.cutoff,
            min_time=args.min_time,
        )
        records.append(record)
        print(f"Generated {len(record['saved'])} files for {entry['replacement_path']}")

    if not args.dry_run:
        out_root = Path(args.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "single_figure_manifest.json").write_text(
            json.dumps(records, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
