import itertools
import json
import math
import os

import numpy as np

from eigensolver.level_stats import level_stats_for_params
from src.analysis.batch_analysis import parameter_label, select_axes
from src.batch_utils import normalize_sweep_values
from src.export.path_utils import get_data_dir


SPECTRAL_RESULTS_FILENAME = "spectral_gap_ratio.npz"


def _nan_grid(rows, cols):
    return [[math.nan for _ in range(cols)] for _ in range(rows)]


def _zero_grid(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]


def _bool_grid(rows, cols, value=False):
    return [[value for _ in range(cols)] for _ in range(rows)]


def _string_grid(rows, cols):
    return [["" for _ in range(cols)] for _ in range(rows)]


def _axis_index(values, value):
    for i, candidate in enumerate(values):
        try:
            if math.isclose(float(candidate), float(value), rel_tol=1e-12, abs_tol=1e-12):
                return i
        except (TypeError, ValueError):
            pass
        if candidate == value:
            return i
    raise ValueError(f"value {value!r} is not present in axis values {values!r}")


def _transpose(grid):
    return [list(row) for row in zip(*grid)]


def analyze_spectral_batch(
    manifest,
    x=None,
    y=None,
    sector="auto",
    trim_fraction=0.25,
    max_full_dim=None,
    dtype="auto",
    backend="cpu",
    real_tol=1e-12,
    tol=1e-10,
):
    manifest = dict(manifest)
    manifest["sweep"] = normalize_sweep_values(manifest["sweep"])
    x_axis, y_axis = select_axes(manifest, x=x, y=y)
    x_values = list(manifest["sweep"][x_axis])
    y_values = list(manifest["sweep"][y_axis]) if y_axis is not None else [None]

    rows = len(x_values)
    cols = len(y_values)
    results = {
        "mean_r": _nan_grid(rows, cols),
        "std_r": _nan_grid(rows, cols),
        "n_ratios": _zero_grid(rows, cols),
        "n_levels": _zero_grid(rows, cols),
        "sector_k": _nan_grid(rows, cols),
        "sector_sz": _nan_grid(rows, cols),
        "sector_dim": _zero_grid(rows, cols),
        "full_dim": _zero_grid(rows, cols),
        "is_real": _bool_grid(rows, cols),
        "eig_dtype": _string_grid(rows, cols),
        "backend": _string_grid(rows, cols),
        "conserved": _bool_grid(rows, cols),
        "bad": _bool_grid(rows, cols),
    }
    messages = [[None for _ in range(cols)] for _ in range(rows)]

    keys = list(manifest["sweep"].keys())
    value_lists = [manifest["sweep"][key] for key in keys]
    params_default = manifest.get("params_default", {})
    errors = []

    for combo in itertools.product(*value_lists):
        params = dict(params_default)
        params.update({key: value for key, value in zip(keys, combo)})
        x_idx = _axis_index(x_values, params[x_axis])
        y_idx = 0 if y_axis is None else _axis_index(y_values, params[y_axis])

        try:
            stats = level_stats_for_params(
                params,
                sector=sector,
                trim_fraction=trim_fraction,
                tol=tol,
                max_full_dim=max_full_dim,
                dtype=dtype,
                backend=backend,
                real_tol=real_tol,
            )
        except Exception as exc:
            results["bad"][x_idx][y_idx] = True
            messages[x_idx][y_idx] = str(exc)
            errors.append(
                {
                    "x": params[x_axis],
                    "y": None if y_axis is None else params[y_axis],
                    "error": str(exc),
                }
            )
            continue

        for metric_name in (
            "mean_r",
            "std_r",
            "n_ratios",
            "n_levels",
            "sector_dim",
            "full_dim",
            "is_real",
            "conserved",
        ):
            results[metric_name][x_idx][y_idx] = stats[metric_name]
        results["eig_dtype"][x_idx][y_idx] = stats["eig_dtype"]
        results["backend"][x_idx][y_idx] = stats["backend"]
        results["sector_k"][x_idx][y_idx] = math.nan if stats["sector_k"] is None else stats["sector_k"]
        results["sector_sz"][x_idx][y_idx] = math.nan if stats["sector_sz"] is None else stats["sector_sz"]
        messages[x_idx][y_idx] = stats["conservation_message"]

    return {
        "manifest": manifest,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": None if y_axis is None else y_values,
        "sector": sector,
        "trim_fraction": trim_fraction,
        "max_full_dim": max_full_dim,
        "dtype": dtype,
        "backend": backend,
        "real_tol": real_tol,
        "results": results,
        "messages": messages,
        "errors": errors,
    }


def _array(grid, dtype=float):
    return np.asarray(grid, dtype=dtype)


def save_spectral_results(batch_result):
    out_dir = get_data_dir(batch_result["manifest"].get("subdir", ""))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SPECTRAL_RESULTS_FILENAME)
    payload = {
        "x_values": np.asarray(batch_result["x_values"]),
        "y_values": np.asarray([] if batch_result["y_values"] is None else batch_result["y_values"]),
        "mean_r": _array(batch_result["results"]["mean_r"]),
        "std_r": _array(batch_result["results"]["std_r"]),
        "n_ratios": _array(batch_result["results"]["n_ratios"], dtype=int),
        "n_levels": _array(batch_result["results"]["n_levels"], dtype=int),
        "sector_k": _array(batch_result["results"]["sector_k"]),
        "sector_sz": _array(batch_result["results"]["sector_sz"]),
        "sector_dim": _array(batch_result["results"]["sector_dim"], dtype=int),
        "full_dim": _array(batch_result["results"]["full_dim"], dtype=int),
        "is_real": _array(batch_result["results"]["is_real"], dtype=bool),
        "eig_dtype": np.asarray(batch_result["results"]["eig_dtype"]),
        "backend": np.asarray(batch_result["results"]["backend"]),
        "conserved": _array(batch_result["results"]["conserved"], dtype=bool),
        "bad": _array(batch_result["results"]["bad"], dtype=bool),
        "metadata_json": json.dumps(
            {
                "subdir": batch_result["manifest"].get("subdir", ""),
                "x_axis": batch_result["x_axis"],
                "y_axis": batch_result["y_axis"],
                "sector": batch_result["sector"],
                "trim_fraction": batch_result["trim_fraction"],
                "max_full_dim": batch_result["max_full_dim"],
                "dtype": batch_result["dtype"],
                "backend": batch_result["backend"],
                "real_tol": batch_result["real_tol"],
                "errors": batch_result["errors"],
            },
            sort_keys=True,
        ),
    }
    np.savez_compressed(path, **payload)
    return path


def _figure_dir():
    out_dir = os.path.join("data", "figs", "analysis1")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _safe_part(value):
    return str(value).replace(os.sep, "_").replace(" ", "_")


def plot_spectral_results(batch_result, plot_pane=None):
    import matplotlib.pyplot as plt

    out_dir = _figure_dir()
    x_values = batch_result["x_values"]
    y_values = batch_result["y_values"]
    mean_r = batch_result["results"]["mean_r"]
    paths = []

    fig, ax = plt.subplots(figsize=(7, 4 if y_values is None else 5))
    if y_values is None:
        ax.plot(x_values, [row[0] for row in mean_r], marker="o")
        ax.set_ylabel("Mean adjacent gap ratio")
        suffix = f"1D_x-{_safe_part(batch_result['x_axis'])}"
    else:
        mesh = ax.pcolormesh(x_values, y_values, _transpose(mean_r), cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="Mean adjacent gap ratio")
        suffix = f"2D_x-{_safe_part(batch_result['x_axis'])}_y-{_safe_part(batch_result['y_axis'])}"
        ax.set_ylabel(parameter_label(batch_result["y_axis"]))

    ax.set_xlabel(parameter_label(batch_result["x_axis"]))
    ax.set_title(f"Spectral adjacent gap ratio ({batch_result['sector']}, trim={batch_result['trim_fraction']})")
    fig.tight_layout()

    fname = f"spectral_gap_ratio_{suffix}.jpg"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    if plot_pane is not None:
        plot_pane.push(fig)
    plt.close(fig)
    paths.append(path)
    return paths
