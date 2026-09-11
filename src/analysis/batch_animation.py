import datetime
import json
import math
import os

import numpy as np

from src.analysis.batch_analysis import (
    _analysis_bundle_dir,
    _axis_values,
    _default_loader,
    _make_filter,
    _matched_params_summary,
    _psi_from_match,
    _write_json,
    fraction_metric_at,
    load_manifest_or_config,
    metric_label,
    parameter_label,
    select_axes,
)


ANIMATION_MANIFEST_FILENAME = "animation_manifest.json"

ANIMATION_METRICS = ("qd_slope", "discord_fhalf")


def time_cube_filename(metric):
    return f"{metric}_time_cube.npz"


def animation_filename(metric):
    return f"{metric}_time_animation.mp4"


def _scalar_float(value):
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr.item())
    return float(value)


def _sample_times(psi):
    values = np.asarray(psi.I_S_Ef_fractionsT_runAv)
    return np.arange(values.shape[1], dtype=float) * _scalar_float(psi.printT)


def _compatible_times(left, right):
    return left.shape == right.shape and np.allclose(left, right, rtol=0.0, atol=1e-9)


def _empty_cube(times, x_values, y_values):
    shape = (len(times), len(x_values), len(y_values))
    return np.full(shape, np.nan, dtype=float), np.ones(shape, dtype=bool)


def _metric_value_at_time(psi, metric, time_index, set_min_time):
    if metric == "qd_slope":
        min_time = set_min_time / _scalar_float(psi.printT)
        try:
            slope = psi.redundancy_slope_at_T(time_index, 0.5, minTime=min_time)
        except (IndexError, ValueError, FloatingPointError):
            return math.nan, True
        if slope is None or slope[0] == -1:
            return math.nan, True
        try:
            value = float(slope[0])
        except (TypeError, ValueError):
            return math.nan, True
    elif metric == "discord_fhalf":
        value = fraction_metric_at(
            psi,
            "Discord_Z_S_Ef_fractionsT_runAv",
            time_index,
            target_fraction=0.5,
        )
    else:
        raise ValueError(f"unsupported animation metric: {metric!r}")
    if not math.isfinite(value):
        return math.nan, True
    return value, False


def _animation_bundle_seed(manifest, x_axis, y_axis, x_values, y_values, set_min_time, redundancy_min_fraction, filters, old_data):
    return {
        "manifest": manifest,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": y_values,
        "Tsample": "time",
        "setMinTime": set_min_time,
        "redundancyminfraction": redundancy_min_fraction,
        "filters": filters or {},
        "old_data": old_data,
        "matched_runs": 0,
        "missing_runs": 0,
        "ambiguous_matches": 0,
        "matched_params": [],
    }


def build_time_cube(
    manifest,
    x=None,
    y=None,
    set_min_time=None,
    redundancy_min_fraction=None,
    filters=None,
    old_data=False,
    metric="qd_slope",
    loader=None,
):
    if metric not in ANIMATION_METRICS:
        raise ValueError(f"unsupported animation metric: {metric!r}")

    analysis = manifest.get("analysis", {})
    x_axis, y_axis = select_axes(manifest, x=x, y=y)
    if y_axis is None:
        raise ValueError("time-resolved batch animation currently requires a 2D sweep")

    set_min_time = 0 if set_min_time is None else set_min_time
    redundancy_min_fraction = (
        analysis.get("redundancyminfraction", 0.4)
        if redundancy_min_fraction is None
        else redundancy_min_fraction
    )
    loader = loader or _default_loader
    x_values = _axis_values(manifest, x_axis)
    y_values = _axis_values(manifest, y_axis)

    result = _animation_bundle_seed(
        manifest,
        x_axis,
        y_axis,
        x_values,
        y_values,
        set_min_time,
        redundancy_min_fraction,
        filters,
        old_data,
    )
    result["metric"] = metric

    cells = []
    times = None
    for xi, x_value in enumerate(x_values):
        for yi, y_value in enumerate(y_values):
            req = _make_filter(
                manifest,
                x_axis,
                x_value,
                y_axis=y_axis,
                y_value=y_value,
                filters=filters,
            )
            matches = loader(req, manifest.get("subdir", ""), old_data=old_data)
            if not matches:
                result["missing_runs"] += 1
                continue
            result["matched_runs"] += 1
            if len(matches) > 1:
                result["ambiguous_matches"] += 1
                print(f"[WARN] {len(matches)} matches for {req}; using the first")

            result["matched_params"].append(matches[0][0])
            psi = _psi_from_match(matches[0], old_data=old_data)
            psi_times = _sample_times(psi)
            if times is None:
                times = psi_times
            elif not _compatible_times(times, psi_times):
                raise ValueError(f"matched run has incompatible saved sample times for filter {req}")
            cells.append((xi, yi, psi))

    if times is None:
        raise ValueError("no matching completed runs were found; cannot build a time cube")

    cube, bad_mask = _empty_cube(times, x_values, y_values)
    for xi, yi, psi in cells:
        for time_index in range(len(times)):
            value, is_bad = _metric_value_at_time(psi, metric, time_index, set_min_time)
            cube[time_index, xi, yi] = value
            bad_mask[time_index, xi, yi] = is_bad

    result["times"] = times
    result["cube"] = cube
    result["bad_mask"] = bad_mask
    result["cache_filename"] = time_cube_filename(metric)
    result["animation_filename"] = animation_filename(metric)
    return result


def animation_bundle_dir(animation_result, analysis_name=None):
    return _analysis_bundle_dir(animation_result, analysis_name=analysis_name)


def _finite_color_limits(cube):
    finite = np.asarray(cube, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-12):
        pad = max(abs(vmin) * 0.05, 1e-9)
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _frame_indices(times, t_start=None, t_stop=None):
    mask = np.ones(len(times), dtype=bool)
    if t_start is not None:
        mask &= times >= t_start
    if t_stop is not None:
        mask &= times <= t_stop
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("selected time window contains no frames")
    return indices


def save_time_cube(animation_result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metric = animation_result["metric"]
    path = os.path.join(out_dir, time_cube_filename(metric))
    np.savez_compressed(
        path,
        times=np.asarray(animation_result["times"], dtype=float),
        metric_values=np.asarray(animation_result["cube"], dtype=float),
        bad_mask=np.asarray(animation_result["bad_mask"], dtype=bool),
        x_values=np.asarray(animation_result["x_values"], dtype=object),
        y_values=np.asarray(animation_result["y_values"], dtype=object),
        x_axis=np.asarray(animation_result["x_axis"]),
        y_axis=np.asarray(animation_result["y_axis"]),
        metric=np.asarray(animation_result["metric"]),
    )
    animation_result["cache_path"] = path
    return path


def load_cached_time_cube(out_dir):
    manifest_path = os.path.join(out_dir, ANIMATION_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"missing animation manifest: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        metadata = json.load(f)
    metric = metadata.get("metric", "qd_slope")
    cache_path = os.path.join(out_dir, time_cube_filename(metric))
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"missing time-cube cache: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as loaded:
        if "metric_values" in loaded:
            cube = loaded["metric_values"]
        else:
            cube = loaded["qd_slope"]
        result = {
            "manifest": _read_json_if_exists(os.path.join(out_dir, "batch_manifest.json")) or {"subdir": metadata["subdir"]},
            "x_axis": str(loaded["x_axis"].item()),
            "y_axis": str(loaded["y_axis"].item()),
            "x_values": loaded["x_values"].tolist(),
            "y_values": loaded["y_values"].tolist(),
            "metric": str(loaded["metric"].item()),
            "times": np.asarray(loaded["times"], dtype=float),
            "cube": np.asarray(cube, dtype=float),
            "bad_mask": np.asarray(loaded["bad_mask"], dtype=bool),
            "filters": metadata.get("filters", {}),
            "old_data": metadata.get("old_data", False),
            "setMinTime": metadata.get("setMinTime"),
            "redundancyminfraction": metadata.get("redundancyminfraction"),
            "matched_runs": metadata.get("matched_runs", 0),
            "missing_runs": metadata.get("missing_runs", 0),
            "ambiguous_matches": metadata.get("ambiguous_matches", 0),
            "matched_params": _read_json_if_exists(os.path.join(out_dir, "matched_run_params.json")) or [],
            "cache_path": cache_path,
            "cache_filename": time_cube_filename(metric),
            "animation_filename": animation_filename(metric),
        }
    return result


def _read_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_readme(animation_result, out_dir, rendered_path=None):
    lines = [
        f"# {animation_result['manifest'].get('subdir', '')}",
        "",
        "## Animation",
        f"- metric: {animation_result['metric']}",
        f"- x: {animation_result['x_axis']}",
        f"- y: {animation_result['y_axis']}",
        f"- setMinTime: {animation_result['setMinTime']}",
        f"- matched runs: {animation_result['matched_runs']}",
        f"- missing runs: {animation_result['missing_runs']}",
        f"- ambiguous matches: {animation_result['ambiguous_matches']}",
        f"- cache: {time_cube_filename(animation_result['metric'])}",
    ]
    if rendered_path:
        lines.append(f"- animation: {os.path.basename(rendered_path)}")
    lines.extend(
        [
            "",
            "Time-resolved metrics are computed at each fixed saved sample index.",
            "The qd_slope animation uses redundancy_slope_at_T(index) and never uses the best-time/max_redundancy_slope search used by static Tsample=-1 plots.",
        ]
    )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_animation_metadata(animation_result, out_dir, rendered_path=None, t_start=None, t_stop=None):
    times = np.asarray(animation_result["times"], dtype=float)
    frame_indices = _frame_indices(times, t_start=t_start, t_stop=t_stop)
    vmin, vmax = _finite_color_limits(animation_result["cube"])
    manifest = animation_result["manifest"]
    metadata = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "subdir": manifest.get("subdir", ""),
        "output_dir": out_dir,
        "x": animation_result["x_axis"],
        "y": animation_result["y_axis"],
        "metric": animation_result["metric"],
        "setMinTime": animation_result["setMinTime"],
        "redundancyminfraction": animation_result["redundancyminfraction"],
        "filters": animation_result.get("filters", {}),
        "old_data": animation_result.get("old_data", False),
        "frame_count": int(times.size),
        "rendered_frame_count": int(frame_indices.size),
        "physical_time_range": [float(times[0]), float(times[-1])],
        "rendered_time_range": [float(times[frame_indices[0]]), float(times[frame_indices[-1]])],
        "matched_runs": animation_result["matched_runs"],
        "missing_runs": animation_result["missing_runs"],
        "ambiguous_matches": animation_result["ambiguous_matches"],
        "color_scale": {"vmin": vmin, "vmax": vmax},
        "source_cache_filename": time_cube_filename(animation_result["metric"]),
        "animation_filename": os.path.basename(rendered_path) if rendered_path else None,
    }
    _write_json(os.path.join(out_dir, "batch_manifest.json"), manifest)
    _write_json(
        os.path.join(out_dir, "effective_config.json"),
        {
            "subdir": manifest.get("subdir", ""),
            "x": animation_result["x_axis"],
            "y": animation_result["y_axis"],
            "metric": animation_result["metric"],
            "setMinTime": animation_result["setMinTime"],
            "redundancyminfraction": animation_result["redundancyminfraction"],
            "filters": animation_result.get("filters", {}),
            "old_data": animation_result.get("old_data", False),
            "params_default": manifest.get("params_default", {}),
            "sweep": manifest.get("sweep", {}),
            "analysis": manifest.get("analysis", {}),
        },
    )
    _write_json(os.path.join(out_dir, ANIMATION_MANIFEST_FILENAME), metadata)
    _write_json(os.path.join(out_dir, "matched_run_params.json"), animation_result.get("matched_params", []))
    _write_json(
        os.path.join(out_dir, "matched_params_summary.json"),
        _matched_params_summary(animation_result.get("matched_params", [])),
    )
    _write_readme(animation_result, out_dir, rendered_path=rendered_path)
    return metadata


def render_time_cube_animation(animation_result, out_dir, fps=6, t_start=None, t_stop=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from matplotlib.colors import Normalize

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib cannot find an ffmpeg writer; cache was not rendered to mp4")

    os.makedirs(out_dir, exist_ok=True)
    times = np.asarray(animation_result["times"], dtype=float)
    cube = np.asarray(animation_result["cube"], dtype=float)
    bad_mask = np.asarray(animation_result["bad_mask"], dtype=bool)
    frame_indices = _frame_indices(times, t_start=t_start, t_stop=t_stop)
    x_values = np.asarray(animation_result["x_values"], dtype=float)
    y_values = np.asarray(animation_result["y_values"], dtype=float)
    vmin, vmax = _finite_color_limits(cube)

    path = os.path.join(out_dir, animation_filename(animation_result["metric"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=metric_label(animation_result["metric"]))

    writer_cls = animation.writers["ffmpeg"]
    writer = writer_cls(fps=fps, metadata={"artist": "pySL_qubits"})
    with writer.saving(fig, path, dpi=150):
        for frame_index in frame_indices:
            ax.clear()
            frame = cube[frame_index]
            bad = bad_mask[frame_index]
            ax.pcolormesh(
                x_values,
                y_values,
                frame.T,
                cmap="viridis",
                norm=norm,
                shading="auto",
            )
            bad_x = []
            bad_y = []
            for xi in range(len(x_values)):
                for yi in range(len(y_values)):
                    if bad[xi, yi]:
                        bad_x.append(x_values[xi])
                        bad_y.append(y_values[yi])
            if bad_x:
                ax.scatter(bad_x, bad_y, marker="x", color="red", s=60)
            ax.set_xlabel(parameter_label(animation_result["x_axis"]))
            ax.set_ylabel(parameter_label(animation_result["y_axis"]))
            ax.set_title(
                f"{metric_label(animation_result['metric'])}: "
                f"{parameter_label(animation_result['x_axis'])} vs "
                f"{parameter_label(animation_result['y_axis'])}\n"
                f"{animation_result['manifest'].get('subdir', '')}  T={times[frame_index]:.6g}"
            )
            ax.set_xlim(float(np.min(x_values)), float(np.max(x_values)))
            ax.set_ylim(float(np.min(y_values)), float(np.max(y_values)))
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    animation_result["animation_path"] = path
    return path


def load_manifest_for_animation(subdir=None, config=None):
    return load_manifest_or_config(subdir=subdir, config=config)
