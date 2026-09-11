"""
CLI README

Usage:
  python pySL_batchAnimate.py (--subdir SUBDIR | --config CONFIG) [options]

Arguments:
  --local-config FILE         Ignored JSON file for local animation defaults.
  --subdir SUBDIR             Data subdirectory under data/; loads its manifest.
  --config CONFIG             Python batch config module or .py path.
  --x, --y                    Sweep parameter names for plot axes. A 2D sweep is required.
  --setMinTime TIME           Minimum physical time for redundancy metrics. Default: 0.
  --redundancyminfraction X   Reserved for future metrics; saved in metadata.
  --metric METRIC             Animated metric: qd_slope or discord_fhalf.
  --fps FPS                   MP4 frame rate. Default: 6.
  --t-start TIME              First physical time to render.
  --t-stop TIME               Last physical time to render.
  --filter KEY=VALUE          Extra run-parameter filter. Repeatable.
  --analysis-name NAME        Output bundle folder under data/figs/<subdir>/.
  --old-data                  Load legacy old-data pickle layout.
  --force-recompute-cache     Recompute qd_slope_time_cube.npz even if it exists.
  --cache-only                Build cache/metadata, skip MP4 rendering.
  --render-only               Render from an existing cache. Requires --analysis-name.
"""

import argparse
import json
import os

from src.analysis.batch_animation import (
    ANIMATION_METRICS,
    build_time_cube,
    time_cube_filename,
    animation_bundle_dir,
    load_cached_time_cube,
    load_manifest_for_animation,
    render_time_cube_animation,
    save_time_cube,
    write_animation_metadata,
)
from src.analysis.local_config import load_local_config
from src.batch_utils import load_batch_config


DEFAULT_ANIMATION_CONFIG = {
    "config": None,
    "subdir": None,
    "x": None,
    "y": None,
    "setMinTime": 0,
    "redundancyminfraction": None,
    "metric": "qd_slope",
    "fps": 6,
    "t_start": None,
    "t_stop": None,
    "filters": {},
    "analysis_name": None,
    "dataOld": False,
    "force_recompute_cache": False,
    "cache_only": False,
    "render_only": False,
}


def _parse_filter(value):
    key, sep, raw = value.partition("=")
    key = key.strip()
    if not sep or not key:
        raise argparse.ArgumentTypeError("filters must use KEY=VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def main():
    parser = argparse.ArgumentParser(description="Animate a time-resolved pySL batch-analysis metric.")
    parser.add_argument(
        "--local-config",
        default="analysis_configs/batch_animate.local.json",
        help="Ignored JSON config file for local animation defaults.",
    )
    parser.add_argument("--subdir", help="Data subdirectory under data/.")
    parser.add_argument("--config", help="Python config module or .py path.")
    parser.add_argument("--x", help="Sweep parameter to use as the x axis.")
    parser.add_argument("--y", help="Sweep parameter to use as the y axis.")
    parser.add_argument("--setMinTime", type=float, help="Minimum physical time for redundancy metrics. Default: 0.")
    parser.add_argument(
        "--redundancyminfraction",
        type=float,
        help="Reserved for future metrics; saved in metadata.",
    )
    parser.add_argument("--metric", choices=ANIMATION_METRICS, default=None)
    parser.add_argument("--fps", type=int, help="MP4 frames per second.")
    parser.add_argument("--t-start", type=float, help="First physical time to render.")
    parser.add_argument("--t-stop", type=float, help="Last physical time to render.")
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        type=_parse_filter,
        metavar="KEY=VALUE",
        help="Extra run-parameter filter, for example --filter nqubits_E=15.",
    )
    parser.add_argument("--analysis-name", help="Output bundle folder name under data/figs/<subdir>/.")
    parser.add_argument("--old-data", action="store_true", default=None, help="Load legacy old-data pickle layout.")
    parser.add_argument(
        "--force-recompute-cache",
        action="store_true",
        default=None,
        help="Recompute the time-cube cache even if it exists.",
    )
    parser.add_argument("--cache-only", action="store_true", default=None, help="Build cache/metadata only.")
    parser.add_argument(
        "--render-only",
        action="store_true",
        default=None,
        help="Render from an existing cache. Requires --analysis-name.",
    )
    args = parser.parse_args()

    local_config = load_local_config(DEFAULT_ANIMATION_CONFIG, args.local_config, "batch animation")

    config_arg = args.config if args.config is not None else local_config["config"]
    subdir = args.subdir if args.subdir is not None else local_config["subdir"]
    x_axis = args.x if args.x is not None else local_config["x"]
    y_axis = args.y if args.y is not None else local_config["y"]
    set_min_time = args.setMinTime if args.setMinTime is not None else local_config["setMinTime"]
    redundancy_min_fraction = (
        args.redundancyminfraction
        if args.redundancyminfraction is not None
        else local_config["redundancyminfraction"]
    )
    metric = args.metric if args.metric is not None else local_config["metric"]
    fps = args.fps if args.fps is not None else local_config["fps"]
    t_start = args.t_start if args.t_start is not None else local_config["t_start"]
    t_stop = args.t_stop if args.t_stop is not None else local_config["t_stop"]
    old_data = args.old_data if args.old_data is not None else local_config["dataOld"]
    analysis_name = args.analysis_name if args.analysis_name is not None else local_config["analysis_name"]
    force_recompute_cache = (
        args.force_recompute_cache
        if args.force_recompute_cache is not None
        else local_config["force_recompute_cache"]
    )
    cache_only = args.cache_only if args.cache_only is not None else local_config["cache_only"]
    render_only = args.render_only if args.render_only is not None else local_config["render_only"]
    filters = dict(local_config.get("filters") or {})
    for key, value in args.filters or []:
        filters[key] = value

    if cache_only and render_only:
        parser.error("--cache-only and --render-only are mutually exclusive")
    if config_arg is None and subdir is None:
        parser.error("provide --subdir to load a manifest, or --config to analyze from a Python config")
    if render_only and analysis_name is None:
        parser.error("--render-only requires --analysis-name so the existing cache directory is unambiguous")

    config = load_batch_config(config_arg) if config_arg else None
    manifest = load_manifest_for_animation(subdir=subdir, config=config)

    seed_result = {
        "manifest": manifest,
        "x_axis": x_axis or manifest.get("analysis", {}).get("x"),
        "y_axis": y_axis or manifest.get("analysis", {}).get("y"),
        "x_values": [],
        "y_values": [],
        "Tsample": "time",
        "setMinTime": 0 if set_min_time is None else set_min_time,
        "redundancyminfraction": (
            redundancy_min_fraction
            if redundancy_min_fraction is not None
            else manifest.get("analysis", {}).get("redundancyminfraction", 0.4)
        ),
        "filters": filters,
        "old_data": old_data,
        "matched_runs": 0,
        "missing_runs": 0,
        "ambiguous_matches": 0,
        "matched_params": [],
    }
    out_dir = animation_bundle_dir(seed_result, analysis_name=analysis_name)
    cache_path = os.path.join(out_dir, time_cube_filename(metric))

    if render_only:
        result = load_cached_time_cube(out_dir)
    elif os.path.exists(cache_path) and not force_recompute_cache:
        result = load_cached_time_cube(out_dir)
        print(f"Loaded existing cache: {cache_path}")
    else:
        result = build_time_cube(
            manifest,
            x=x_axis,
            y=y_axis,
            set_min_time=set_min_time,
            redundancy_min_fraction=redundancy_min_fraction,
            filters=filters,
            old_data=old_data,
            metric=metric,
        )
        out_dir = animation_bundle_dir(result, analysis_name=analysis_name)
        cache_path = save_time_cube(result, out_dir)
        print(f"Saved cache: {cache_path}")

    rendered_path = None
    if not cache_only:
        rendered_path = render_time_cube_animation(
            result,
            out_dir,
            fps=fps,
            t_start=t_start,
            t_stop=t_stop,
        )
        print(f"Saved animation: {rendered_path}")

    write_animation_metadata(result, out_dir, rendered_path=rendered_path, t_start=t_start, t_stop=t_stop)
    print("Animation analysis complete")
    print(f"subdir = {manifest.get('subdir', '')}")
    print(f"x = {result['x_axis']}")
    print(f"y = {result['y_axis']}")
    print(f"metric = {result['metric']}")
    if filters:
        print(f"filters = {filters}")
    print(f"bundle = {out_dir}")


if __name__ == "__main__":
    main()
