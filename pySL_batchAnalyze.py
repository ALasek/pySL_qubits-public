"""
CLI README

Usage:
  python pySL_batchAnalyze.py (--subdir SUBDIR | --config CONFIG) [options]

Arguments:
  --local-config FILE         Ignored JSON file for local analysis defaults.
  --subdir SUBDIR             Data subdirectory under data/; loads its manifest.
  --config CONFIG             Python batch config module or .py path, useful
                              before a manifest exists.
  --x, --y                    Sweep parameter names for plot axes.
  --Tsample TIME              Sample time; use -1 for best-time scan.
  --setMinTime TIME           Minimum physical time for redundancy metrics.
  --redundancyminfraction X   Minimum redundancy fraction threshold.
  --metrics NAME [...]        Metrics to export, in display order.
  --filter KEY=VALUE          Extra run-parameter filter. Repeatable.
                              Values are parsed as JSON when possible.
  --analysis-name NAME        Optional output bundle folder name under
                              data/figs/<subdir>/.
  --old-data                  Load legacy old-data pickle layout.
  --plot-data BOOL            Compatibility flag; summary plots are saved
                              and shown in the plot pane.
  --no-pane                   Save figures without opening a Tk plot pane.
"""

import argparse
import json

from src.analysis.batch_analysis import analyze_batch, load_manifest_or_config, save_metric_plots
from src.analysis.local_config import load_local_config
from src.batch_utils import load_batch_config


DEFAULT_ANALYSIS_CONFIG = {
    "config": None,
    "subdir": None,
    "x": None,
    "y": None,
    "Tsample": None,
    "setMinTime": None,
    "redundancyminfraction": None,
    "holevo_delta": None,
    "holevo_quantile": None,
    "metrics": None,
    "filters": {},
    "analysis_name": None,
    "plotData": False,
    "dataOld": False,
}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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


def _create_plot_pane():
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    from src.plot.plot_pane import PlotPane

    return PlotPane(title="Batch Analysis Plots")


def main():
    parser = argparse.ArgumentParser(description="Analyze a pySL batch from its manifest.")
    parser.add_argument(
        "--local-config",
        default="analysis_configs/batch_analyze.local.json",
        help="Ignored JSON config file for local analysis defaults.",
    )
    parser.add_argument("--subdir", help="Data subdirectory under data/.")
    parser.add_argument(
        "--config",
        help="Python config module or .py path. Useful before a manifest exists.",
    )
    parser.add_argument("--x", help="Sweep parameter to use as the x axis.")
    parser.add_argument("--y", help="Sweep parameter to use as the y axis.")
    parser.add_argument("--Tsample", type=float, help="Sample time; use -1 for best-time scan.")
    parser.add_argument("--setMinTime", type=float, help="Minimum physical time for redundancy metrics.")
    parser.add_argument(
        "--redundancyminfraction",
        type=float,
        help="Minimum fraction threshold for redundancy score.",
    )
    parser.add_argument(
        "--holevo-delta",
        type=float,
        help="Information deficit delta used for Holevo redundancy and record thresholds.",
    )
    parser.add_argument(
        "--holevo-quantile",
        type=float,
        help="Stored fragment-information quantile used for R_delta,q.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Metrics to export, in display order.",
    )
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        type=_parse_filter,
        metavar="KEY=VALUE",
        help="Extra run-parameter filter, for example --filter nqubits_E=15.",
    )
    parser.add_argument(
        "--analysis-name",
        help="Output bundle folder name under data/figs/<subdir>/.",
    )
    parser.add_argument(
        "--old-data",
        action="store_true",
        default=None,
        help="Load legacy old-data pickle layout.",
    )
    parser.add_argument(
        "--plot-data",
        type=_parse_bool,
        default=None,
        help="Reserved compatibility flag; batch summary plots are always saved.",
    )
    parser.add_argument(
        "--no-pane",
        action="store_true",
        help="Save analysis figures without opening the Tk plot pane.",
    )
    args = parser.parse_args()

    local_config = load_local_config(DEFAULT_ANALYSIS_CONFIG, args.local_config, "batch analysis")

    config_arg = args.config if args.config is not None else local_config["config"]
    subdir = args.subdir if args.subdir is not None else local_config["subdir"]
    x_axis = args.x if args.x is not None else local_config["x"]
    y_axis = args.y if args.y is not None else local_config["y"]
    tsample = args.Tsample if args.Tsample is not None else local_config["Tsample"]
    set_min_time = args.setMinTime if args.setMinTime is not None else local_config["setMinTime"]
    redundancy_min_fraction = (
        args.redundancyminfraction
        if args.redundancyminfraction is not None
        else local_config["redundancyminfraction"]
    )
    old_data = args.old_data if args.old_data is not None else local_config["dataOld"]
    analysis_name = (
        args.analysis_name
        if args.analysis_name is not None
        else local_config["analysis_name"]
    )
    holevo_delta = args.holevo_delta if args.holevo_delta is not None else local_config["holevo_delta"]
    holevo_quantile = (
        args.holevo_quantile
        if args.holevo_quantile is not None
        else local_config["holevo_quantile"]
    )
    metrics = args.metrics if args.metrics is not None else local_config["metrics"]
    filters = dict(local_config.get("filters") or {})
    for key, value in args.filters or []:
        filters[key] = value

    if config_arg is None and subdir is None:
        parser.error("provide --subdir to load a manifest, or --config to analyze from a Python config")

    config = load_batch_config(config_arg) if config_arg else None
    manifest = load_manifest_or_config(subdir=subdir, config=config)
    if metrics is None:
        metrics = manifest.get("analysis", {}).get("metrics")

    result = analyze_batch(
        manifest,
        x=x_axis,
        y=y_axis,
        tsample=tsample,
        set_min_time=set_min_time,
        redundancy_min_fraction=redundancy_min_fraction,
        holevo_delta=holevo_delta,
        holevo_quantile=holevo_quantile,
        filters=filters,
        old_data=old_data,
    )
    plot_pane = None if args.no_pane else _create_plot_pane()
    paths = save_metric_plots(
        result,
        metrics=metrics,
        plot_pane=plot_pane,
        analysis_name=analysis_name,
    )

    print("Analysis complete")
    print(f"subdir = {result['manifest'].get('subdir', '')}")
    print(f"x = {result['x_axis']}")
    print(f"y = {result['y_axis']}")
    print(f"Tsample = {result['Tsample']}")
    if filters:
        print(f"filters = {filters}")
    print(f"bundle = {result['analysis_bundle_dir']}")
    print("Saved figures:")
    for path in paths:
        print(f"  {path}")
    if plot_pane is not None:
        plot_pane.wait()


if __name__ == "__main__":
    main()
