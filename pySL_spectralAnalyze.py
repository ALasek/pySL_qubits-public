"""
CLI README

Usage:
  python pySL_spectralAnalyze.py --config CONFIG [options]

Arguments:
  --config CONFIG          Python batch config module or .py path.
  --x, --y                 Sweep parameter names for plot axes.
  --sector MODE            auto, middle, largest, full, or integer sector k.
                           auto uses the middle Sz sector when conserved and
                           full diagonalization otherwise.
  --trim-fraction X        Fraction of spectrum to drop from each edge before
                           computing adjacent gap ratios.
  --dtype DTYPE            auto, float32, float64, complex64, or complex128.
                           auto uses float32 for real H and complex64 for
                           complex H.
  --backend BACKEND        cpu, gpu, or auto.
  --max-full-dim N         Optional maximum dense full-Hilbert dimension allowed
                           when Sz is not conserved. Omit for no hard guard.
  --no-pane                Save figures without showing the plot pane.
  --no-plot                Save NPZ results only.
"""

import argparse

import matplotlib.pyplot as plt

from src.analysis.batch_analysis import load_manifest_or_config
from src.analysis.spectral_analysis import (
    analyze_spectral_batch,
    plot_spectral_results,
    save_spectral_results,
)
from src.batch_utils import load_batch_config


DEFAULT_CONFIG = "batch_configs.mironowicz_mbl_sweep_TEST_small"


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Compute spectral gap-ratio metrics over a pySL batch sweep.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Python config module or .py path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--x", help="Sweep parameter to use as the x axis.")
    parser.add_argument("--y", help="Sweep parameter to use as the y axis.")
    parser.add_argument(
        "--sector",
        default="auto",
        help="auto, middle, largest, full, or integer sector k.",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.25,
        help="Fraction of spectrum to drop from each edge before gap-ratio statistics.",
    )
    parser.add_argument(
        "--max-full-dim",
        type=int,
        default=None,
        help="Optional maximum dense full-Hilbert dimension allowed when Sz is not conserved.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float64", "complex64", "complex128"),
        help="Dense eigensolver dtype. auto uses float32 for real H and complex64 for complex H.",
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=("cpu", "gpu", "auto"),
        help="Dense eigensolver backend.",
    )
    parser.add_argument(
        "--no-pane",
        action="store_true",
        help="Do not show saved summary plots in the plot pane.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create summary plots; only save spectral_gap_ratio.npz.",
    )
    return parser.parse_args()


def _create_plot_pane(enabled):
    if not enabled:
        return None
    plt.switch_backend("Agg")
    from src.plot.plot_pane import PlotPane

    return PlotPane(title="Spectral Analysis Plots")


def main():
    args = _parse_cli_args()

    config = load_batch_config(args.config)
    manifest = load_manifest_or_config(config=config)
    result = analyze_spectral_batch(
        manifest,
        x=args.x,
        y=args.y,
        sector=args.sector,
        trim_fraction=args.trim_fraction,
        max_full_dim=args.max_full_dim,
        dtype=args.dtype,
        backend=args.backend,
    )
    result_path = save_spectral_results(result)

    plot_paths = []
    plot_pane = None
    if not args.no_plot:
        plot_pane = _create_plot_pane(not args.no_pane)
        plot_paths = plot_spectral_results(result, plot_pane=plot_pane)

    print("Spectral analysis complete")
    print(f"subdir = {result['manifest'].get('subdir', '')}")
    print(f"x = {result['x_axis']}")
    print(f"y = {result['y_axis']}")
    print(f"sector = {result['sector']}")
    print(f"trim_fraction = {result['trim_fraction']}")
    print(f"dtype = {result['dtype']}")
    print(f"backend = {result['backend']}")
    print(f"saved results = {result_path}")
    if result["errors"]:
        print(f"errors = {len(result['errors'])}")
        for item in result["errors"][:10]:
            print(f"  x={item['x']} y={item['y']}: {item['error']}")
    if plot_paths:
        print("Saved figures:")
        for path in plot_paths:
            print(f"  {path}")

    if plot_pane is not None:
        plot_pane.wait()


if __name__ == "__main__":
    main()
