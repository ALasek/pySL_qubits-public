import csv
import hashlib
import json
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import apply_style, save_figure
from plot_threshold_sizes import LAMBDAS


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures/paper_a"
OUTPUT = ROOT / "analysis/figure_typography_2026_09_09"


def main():
    apply_style()
    source = ROOT / "figures/paper_a/source_manifests/threshold_sizes_complete_2026_09_08/threshold_sizes_manifest.json"
    panel_a = ROOT / "figures/paper_a/source_manifests/fragment_threshold_scaling.csv"
    plotter = Path(__file__).with_name("plot_threshold_sizes.py")
    rows = json.loads(source.read_text())["groups"]
    args = SimpleNamespace(panel_a_csv=panel_a, output=FIGURES)
    # Reuse the maintained plotting block, bypassing its numerical calculations.
    code = plotter.read_text().split("    with args.panel_a_csv.open", 1)[1]
    code = "    with args.panel_a_csv.open" + code.split("    manifest =", 1)[0]
    exec(compile(dedent(code), str(plotter), "exec"), {**globals(), "args": args, "rows": rows})

    plt.rcdefaults()
    apply_style()
    curves = ROOT / "analysis/pointer_basis_expanded_2026_09_09/curves.csv"
    with curves.open() as stream:
        values = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for size, style in zip((1, 2, 4, 6, 8), ("-", "--", "-.", ":", "-")):
        selected = [r for r in values if int(r["size"]) == size]
        beta, gain, low, high = (np.array([float(r[key]) for r in selected])
                                 for key in ("beta", "gain", "low", "high"))
        line, = ax.plot(beta / np.pi, gain, linestyle=style, linewidth=1.6, label=f"$m={size}$")
        ax.fill_between(beta / np.pi, low, high, color=line.get_color(), alpha=.13, linewidth=0)
    ax.axhline(0, color="gray", ls=":", lw=.8)
    ax.set(xlim=(0, .5), ylim=(-1.03, .035), xlabel=r"System measurement angle $\beta/\pi$",
           ylabel=r"$\max_\phi\langle\chi_{\beta,\phi}\rangle-\langle\chi_z\rangle$ (bits)")
    ax.set_xticks([0, .125, .25, .375, .5])
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(alpha=.18)
    save_figure(fig, FIGURES / "pointer_basis_expanded.pdf")
    plt.close(fig)
    OUTPUT.mkdir(exist_ok=True)
    inputs = (source, panel_a, plotter, curves, Path(__file__))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUTPUT / "saved_sources.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
