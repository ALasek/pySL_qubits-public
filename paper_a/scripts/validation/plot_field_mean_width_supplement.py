import csv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import apply_style, save_figure

apply_style()


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "analysis/field_mean_width_exploration_2026_09_09/summary.csv"
OUTPUT = ROOT / "figures/paper_a/supplement/field_mean_width_comparison.pdf"


def main():
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.3), layout="constrained")
    data = np.load(SOURCE.with_name("realization_minima.npz"))
    labels = (r"$\chi_z(f=1/2)/H_Z(S)$", r"$D_z(f=1/2)/H_Z(S)$",
              r"conditional $R_{0.1}^{(q_{0.1})}$", r"Mean $\chi_{\min}$")
    rng = np.random.default_rng(20260909)
    colors = ("tab:blue", "tab:green", "tab:red", "tab:purple", "tab:orange")
    intervals = []
    for eta, color, style, marker in zip((0, .25, .5, .75, 1), colors,
                                        ("-", "--", "-.", ":", "-"), ("o", "s", "^", "D", "v")):
        selected = np.flatnonzero((data["etas"] == eta) | (data["ratios"] == 0))
        label = rf"$\eta={eta:g}$" + (" (uniform)" if eta == 0 else " (zero mean)" if eta == 1 else "")
        for panel, ax in enumerate(axes.flat):
            central, low, high, counts = [], [], [], []
            for index in selected:
                values = data["statistics"][index, panel]
                values = values[np.isfinite(values)]
                if len(values):
                    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                    lo, hi = np.quantile(draws, [.025, .975])
                    value = values.mean()
                else:
                    value = lo = hi = np.nan
                central.append(value); low.append(lo); high.append(hi); counts.append(len(values))
                intervals.append(dict(eta=eta, ratio=data["ratios"][index], panel=panel+1,
                                      mean=value, low=lo, high=hi, realizations=len(values)))
            central = np.array(central)
            x = data["ratios"][selected]
            ax.errorbar(x, central, yerr=np.maximum([central-low, high-central], 0),
                        color=color, marker=marker, linestyle=style, markersize=3.5, linewidth=1,
                        elinewidth=.7, capsize=1.5, label=label)
            partial = (np.array(counts) < 24) & np.isfinite(central)
            ax.plot(x[partial], central[partial], linestyle="none", marker=marker, markersize=3.5,
                    markerfacecolor="white", markeredgecolor=color, zorder=5)
    for index, (ax, label) in enumerate(zip(axes.flat, labels)):
        ax.set_xscale("symlog", linthresh=.25, linscale=1.3, base=10)
        ticks = [0, .1, .25, .5, 1, 2, 5, 10, 30]
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks], fontsize=8)
        ax.set(xlabel=r"$h_{\rm rms}/g$", ylabel=label)
        ax.text(.03, .95, f"({chr(97+index)})", transform=ax.transAxes, va="top")
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8, loc="lower right", ncol=2)
    for ax, threshold in ((axes[0,0], .9), (axes[0,1], .1)):
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1)
    axes[0,0].set_ylim(-.03, 1.05)
    axes[0,1].set_ylim(-.03, .3)
    axes[1,1].set_ylim(-.03, 1.05)
    with OUTPUT.with_name("supplement_intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=intervals[0])
        writer.writeheader(); writer.writerows(intervals)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
