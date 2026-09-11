import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import apply_style, save_figure

apply_style()
import numpy as np

from recalculate_exact import file_hash


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/supplement_polish_2026_09_09"
FIGURES = ROOT / "figures/paper_a/supplement"
STYLES = {
    "theta_zero": (r"$\theta=0$; vary $p$", "o", "#0072B2"),
    "p_half": (r"$p=1/2$; vary $\theta$", "s", "#D55E00"),
    "p_zero": (r"$p=0$; vary $\theta$", "^", "#009E73"),
    "diagonal": (r"$p=(2-\sqrt{2})/4$, $\theta=\pi/4$", "D", "#6F4E7C"),
}


def geometry(row):
    p, theta = float(row["psi_bias"]), float(row["theta"])
    if math.isclose(theta, 0):
        return "theta_zero"
    if math.isclose(p, .5):
        return "p_half"
    if math.isclose(p, 0):
        return "p_zero"
    assert math.isclose(theta, math.pi/4) and math.isclose(p, (2-math.sqrt(2))/4)
    return "diagonal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--point-csv", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with args.point_csv.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["batch"] == "Paper_A_final_SubmissionMatchedLambda"]
    assert len(rows) == 38
    samples, evidence = {}, []
    for row in rows:
        row["geometry"] = geometry(row)
        source = args.data_root/row["batch"]/"runs"/row["run_id"]
        p = json.loads((source/"params.json").read_text())
        with np.load(source/"results.npz", allow_pickle=False) as d:
            values = np.median(d["Holevo_Z_S_Ef_fractionsT"][8, 20:41], axis=0)/np.log(2)
        np.testing.assert_allclose(values.mean(), float(row["holevo_fhalf"]), atol=1e-12)
        key = (round(float(row["lambda"]), 12), row["geometry"])
        assert key not in samples
        samples[key] = (p["run_seeds"], values)
        evidence.append(dict(run_id=row["run_id"], params_sha256=file_hash(source/"params.json"),
                             results_sha256=file_hash(source/"results.npz")))
    rng = np.random.default_rng(20260909)
    indices = rng.integers(0, 24, size=(2000, 24))
    residuals = []
    for (lam, name), (seeds, values) in sorted(samples.items()):
        if name == "theta_zero":
            continue
        reference_seeds, reference = samples[(lam, "theta_zero")]
        assert seeds == reference_seeds
        difference = values-reference
        low, high = np.quantile(difference[indices].mean(axis=1), [.025, .975])
        residuals.append(dict(lambda_value=lam, geometry=name, mean=float(difference.mean()),
                              low=float(low), high=float(high), paired_values=difference.tolist()))
    (OUT/"s1_paired_differences.json").write_text(json.dumps(residuals, indent=2)+"\n")
    with (OUT/"s1_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(10.8, 8.2), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, .6))
    axes = [fig.add_subplot(grid[i//2, i%2]) for i in range(4)]
    residual_ax = fig.add_subplot(grid[2, :])
    metrics = ("holevo_fhalf", "discord_fhalf", "redundancy_q10", "holevo_fhalf_min")
    labels = (r"$\chi_z(f=1/2)/H_Z(S)$", r"$D_z(f=1/2)/H_Z(S)$",
              r"conditional $R_{0.1}^{(q_{0.1})}$", r"Mean $\chi_{\min}$")
    for ax, metric, label, letter in zip(axes, metrics, labels, "abcd"):
        for name, (legend, marker, color) in STYLES.items():
            selected = sorted([r for r in rows if r["geometry"] == name], key=lambda r:float(r["lambda"]))
            x = np.array([float(r["lambda"]) for r in selected])
            y = np.array([float(r[metric]) for r in selected])
            lo = np.array([float(r[metric+"_ci_low"]) for r in selected])
            hi = np.array([float(r[metric+"_ci_high"]) for r in selected])
            ax.errorbar(x, y, yerr=np.maximum([y-lo, hi-y], 0), marker=marker, mfc="none",
                        color=color, markersize=5, linewidth=.9, capsize=2, label=legend)
        ax.set(xlabel=r"$\Lambda$", ylabel=label, xlim=(-.025, 1.025))
        ax.text(.02, .96, f"({letter})", transform=ax.transAxes, va="top")
        ax.grid(alpha=.18)
    for name, (legend, marker, color) in STYLES.items():
        rs = [r for r in residuals if r["geometry"] == name]
        if not rs:
            continue
        y = np.array([r["mean"] for r in rs]); lo = np.array([r["low"] for r in rs]); hi=np.array([r["high"] for r in rs])
        residual_ax.errorbar([r["lambda_value"] for r in rs], y, yerr=np.maximum([y-lo, hi-y],0),
                            fmt=marker, color=color, mfc="none", ms=5, capsize=2)
    residual_ax.axhline(0, color="gray", lw=.8)
    residual_ax.set(xlabel=r"$\Lambda$", ylabel=r"Paired $\Delta\chi_z/H_Z(S)$", xlim=(-.025,1.025))
    residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    residual_ax.text(.02, .95, "(e)", transform=residual_ax.transAxes, va="top")
    residual_ax.grid(alpha=.18)
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc="outside upper center", ncol=2, frameon=False, fontsize=9)
    save_figure(fig, FIGURES/"matched_lambda_records.pdf")
    plt.close(fig)

    source = ROOT/"analysis/fragment_percentiles_2026_09_08/threshold_comparison.csv"
    with source.open() as stream:
        thresholds = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True,layout="constrained")
    for ax, lam, letter in zip(axes, (.75,.933012702,1.), "abc"):
        for criterion,color,marker,offset,label in zip(("mean","q25","q10"),("#2878b5","#d58a20","#af3951"),
                  ("s","D","o"),(-.2,0,.2),("Mean information","25th percentile","10th percentile")):
            rs = [r for r in thresholds if math.isclose(float(r["lambda"]),lam) and r["criterion"]==criterion]
            ax.plot([],[],color=color,marker=marker,ls="none",label=label)
            for r in rs:
                y=float(r["mean"]); x=float(r["n_environment"])+offset
                ax.errorbar(x,y,yerr=[[y-float(r["low"])] ,[float(r["high"])-y]],fmt=marker,color=color,
                            mfc=color if r["defined"]==r["total"] else "white",ms=5,capsize=2)
        ax.set(title=rf"$\Lambda={lam:.3g}$",xlabel=r"Environment size $N_E$",xticks=[8,12,16,20,24],ylim=(1.8,7.3))
        ax.text(.03,.97,f"({letter})",transform=ax.transAxes,va="top")
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Mean threshold fragment size")
    axes[0].legend(frameon=False,fontsize=10,loc="upper left",bbox_to_anchor=(.06,.92))
    save_figure(fig, FIGURES/"threshold_percentile_comparison.pdf")
    plt.close(fig)
    (OUT/"s1_s3_manifest.json").write_text(json.dumps(dict(source_csv=str(args.point_csv),
        source_csv_sha256=file_hash(args.point_csv), threshold_csv_sha256=file_hash(source),
        script_sha256=file_hash(Path(__file__)), bootstrap_seed=20260909, bootstrap_draws=2000,
        residual="Difference of realization-level time-median fragment means, relative to theta=0 at the same Lambda; paired realization resampling.",
        max_absolute_paired_mean=max(abs(r["mean"]) for r in residuals),sources=evidence),indent=2)+"\n")
    print("S1/S3 written; largest paired mean:",max(abs(r["mean"]) for r in residuals))


if __name__ == "__main__":
    main()
