import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts/validation"), str(ROOT / "scripts"), str(ROOT.parent)]

from paper_plot_style import apply_style
from replot_main_typography import styled_saves
import plot_holevo_plateau_overview as plateau
import plot_matched_lambda_fragment_sweep as matched
from analysis_scripts import paper_a_field_profile_control as field
from analysis_scripts import paper_a_submission_panels as stability
import plot_saved_s1_s3
import polish_s2_s4
import plot_field_mean_width_supplement as mean_width
import replot_saved_typography


TABLES = ROOT / "figures/paper_a/source_manifests/exact_2026_09_05/plots"


def read_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def plot_main(output):
    surface = read_rows(TABLES / "plateau/holevo_plateau_surface.csv")
    slices = read_rows(TABLES / "plateau/holevo_plateau_slice.csv")
    sizes = sorted({int(r["fragment_size"]) for r in surface})
    times = sorted({r["time"] for r in surface})
    lookup = {(int(r["fragment_size"]), r["time"]): r["mean_holevo_norm"] for r in surface}
    data = dict(times=np.array(times), surface_fractions=np.array(sizes)/16,
                surface=np.array([[lookup[m, t] for t in times] for m in sizes]),
                slice_sizes=np.array([r["fragment_size"] for r in slices]),
                threshold_size=next(int(r["fragment_size"]) for r in slices if r["q10_holevo_norm"] >= .9))
    for key, column in {"mean": "mean_holevo_norm", "mean_low": "mean_ci_low",
                        "mean_high": "mean_ci_high", "q10": "q10_holevo_norm",
                        "q10_low": "q10_ci_low", "q10_high": "q10_ci_high"}.items():
        data[key] = np.array([r[column] for r in slices])
    with styled_saves():
        plateau.plot(data, output, 300)
        matched.plot(read_rows(TABLES / "matched_sweep/matched_lambda_fragment_sweep.csv"), output, 300)

    points = []
    curves = read_rows(TABLES / "field/field_profile_fragment_curves.csv")
    for row in read_rows(TABLES / "field/field_profile_points.csv"):
        params = {key: row[key] for key in ("field_geometry", "field_profile", "field_strength_ratio")}
        params["AverageOverRunsN"] = int(row["realizations"])
        metrics = {key: tuple(row[key + suffix] for suffix in ("", "_ci_low", "_ci_high", "_n"))
                   for key in field.METRICS}
        fragment_curves = {}
        for key in ("holevo_mean", "holevo_q10", "discord_mean"):
            values = np.full((9, 1), np.nan)
            for r in curves:
                if r["field_profile"] == row["field_profile"] and r["field_strength_ratio"] == row["field_strength_ratio"]:
                    values[int(r["fragment_size"]), 0] = r[key]
            fragment_curves[key] = values
        points.append(SimpleNamespace(params=params, metrics=metrics, fragment_curves=fragment_curves))
    with styled_saves():
        field._plot_aligned(points, output)
        field._plot_fragment_curves(points, output)
    for old, new in (("field_profile_aligned_records", "aligned_field_profiles"),
                     ("field_profile_fragment_scaling", "fragment_scaling")):
        (output / f"{old}.pdf").replace(output / f"{new}.pdf")

    points = [SimpleNamespace(params={"Mironowicz_theta": math.pi/4, "H_SE_J": .1,
                                       "Mironowicz_alpha2": r["field_width"], "psi_bias": r["p"]},
                              metrics={key: (r[key],) for key in stability.METRICS})
              for r in read_rows(TABLES / "submission/stability_theta_0.250.csv")]
    with styled_saves():
        stability._plot_stability(points, output)
    (output / "stability_theta_0.250.pdf").replace(output / "stability_theta_pi4.pdf")


def main():
    parser = argparse.ArgumentParser(description="Redraw all Paper A plots from bundled processed data; no GPU or raw trajectories.")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    supplement = output / "supplement"
    supplement.mkdir(exist_ok=True)
    apply_style()
    plot_main(output)
    replot_saved_typography.FIGURES = output
    replot_saved_typography.OUTPUT = output
    with styled_saves():
        replot_saved_typography.main()
    plt.rcdefaults()
    apply_style()
    plot_saved_s1_s3.FIGURES = supplement
    plot_saved_s1_s3.main()
    plt.rcParams.update({"font.size": 9})
    polish_s2_s4.plot_s2(polish_s2_s4.load_csv(polish_s2_s4.S2_SOURCE), supplement / "higher_n_validation.pdf")
    polish_s2_s4.plot_s4(polish_s2_s4.load_csv(polish_s2_s4.S4_SOURCE), supplement / "stability_endpoints_compact.pdf")
    mean_width.OUTPUT = supplement / "field_mean_width_comparison.pdf"
    mean_width.main()
    # Original plotters expect preview filenames; their temporary placeholders are not outputs.
    for stem in ("holevo_plateau_overview", "matched_lambda_fragment_sweep",
                 "field_profile_aligned_records", "field_profile_fragment_scaling",
                 "stability_theta_0.250", "fragment_threshold_scaling"):
        (output / f"{stem}.png").unlink(missing_ok=True)
    figures = sorted(output.rglob("*.pdf"))
    assert len(figures) == 12, figures
    report = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in figures}
    (output / "figure_hashes.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(figures)} plots to {output}. Figure 1 is provided as schematic.tex.")


if __name__ == "__main__":
    main()
