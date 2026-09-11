"""Audit Paper B mechanism controls from raw per-realization arrays."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBDIR = "Paper_B_MechanismDiscrimination_ZX_ExactRMS"
REFERENCE_CASE = "zx_quasiperiodic_chain"
THRESHOLD = 0.9 * np.log(2.0)


def _sem(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0


def _record_window(series, start_index, print_time):
    qualifies = np.isfinite(series) & (series >= THRESHOLD)
    qualifies[:start_index] = False
    indices = np.flatnonzero(qualifies)
    if indices.size == 0:
        return 0.0, 0.0
    end = int(indices[0])
    while end + 1 < qualifies.size and qualifies[end + 1]:
        end += 1
    return (end - int(indices[0])) * print_time, float(np.mean(qualifies[start_index:]))


def _redundancies(curves, n_environment):
    values = []
    for realization in range(curves.shape[-1]):
        crossing = next(
            (
                fragment_size
                for fragment_size in range(1, n_environment // 2 + 1)
                if np.isfinite(curves[fragment_size, realization])
                and curves[fragment_size, realization] >= THRESHOLD
            ),
            None,
        )
        values.append(None if crossing is None else n_environment / crossing)
    return values


def _load_rows(data_dir, run_index):
    rows = []
    for line in run_index.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        params = record["params"]
        n_environment = int(params["nqubits_E"])
        print_time = float(params["printT"])
        start_index = int(math.ceil(20.0 / print_time))
        result_path = data_dir / record["paths"]["results"]
        with np.load(result_path, allow_pickle=False) as result:
            levels = np.asarray(result["fragment_quantile_levels"], dtype=float)
            quantile_index = int(np.argmin(np.abs(levels - 0.1)))
            quantile_curves = result["Holevo_Z_S_Ef_fractionsT_quantiles"][
                quantile_index, :, -1, :
            ]
            half_holevo = np.asarray(
                result["Holevo_Z_S_Ef_fractionsT"][n_environment // 2, -1, :],
                dtype=float,
            )
            half_discord = np.asarray(
                result["Discord_Z_S_Ef_fractionsT"][n_environment // 2, -1, :],
                dtype=float,
            )
            half_holevo_series = np.asarray(
                result["Holevo_Z_S_Ef_fractionsT"][n_environment // 2, :, :],
                dtype=float,
            )
            lifetimes, occupancies = zip(
                *(
                    _record_window(
                        half_holevo_series[:, realization],
                        start_index,
                        print_time,
                    )
                    for realization in range(half_holevo_series.shape[-1])
                )
            )
            site_mean_trace_distance = np.asarray(
                result["SBS_trace_dist_1"][:, -1], dtype=float
            )
            gaps = np.asarray(result["branch_energy_density_gap"], dtype=float)
            widths = np.asarray(result["branch_energy_density_std"], dtype=float)
            gap_ratio = np.divide(
                gaps,
                np.max(widths, axis=0),
                out=np.full_like(gaps, np.nan),
                where=np.max(widths, axis=0) > 0,
            )
            redundancies = _redundancies(quantile_curves, n_environment)
            finite_redundancies = [value for value in redundancies if value is not None]
            rows.append(
                {
                    "run_id": record["run_id"],
                    "run_seeds": record["metadata"]["run_seeds"],
                    "case": params["PaperB_case"],
                    "n_environment": n_environment,
                    "field_strength": float(params["PaperB_field_strength"]),
                    "field_to_coupling_ratio": float(params["PaperB_field_strength"])
                    / float(params["H_EE_J"]),
                    "redundancy_per_realization": redundancies,
                    "redundancy_crossings": len(finite_redundancies),
                    "redundancy_mean_if_all_cross": (
                        float(np.mean(finite_redundancies))
                        if len(finite_redundancies) == len(redundancies)
                        else None
                    ),
                    "redundancy_sem_if_all_cross": (
                        _sem(finite_redundancies)
                        if len(finite_redundancies) == len(redundancies)
                        else None
                    ),
                    "half_holevo_per_realization": half_holevo.tolist(),
                    "half_holevo_mean": float(np.mean(half_holevo)),
                    "half_holevo_sem": _sem(half_holevo),
                    "half_discord_per_realization": half_discord.tolist(),
                    "half_discord_mean": float(np.mean(half_discord)),
                    "half_discord_sem": _sem(half_discord),
                    "record_lifetime_per_realization": list(lifetimes),
                    "record_lifetime_mean": float(np.mean(lifetimes)),
                    "record_occupancy_per_realization": list(occupancies),
                    "record_occupancy_mean": float(np.mean(occupancies)),
                    "single_site_mean_trace_distance_q10": float(
                        np.nanquantile(site_mean_trace_distance, 0.1)
                    ),
                    "max_branch_energy_gap_to_width": float(np.nanmax(gap_ratio)),
                    "max_exported_norm_drift": float(
                        np.max(np.abs(result["norms"] - 1.0))
                    ),
                    "max_exported_modified_norm_drift": float(
                        np.max(np.abs(result["modified_norms"] - 1.0))
                    ),
                }
            )
    return rows


def _paired_comparisons(rows):
    by_key = {
        (row["field_strength"], row["n_environment"], row["case"]): row for row in rows
    }
    comparisons = []
    for row in rows:
        if row["case"] == REFERENCE_CASE:
            continue
        reference = by_key[(row["field_strength"], row["n_environment"], REFERENCE_CASE)]
        if row["run_seeds"] != reference["run_seeds"]:
            raise ValueError(f"unpaired run seeds for {row['run_id']}")
        comparisons.append(
            {
                "case": row["case"],
                "n_environment": row["n_environment"],
                "field_to_coupling_ratio": row["field_to_coupling_ratio"],
                "delta_half_holevo_per_realization": (
                    np.asarray(row["half_holevo_per_realization"])
                    - np.asarray(reference["half_holevo_per_realization"])
                ).tolist(),
                "delta_half_holevo_mean": row["half_holevo_mean"]
                - reference["half_holevo_mean"],
                "delta_half_discord_mean": row["half_discord_mean"]
                - reference["half_discord_mean"],
                "delta_record_occupancy_mean": row["record_occupancy_mean"]
                - reference["record_occupancy_mean"],
            }
        )
    return comparisons


def _format_redundancies(values):
    return ", ".join("<2" if value is None else f"{value:.2f}" for value in values)


def _markdown(audit):
    lines = [
        "# Paper B mechanism-discrimination audit",
        "",
        f"- generated: {audit['created_at']}",
        f"- completed combinations: {audit['completed_combinations']}",
        f"- robust threshold: 0.9 ln(2) = {audit['threshold']:.6f}",
        "- censored realizations are shown as `<2` and are not averaged away",
        "",
        "| W/K | N_E | case | crossings | R_delta,q by realization | Holevo F=1/2 | discord F=1/2 | occupancy | lifetime | site q10 of run-mean TD |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        audit["rows"],
        key=lambda item: (
            item["field_to_coupling_ratio"],
            item["n_environment"],
            item["case"],
        ),
    ):
        lines.append(
            f"| {row['field_to_coupling_ratio']:.0f} | {row['n_environment']} | "
            f"`{row['case']}` | {row['redundancy_crossings']}/5 | "
            f"{_format_redundancies(row['redundancy_per_realization'])} | "
            f"{row['half_holevo_mean']:.4f} +/- {row['half_holevo_sem']:.4f} | "
            f"{row['half_discord_mean']:.4f} +/- {row['half_discord_sem']:.4f} | "
            f"{row['record_occupancy_mean']:.3f} | {row['record_lifetime_mean']:.1f} | "
            f"{row['single_site_mean_trace_distance_q10']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired differences from ordered quasiperiodic fields",
            "",
            "| W/K | N_E | control | Delta Holevo | Delta discord | Delta occupancy |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        audit["paired_comparisons"],
        key=lambda item: (
            item["field_to_coupling_ratio"],
            item["n_environment"],
            item["case"],
        ),
    ):
        lines.append(
            f"| {row['field_to_coupling_ratio']:.0f} | {row['n_environment']} | "
            f"`{row['case']}` | {row['delta_half_holevo_mean']:+.4f} | "
            f"{row['delta_half_discord_mean']:+.4f} | "
            f"{row['delta_record_occupancy_mean']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", default=DEFAULT_SUBDIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data" / args.subdir
    manifest = json.loads((data_dir / "batch_manifest.json").read_text(encoding="utf-8"))
    rows = _load_rows(data_dir, data_dir / "run_index.jsonl")
    keys = {
        (row["field_strength"], row["n_environment"], row["case"]) for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("duplicate parameter combinations in run index")
    expected = int(manifest["combo_count"])
    if len(rows) != expected and not args.allow_incomplete:
        raise ValueError(f"expected {expected} completed combinations, found {len(rows)}")

    audit = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subdir": args.subdir,
        "threshold": float(THRESHOLD),
        "expected_combinations": expected,
        "completed_combinations": len(rows),
        "rows": rows,
        "paired_comparisons": _paired_comparisons(rows) if len(rows) == expected else [],
    }
    output_dir = args.output_dir or REPO_ROOT / "data" / "figs" / args.subdir / "mechanism_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "audit.md").write_text(_markdown(audit), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
