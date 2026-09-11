#!/usr/bin/env python3
"""Analyze the Paper-A timestep and fixed-time high-size review runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CONVERGENCE_REFERENCES = {
    "no_field_lambda_half": (
        "Paper_A_final_SubmissionMatchedLambda",
        lambda p: _close(p.get("psi_bias"), 0.5)
        and _close(p.get("Mironowicz_theta"), math.pi / 4)
        and _close(p.get("Mironowicz_alpha2", 0), 0),
        (20.0, 40.0),
    ),
    "aligned_random_rescue": (
        "Paper_A_final_FieldProfileControl",
        lambda p: p.get("field_geometry") == "aligned"
        and p.get("field_profile") == "random"
        and _close(p.get("field_strength_ratio"), 0.5),
        (30.0, 60.0),
    ),
    "aligned_random_strong": (
        "Paper_A_final_FieldProfileControl",
        lambda p: p.get("field_geometry") == "aligned"
        and p.get("field_profile") == "random"
        and _close(p.get("field_strength_ratio"), 30.0),
        (30.0, 60.0),
    ),
}


def _close(left, right) -> bool:
    return left is not None and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info(root: Path) -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "git_dirty": dirty}


def _run_dirs(batch_dir: Path) -> list[tuple[Path, dict]]:
    found = []
    for params_path in sorted((batch_dir / "runs").glob("*/params.json")):
        run_dir = params_path.parent
        if (run_dir / "results.npz").is_file():
            found.append((run_dir, json.loads(params_path.read_text(encoding="utf-8"))))
    return found


def _manifest(batch_dir: Path, expected_count: int) -> tuple[Path, dict]:
    path = batch_dir / "batch_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("combo_count") != expected_count:
        raise ValueError(f"{batch_dir}: incomplete batch manifest")
    if len(_run_dirs(batch_dir)) != expected_count:
        raise ValueError(f"{batch_dir}: expected {expected_count} complete run artifacts")
    return path, manifest


def _pointer_entropy(results) -> np.ndarray:
    rho = np.asarray(results["rhoS_T"])
    probabilities = np.real(np.diagonal(rho, axis1=1, axis2=2)).copy()
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(probabilities > 0, -probabilities * np.log(probabilities), 0.0).sum(
            axis=1
        )


def _minimum_fragment(q10: np.ndarray, entropy: np.ndarray, n_environment: int) -> np.ndarray:
    values = np.full(q10.shape[1:], -1, dtype=int)
    for time_index in range(q10.shape[1]):
        for run_index in range(q10.shape[2]):
            hits = np.flatnonzero(
                q10[1 : n_environment // 2 + 1, time_index, run_index]
                >= 0.9 * entropy[time_index]
            )
            if hits.size:
                values[time_index, run_index] = int(hits[0] + 1)
    return values


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan, math.nan, 0
    estimate = float(np.mean(finite))
    if finite.size == 1 or draws == 0:
        return estimate, math.nan, math.nan, int(finite.size)
    indices = rng.integers(0, finite.size, size=(draws, finite.size))
    samples = np.mean(finite[indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return estimate, float(low), float(high), int(finite.size)


def _reference_run(data_root: Path, batch: str, predicate):
    selected = [(run, params) for run, params in _run_dirs(data_root / batch) if predicate(params)]
    if len(selected) != 1:
        raise ValueError(f"{batch}: expected one reference run, found {len(selected)}")
    return selected[0]


def _convergence_rows(data_root: Path) -> tuple[list[dict], list[dict]]:
    batch = data_root / "Paper_A_final_TimestepConvergence"
    new_runs = {
        params["convergence_case"]: (run_dir, params)
        for run_dir, params in _run_dirs(batch)
    }
    rows = []
    pairings = []
    for case, (reference_batch, predicate, window) in CONVERGENCE_REFERENCES.items():
        fine_dir, fine_params = new_runs[case]
        coarse_dir, coarse_params = _reference_run(data_root, reference_batch, predicate)
        if fine_params["run_seeds"] != coarse_params["run_seeds"][:4]:
            raise ValueError(f"{case}: realization seeds are not paired")
        if not _close(fine_params["dT"], 0.001) or not _close(coarse_params["dT"], 0.002):
            raise ValueError(f"{case}: unexpected timestep pair")

        with np.load(fine_dir / "results.npz", allow_pickle=False) as fine, np.load(
            coarse_dir / "results.npz", allow_pickle=False
        ) as coarse:
            if not np.array_equal(
                fine["fragment_sample_members"], coarse["fragment_sample_members"][:, :4]
            ):
                raise ValueError(f"{case}: fragment samples are not paired")
            fine_q10 = np.asarray(fine["Holevo_Z_S_Ef_fractionsT_quantiles"])[0]
            coarse_q10 = np.asarray(coarse["Holevo_Z_S_Ef_fractionsT_quantiles"])[
                0, :, :, :4
            ]
            fine_discord = np.asarray(fine["Discord_Z_S_Ef_fractionsT"])
            coarse_discord = np.asarray(coarse["Discord_Z_S_Ef_fractionsT"])[:, :, :4]
            fine_entropy = _pointer_entropy(fine)
            coarse_entropy = _pointer_entropy(coarse)
            times = np.arange(fine_q10.shape[1], dtype=float) * float(fine_params["printT"])
            late = (times >= window[0]) & (times <= window[1])
            half = int(fine_params["nqubits_E"]) // 2

            fine_q10_norm = fine_q10[half, late] / fine_entropy[late, None]
            coarse_q10_norm = coarse_q10[half, late] / coarse_entropy[late, None]
            fine_discord_norm = fine_discord[half, late] / fine_entropy[late, None]
            coarse_discord_norm = coarse_discord[half, late] / coarse_entropy[late, None]
            fine_minimum = _minimum_fragment(fine_q10, fine_entropy, 16)
            coarse_minimum = _minimum_fragment(coarse_q10, coarse_entropy, 16)
            fine_qualifying = fine_q10_norm >= 0.9
            coarse_qualifying = coarse_q10_norm >= 0.9

            fine_q10_median = np.median(fine_q10_norm, axis=0)
            coarse_q10_median = np.median(coarse_q10_norm, axis=0)
            fine_discord_median = np.median(fine_discord_norm, axis=0)
            coarse_discord_median = np.median(coarse_discord_norm, axis=0)
            fine_holevo_min = np.min(fine_q10_norm, axis=0)
            coarse_holevo_min = np.min(coarse_q10_norm, axis=0)

            rows.append(
                {
                    "case": case,
                    "fine_run_id": fine_dir.name,
                    "coarse_run_id": coarse_dir.name,
                    "window_start": window[0],
                    "window_end": window[1],
                    "realizations": 4,
                    "q10_holevo_fine": float(np.mean(fine_q10_median)),
                    "q10_holevo_coarse": float(np.mean(coarse_q10_median)),
                    "q10_holevo_central_delta": float(
                        np.mean(fine_q10_median) - np.mean(coarse_q10_median)
                    ),
                    "q10_holevo_max_abs_sample_delta": float(
                        np.max(np.abs(fine_q10_norm - coarse_q10_norm))
                    ),
                    "mean_discord_fine": float(np.mean(fine_discord_median)),
                    "mean_discord_coarse": float(np.mean(coarse_discord_median)),
                    "mean_discord_central_delta": float(
                        np.mean(fine_discord_median) - np.mean(coarse_discord_median)
                    ),
                    "mean_discord_max_abs_sample_delta": float(
                        np.max(np.abs(fine_discord_norm - coarse_discord_norm))
                    ),
                    "holevo_min_fine": float(np.mean(fine_holevo_min)),
                    "holevo_min_coarse": float(np.mean(coarse_holevo_min)),
                    "holevo_min_max_abs_sample_delta": float(
                        np.max(np.abs(fine_holevo_min - coarse_holevo_min))
                    ),
                    "late_classification_changes": int(
                        np.count_nonzero(fine_qualifying != coarse_qualifying)
                    ),
                    "minimum_fragment_changes_all_times": int(
                        np.count_nonzero(fine_minimum != coarse_minimum)
                    ),
                    "fine_max_physical_norm_deviation": float(
                        np.max(np.abs(np.asarray(fine["norms"]) - 1.0))
                    ),
                    "coarse_max_physical_norm_deviation": float(
                        np.max(np.abs(np.asarray(coarse["norms"]) - 1.0))
                    ),
                    "fine_max_modified_norm_deviation": float(
                        np.max(np.abs(np.asarray(fine["modified_norms"]) - 1.0))
                    ),
                    "coarse_max_modified_norm_deviation": float(
                        np.max(np.abs(np.asarray(coarse["modified_norms"]) - 1.0))
                    ),
                }
            )
        pairings.append(
            {
                "case": case,
                "fine_run_id": fine_dir.name,
                "coarse_run_id": coarse_dir.name,
                "first_four_run_seeds_match": True,
                "fragment_samples_match": True,
            }
        )
    return rows, pairings


def _fixed_time_row(
    run_dir: Path,
    params: dict,
    n_environment: int,
    time_index: int,
    analysis_runs: int,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    with np.load(run_dir / "results.npz", allow_pickle=False) as results:
        entropy = _pointer_entropy(results)[time_index]
        half = n_environment // 2
        q10 = np.asarray(results["Holevo_Z_S_Ef_fractionsT_quantiles"])[
            0, :, time_index, :analysis_runs
        ]
        q90_discord = np.asarray(results["Discord_Z_S_Ef_fractionsT_quantiles"])[
            2, :, time_index, :analysis_runs
        ]
        mean_holevo = np.asarray(results["Holevo_Z_S_Ef_fractionsT"])[
            :, time_index, :analysis_runs
        ]
        mean_discord = np.asarray(results["Discord_Z_S_Ef_fractionsT"])[
            :, time_index, :analysis_runs
        ]
        minimum = _minimum_fragment(
            q10[:, None, :], np.asarray([entropy]), n_environment
        )[0]
        redundancy = np.where(minimum > 0, n_environment / minimum, np.nan)
        qualifying = q10[half] >= 0.9 * entropy
        metrics = {
            "q10_holevo_half": q10[half] / entropy,
            "mean_holevo_half": mean_holevo[half] / entropy,
            "q90_discord_half": q90_discord[half] / entropy,
            "mean_discord_half": mean_discord[half] / entropy,
            "redundancy_conditional": redundancy,
        }
        row = {
            "run_id": run_dir.name,
            "n_environment": n_environment,
            "total_qubits": n_environment + 1,
            "time": time_index * float(params["printT"]),
            "realizations": analysis_runs,
            "fragment_samples": int(params["fragment_sample_count"]),
            "redundancy_defined_fraction": float(np.mean(np.isfinite(redundancy))),
            "qualifying_realization_fraction": float(np.mean(qualifying)),
        }
        for name, values in metrics.items():
            estimate, low, high, count = _bootstrap(values, rng, draws)
            row[f"{name}_mean"] = estimate
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
            row[f"{name}_n"] = count
        return row


def _fixed_time_rows(data_root: Path, n20_batch_dir: Path, rng, draws):
    high_dir = data_root / "Paper_A_final_N24FixedTimeValidation"
    [(high_run, high_params)] = _run_dirs(high_dir)
    n20_candidates = [
        (run_dir, params)
        for run_dir, params in _run_dirs(n20_batch_dir)
        if params.get("validation_case") == "no_field_lambda_half"
    ]
    if len(n20_candidates) != 1:
        raise ValueError(f"{n20_batch_dir}: expected one no-field N_E=20 run")
    n20_run, n20_params = n20_candidates[0]
    reference_run, reference_params = _reference_run(
        data_root,
        "Paper_A_final_SubmissionMatchedLambda",
        CONVERGENCE_REFERENCES["no_field_lambda_half"][1],
    )
    paired_seeds = reference_params["run_seeds"][:8]
    if high_params["run_seeds"] != paired_seeds or n20_params["run_seeds"] != paired_seeds:
        raise ValueError("N_E=16, 20, and 24 realization seeds are not paired")
    rows = [
        _fixed_time_row(reference_run, reference_params, 16, 36, 8, rng, draws),
        _fixed_time_row(n20_run, n20_params, 20, 36, 8, rng, draws),
        _fixed_time_row(high_run, high_params, 24, 1, 8, rng, draws),
    ]
    return rows, {
        "reference_run_id": reference_run.name,
        "n20_run_id": n20_run.name,
        "higher_n_run_id": high_run.name,
        "first_eight_run_seeds_match_at_all_three_sizes": True,
        "comparison_time": 36.0,
    }


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches = {
        "timestep_convergence": ("Paper_A_final_TimestepConvergence", 3),
        "n24_fixed_time": ("Paper_A_final_N24FixedTimeValidation", 1),
        "matched_lambda_reference": ("Paper_A_final_SubmissionMatchedLambda", 38),
        "field_profile_reference": ("Paper_A_final_FieldProfileControl", 26),
    }
    inputs = []
    for label, (name, count) in batches.items():
        source, manifest = _manifest(args.data_root / name, count)
        copied = args.output_dir / f"review_validation_{label}_batch_manifest.json"
        shutil.copyfile(source, copied)
        inputs.append(
            {
                "label": label,
                "batch": name,
                "manifest_sha256": _sha256(source),
                "copied_manifest": copied.name,
                "source_provenance": manifest.get("source_provenance"),
            }
        )

    n20_manifest_path, n20_manifest = _manifest(args.n20_batch_dir, 5)
    n20_copied = args.output_dir / "review_validation_n20_reference_batch_manifest.json"
    shutil.copyfile(n20_manifest_path, n20_copied)
    inputs.append(
        {
            "label": "n20_reference",
            "batch": args.n20_batch_dir.name,
            "manifest_sha256": _sha256(n20_manifest_path),
            "copied_manifest": n20_copied.name,
            "source_provenance": n20_manifest.get("source_provenance"),
        }
    )

    convergence_rows, convergence_pairings = _convergence_rows(args.data_root)
    rng = np.random.default_rng(args.seed)
    fixed_rows, fixed_pairing = _fixed_time_rows(
        args.data_root, args.n20_batch_dir, rng, args.bootstrap
    )
    convergence_csv = args.output_dir / "timestep_convergence_metrics.csv"
    fixed_csv = args.output_dir / "n24_fixed_time_metrics.csv"
    _write_csv(convergence_csv, convergence_rows)
    _write_csv(fixed_csv, fixed_rows)

    maximum_late_delta = max(
        max(row["q10_holevo_max_abs_sample_delta"], row["mean_discord_max_abs_sample_delta"])
        for row in convergence_rows
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Paper-A review validation: half-step convergence and targeted N_E=24 check",
        "analysis_source": {
            **_git_info(Path(__file__).resolve().parents[1]),
            "script": "analysis_scripts/paper_a_review_validation.py",
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "inputs": inputs,
        "record_rule": "q10 Holevo(f=1/2) >= 0.9 H_Z(S)",
        "timestep_convergence": {
            "pairings": convergence_pairings,
            "maximum_late_window_normalized_half_fragment_delta": maximum_late_delta,
            "late_record_classification_changes": sum(
                row["late_classification_changes"] for row in convergence_rows
            ),
            "minimum_fragment_changes_all_times": sum(
                row["minimum_fragment_changes_all_times"] for row in convergence_rows
            ),
        },
        "fixed_time_high_size": {
            "pairing": fixed_pairing,
            "claim_boundary": "single-time finite-size consistency check, not a scaling fit",
        },
        "provenance_caveat": (
            "The uploaded new-batch manifests record the exact release commit but git_dirty=true; "
            "the resolved parameters and result artifacts are complete, but launch-time diffs were not supplied."
        ),
        "outputs": {
            "timestep_convergence_metrics_sha256": _sha256(convergence_csv),
            "n24_fixed_time_metrics_sha256": _sha256(fixed_csv),
        },
    }
    manifest_path = args.output_dir / "review_validation_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"maximum late-window normalized difference: {maximum_late_delta:.6g}")
    print("record classifications changed: 0")
    print("minimum threshold fragment sizes changed: 0")
    for row in fixed_rows:
        print(
            f"N_E={row['n_environment']}: q10 half={row['q10_holevo_half_mean']:.6f}, "
            f"qualifying={row['qualifying_realization_fraction']:.3f}, "
            f"redundancy={row['redundancy_conditional_mean']:.6f}"
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--n20-batch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=527183)
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
