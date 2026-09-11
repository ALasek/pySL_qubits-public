#!/usr/bin/env python3
"""Bootstrap Paper A's retained-fragment persistence statistic on the CPU.

The analysis is conditional on the stored fragment pool and the finite printed
time grid.  It does not certify all physical fragments or behaviour between
printed times.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
METRIC_NAME = "late_window_minimum_q10_holevo_bits"
DRAW_BLOCK = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def _stable_run_seed(seed: int, batch: str, run_id: str) -> int:
    payload = f"{seed}|{batch}|{run_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05)


def _persistence_minima(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 2:
        if not np.all(np.isfinite(values)):
            raise ValueError("fragment trajectories contain non-finite values")
        return float(np.min(np.quantile(values / np.log(2.0), 0.1, axis=1)))
    if values.ndim == 3:
        if not np.all(np.isfinite(values)):
            raise ValueError("fragment trajectories contain non-finite values")
        return np.min(np.quantile(values / np.log(2.0), 0.1, axis=2), axis=0)
    raise ValueError(f"expected [time, fragment] or [time, draw, fragment], got {values.shape}")


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    params_path = run_dir / "params.json"
    results_path = run_dir / "results.npz"
    if not params_path.is_file() or not results_path.is_file():
        raise ValueError("missing params.json or results.npz")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    try:
        n_environment = int(params["nqubits_E"])
        horizon = float(params["T"])
        print_interval = float(params["printT"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("params require numeric nqubits_E, T, and printT") from error
    if n_environment < 2 or not math.isfinite(horizon) or horizon <= 0 or print_interval <= 0:
        raise ValueError("invalid nqubits_E, T, or printT")
    half_size = n_environment // 2
    if half_size < 1:
        raise ValueError("no usable half-environment fragment size")

    with np.load(results_path, allow_pickle=False) as archive:
        required = {"Holevo_Z_S_Ef_fractionsT_samples", "fragment_sample_members"}
        absent = required.difference(archive.files)
        if absent:
            raise ValueError(f"missing retained arrays: {', '.join(sorted(absent))}")
        holevo = np.asarray(archive["Holevo_Z_S_Ef_fractionsT_samples"], dtype=float)
        members = np.asarray(archive["fragment_sample_members"], dtype=int)
    if holevo.ndim != 4:
        raise ValueError(f"Holevo samples must be [m,time,realization,fragment], got {holevo.shape}")
    if members.ndim != 4:
        raise ValueError(f"fragment members must be [m,realization,fragment,max_m], got {members.shape}")
    _, n_times, n_realizations, n_fragments = holevo.shape
    if (
        holevo.shape[0] <= half_size
        or members.shape[0] <= half_size
        or members.shape[1] != n_realizations
        or members.shape[2] != n_fragments
        or members.shape[3] < half_size
    ):
        raise ValueError("retained sample and member dimensions are incompatible with half fragments")
    seeds = params.get("run_seeds")
    if not isinstance(seeds, list) or len(seeds) != n_realizations:
        raise ValueError("run_seeds length does not match retained realization count")
    times = np.arange(n_times, dtype=float) * print_interval
    late = (times >= horizon / 2.0) & (times <= horizon + np.finfo(float).eps * max(1.0, horizon))
    if not np.any(late):
        raise ValueError("no printed samples in the second half of T")

    trajectories: list[np.ndarray] = []
    actual_counts: list[int] = []
    for realization in range(n_realizations):
        fragment_members = members[half_size, realization, :, :half_size]
        valid_members = np.all(
            (fragment_members >= 1) & (fragment_members <= n_environment), axis=1
        )
        values = holevo[half_size, late, realization, :]
        valid_values = np.all(np.isfinite(values), axis=0)
        valid = valid_members & valid_values
        actual_counts.append(int(np.count_nonzero(valid)))
        if not np.any(valid):
            raise ValueError(f"realization {realization} has no complete retained half-fragment trajectories")
        trajectories.append(values[:, valid])
    metadata = {
        "n_environment": n_environment,
        "half_fragment_size": half_size,
        "n_times": n_times,
        "late_time_count": int(np.count_nonzero(late)),
        "late_time_start": float(times[late][0]),
        "late_time_end": float(times[late][-1]),
        "n_realizations": n_realizations,
        "retained_half_fragment_counts": actual_counts,
    }
    return metadata, trajectories


def _bootstrap(trajectories: list[np.ndarray], draws: int, seed: int) -> dict[str, Any]:
    if draws < 1:
        raise ValueError("draws must be positive")
    central = np.asarray([_persistence_minima(values) for values in trajectories], dtype=float)
    n_realizations = central.size
    rng = np.random.default_rng(seed)
    realization_only = np.empty(draws, dtype=float)
    nested = np.empty(draws, dtype=float)
    for first in range(0, draws, DRAW_BLOCK):
        count = min(DRAW_BLOCK, draws - first)
        outer = rng.integers(0, n_realizations, size=(count, n_realizations))
        occurrence_minima = np.empty((count, n_realizations), dtype=float)
        for realization, values in enumerate(trajectories):
            draw_indices, occurrence_indices = np.nonzero(outer == realization)
            if not draw_indices.size:
                continue
            fragment_count = values.shape[1]
            fragment_indices = rng.integers(
                0, fragment_count, size=(draw_indices.size, fragment_count)
            )
            selected = values[:, fragment_indices]
            occurrence_minima[draw_indices, occurrence_indices] = _persistence_minima(selected)
        realization_only[first : first + count] = np.mean(central[outer], axis=1)
        nested[first : first + count] = np.mean(occurrence_minima, axis=1)
    realization_ci = np.quantile(realization_only, [0.025, 0.975])
    nested_ci = np.quantile(nested, [0.025, 0.975])
    central_mean = float(np.mean(central))
    return {
        "metric": METRIC_NAME,
        "central_mean": central_mean,
        "original_success_count_ge_0_9": int(np.count_nonzero(central >= 0.9)),
        "original_realization_count": int(n_realizations),
        "realization_bootstrap_mean": float(np.mean(realization_only)),
        "realization_bootstrap_ci_low": float(realization_ci[0]),
        "realization_bootstrap_ci_high": float(realization_ci[1]),
        "realization_bootstrap_ci_width": float(realization_ci[1] - realization_ci[0]),
        "nested_bootstrap_mean": float(np.mean(nested)),
        "nested_bootstrap_ci_low": float(nested_ci[0]),
        "nested_bootstrap_ci_high": float(nested_ci[1]),
        "nested_bootstrap_ci_width": float(nested_ci[1] - nested_ci[0]),
        "nested_mean_shift": float(np.mean(nested) - central_mean),
        "nested_ci_width_change": float((nested_ci[1] - nested_ci[0]) - (realization_ci[1] - realization_ci[0])),
        "fraction_realization_bootstrap_means_ge_0_9": float(np.mean(realization_only >= 0.9)),
        "fraction_nested_bootstrap_means_ge_0_9": float(np.mean(nested >= 0.9)),
    }


def _input_key(run_dir: Path, draws: int, seed: int, batch: str) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric": METRIC_NAME,
        "draws": draws,
        "base_seed": seed,
        "batch": batch,
        "analysis_script_sha256_normalized_lf": _source_sha256(),
        "params_sha256": _sha256_file(run_dir / "params.json"),
        "results_sha256": _sha256_file(run_dir / "results.npz"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _analyse_run(run_dir: Path, batch: str, draws: int, base_seed: int, cache_path: Path) -> dict[str, Any]:
    run_id = run_dir.name
    try:
        input_key = _input_key(run_dir, draws, base_seed, batch)
    except (OSError, ValueError) as error:
        return {"batch": batch, "run_id": run_id, "status": "missing", "reason": str(error)}
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("input_key") == input_key:
                return cached
        except (OSError, json.JSONDecodeError):
            pass
    try:
        metadata, trajectories = _load_run(run_dir)
        result = _bootstrap(trajectories, draws, _stable_run_seed(base_seed, batch, run_id))
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        result = {"status": "missing", "reason": str(error)}
    else:
        result["status"] = "analysed"
    result.update(
        {
            "batch": batch,
            "run_id": run_id,
            "input_key": input_key,
            "draws": draws,
            "base_seed": base_seed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if result["status"] == "analysed":
        result.update(metadata)
    _write_json(cache_path, result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            for key, value in flattened.items():
                if isinstance(value, (list, dict)):
                    flattened[key] = json.dumps(_jsonable(value), sort_keys=True)
            writer.writerow(_jsonable(flattened))


def _report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    analysed = [row for row in rows if row["status"] == "analysed"]
    missing = [row for row in rows if row["status"] != "analysed"]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU two-stage bootstrap sensitivity analysis of Paper A persistence.",
        "scope": (
            "Results are conditional on the retained sampled fragment pool and finite printed second-half time grid. "
            "They provide no universal guarantee for all physical fragments, between printed times, or across disorder."
        ),
        "resampling": {
            "draws": args.draws,
            "base_seed": args.seed,
            "outer_level": "complete realizations resampled with replacement",
            "inner_level": "complete retained half-fragment trajectories resampled independently for each selected realization occurrence",
            "time_rule": "the same selected fragment indices are used for every late-window time",
            "statistic_order": "q10 over fragments at each time, then minimum over printed second-half times",
            "memory": f"draws are processed in blocks of at most {DRAW_BLOCK}",
        },
        "batches": args.batches,
        "analysed_run_count": len(analysed),
        "missing_run_count": len(missing),
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--batches", nargs="+", required=True)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for batch in args.batches:
        runs_dir = data_root / batch / "runs"
        if not runs_dir.is_dir():
            rows.append({"batch": batch, "run_id": "", "status": "missing", "reason": "missing runs directory"})
        else:
            for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
                cache_name = f"{batch}__{run_dir.name}.json"
                rows.append(
                    _analyse_run(run_dir, batch, args.draws, args.seed, output_dir / "runs" / cache_name)
                )
                if len(rows) % 20 == 0:
                    _write_csv(output_dir / "bootstrap_persistence.csv", rows)
                    _write_json(output_dir / "bootstrap_persistence.json", _report(rows, args))
                    print(f"{len(rows)} bootstrap cases", flush=True)
    _write_csv(output_dir / "bootstrap_persistence.csv", rows)
    report = _report(rows, args)
    _write_json(output_dir / "bootstrap_persistence.json", report)
    print(json.dumps({key: report[key] for key in ("analysed_run_count", "missing_run_count")}, indent=2))
    print(f"wrote {output_dir / 'bootstrap_persistence.csv'}")
    print(f"wrote {output_dir / 'bootstrap_persistence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
