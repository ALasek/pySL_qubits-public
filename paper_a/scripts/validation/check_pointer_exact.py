#!/usr/bin/env python3
"""Audit saved Paper A pointer-basis grids with exact local branch evolution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm


DEFAULT_CHECKPOINT_DIR = Path(
    r"D:\Github\pySL_qubits\data\figs\nonbatch\pointer_prod_bd3b66_T60_m4m6\checkpoints"
)
DEFAULT_RUN_DIR = Path(
    r"D:\Github\pySL_qubits\data\nonbatch\runs\bd3b66ac18c85a619ea58ad7110a907e28de38c6c13b6bd28ee45c5f37bb608f"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\Github\Papers\output\recalculations\paper-a-2026-09-05\pointer_exact")
EXPECTED_GRID = (21, 40)
EXPECTED_TIME = 60.0
EXPECTED_CHECKPOINTS = 240


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _entropy(matrix: np.ndarray, trace: float | None = None, eps: float = 1e-12) -> float:
    values = np.maximum(np.linalg.eigvalsh((matrix + matrix.conj().T) * 0.5).real, 0.0)
    normalizer = float(values.sum()) if trace is None else float(trace)
    if normalizer <= eps:
        return 0.0
    probabilities = values[values > eps] / normalizer
    return float(-(probabilities * np.log2(probabilities)).sum())


def _seeded_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.blake2b(f"{seed}:{label}".encode("utf-8"), digest_size=16).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _local_gammas(params: dict[str, Any], seed: int, time_value: float) -> np.ndarray:
    n_environment = int(params["nqubits_E"])
    theta = float(params["Mironowicz_theta"])
    h0 = float(params.get("Mironowicz_h0", 0.0))
    p_bias = float(params["psi_bias"])
    if not 0.0 <= p_bias <= 1.0:
        raise ValueError(f"psi_bias must be in [0, 1], got {p_bias}")
    coupling_rng = _seeded_rng(seed, "mironowicz.J_SE")
    field_rng = _seeded_rng(seed, "mironowicz.J_E")
    couplings = np.asarray(
        [coupling_rng.gauss(0.0, float(params["H_SE_J"])) for _ in range(n_environment)]
    )
    fields = np.asarray(
        [h0 + field_rng.gauss(0.0, float(params["Mironowicz_alpha2"])) for _ in range(n_environment)]
    )
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    initial_environment = np.array([math.sqrt(p_bias), math.sqrt(1.0 - p_bias)], dtype=complex)
    gammas = np.empty(n_environment, dtype=complex)
    for index, (coupling, field) in enumerate(zip(couplings, fields)):
        h0_branch = field * z
        h1_branch = (
            math.pi * coupling / 2.0 * identity
            - math.pi * coupling * math.cos(theta) / 2.0 * x
            + (field - math.pi * coupling * math.sin(theta) / 2.0) * z
        )
        e0 = expm(-1j * time_value * h0_branch) @ initial_environment
        e1 = expm(-1j * time_value * h1_branch) @ initial_environment
        gammas[index] = np.vdot(e0, e1)
    return gammas


def _blocks_from_gammas(gammas: np.ndarray, fragment: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(fragment, dtype=int) - 1
    if indices.ndim != 1 or not indices.size or len(set(indices.tolist())) != indices.size:
        raise ValueError(f"invalid fragment sites {fragment!r}")
    if np.any(indices < 0) or np.any(indices >= gammas.size):
        raise ValueError(f"fragment sites outside 1..{gammas.size}: {fragment!r}")
    fragment_overlap = np.prod(gammas[indices])
    rest_overlap = np.prod(np.delete(gammas, indices))
    perpendicular_norm = math.sqrt(max(0.0, 1.0 - abs(fragment_overlap) ** 2))
    state_zero = np.array([1.0, 0.0], dtype=complex)
    state_one = np.array([fragment_overlap, perpendicular_norm], dtype=complex)
    g00 = np.outer(state_zero, state_zero.conj()) * 0.5
    g11 = np.outer(state_one, state_one.conj()) * 0.5
    g01 = np.outer(state_zero, state_one.conj()) * (0.5 * rest_overlap.conjugate())
    return g00, g11, g01


def _chi_grid(g00: np.ndarray, g11: np.ndarray, g01: np.ndarray, beta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    g10 = g01.conj().T
    fragment_entropy = _entropy(g00 + g11)
    grid = np.empty((beta.size, phi.size), dtype=float)
    for beta_index, beta_value in enumerate(beta):
        cosine = math.cos(float(beta_value) / 2.0)
        sine = math.sin(float(beta_value) / 2.0)
        for phi_index, phi_value in enumerate(phi):
            phase = complex(math.cos(float(phi_value)), math.sin(float(phi_value)))
            cross = phase * g01 + phase.conjugate() * g10
            rho_plus = cosine * cosine * g00 + sine * sine * g11 + cosine * sine * cross
            rho_minus = sine * sine * g00 + cosine * cosine * g11 - cosine * sine * cross
            p_plus = float(np.trace(rho_plus).real)
            p_minus = float(np.trace(rho_minus).real)
            grid[beta_index, phi_index] = max(
                0.0,
                fragment_entropy - p_plus * _entropy(rho_plus, p_plus) - p_minus * _entropy(rho_minus, p_minus),
            )
    return grid


def _mutual_information(g00: np.ndarray, g11: np.ndarray, g01: np.ndarray) -> float:
    rho_sf = np.block([[g00, g01], [g01.conj().T, g11]])
    rho_s = np.array(
        [[np.trace(g00), np.trace(g01)], [np.trace(g01.conj().T), np.trace(g11)]], dtype=complex
    )
    return max(0.0, _entropy(rho_s) + _entropy(g00 + g11) - _entropy(rho_sf))


def _best(grid: np.ndarray, beta: np.ndarray, phi: np.ndarray) -> dict[str, Any]:
    flat_index = int(np.argmax(grid))
    beta_index, phi_index = np.unravel_index(flat_index, grid.shape)
    return {
        "beta_index": int(beta_index),
        "phi_index": int(phi_index),
        "beta": float(beta[beta_index]),
        "phi": float(phi[phi_index]),
        "holevo_bits": float(grid[beta_index, phi_index]),
        "gain_over_z_bits": float(grid[beta_index, phi_index] - grid[0, 0]),
    }


def _validate_params(params: dict[str, Any]) -> None:
    expected = {
        "H_SE_Special": "Mironowicz_rand",
        "nqubits_S": 1,
        "nqubits_E": 17,
        "psi_S_spec": "x+",
        "psi_E_spec": "bias",
        "H_EE_J": 0.0,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"expected params[{key!r}]={value!r}, got {params.get(key)!r}")
    if not math.isclose(float(params["H_SE_J"]), 0.1, abs_tol=1e-15):
        raise ValueError("expected H_SE_J=0.1")
    if not math.isclose(float(params["Mironowicz_alpha2"]), 0.25, abs_tol=1e-15):
        raise ValueError("expected Mironowicz_alpha2=0.25")
    if not math.isclose(float(params["Mironowicz_theta"]), 0.6, abs_tol=1e-15):
        raise ValueError("expected Mironowicz_theta=0.6")
    if not math.isclose(float(params["psi_bias"]), 0.35, abs_tol=1e-15):
        raise ValueError("expected psi_bias=0.35")


def _brute_force_check() -> float:
    beta = np.linspace(0.0, math.pi, 3)
    phi = np.linspace(0.0, 2.0 * math.pi, 4, endpoint=False)
    parameters = {
        "nqubits_E": 2,
        "Mironowicz_theta": 0.6,
        "Mironowicz_h0": 0.0,
        "psi_bias": 0.35,
        "H_SE_J": 0.1,
        "Mironowicz_alpha2": 0.25,
    }
    gammas = _local_gammas(parameters, 713, 1.7)
    g00, g11, g01 = _blocks_from_gammas(gammas, [1])
    exact_grid = _chi_grid(g00, g11, g01, beta, phi)

    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    state = np.array([math.sqrt(0.35), math.sqrt(0.65)], dtype=complex)
    coupling_rng = _seeded_rng(713, "mironowicz.J_SE")
    field_rng = _seeded_rng(713, "mironowicz.J_E")
    branches = []
    for _ in range(2):
        coupling = coupling_rng.gauss(0.0, 0.1)
        field = field_rng.gauss(0.0, 0.25)
        e0 = expm(-1j * 1.7 * field * z) @ state
        h1 = math.pi * coupling / 2.0 * identity - math.pi * coupling * math.cos(0.6) / 2.0 * x + (
            field - math.pi * coupling * math.sin(0.6) / 2.0
        ) * z
        e1 = expm(-1j * 1.7 * h1) @ state
        branches.append((e0, e1))
    psi = (
        np.kron(np.array([1.0, 0.0]), np.kron(branches[0][0], branches[1][0]))
        + np.kron(np.array([0.0, 1.0]), np.kron(branches[0][1], branches[1][1]))
    ) / math.sqrt(2.0)
    amplitudes = psi.reshape(2, 2, 2)
    direct_g00 = amplitudes[0] @ amplitudes[0].conj().T
    direct_g11 = amplitudes[1] @ amplitudes[1].conj().T
    direct_g01 = amplitudes[0] @ amplitudes[1].conj().T
    direct_grid = _chi_grid(direct_g00, direct_g11, direct_g01, beta, phi)
    error = float(np.max(np.abs(exact_grid - direct_grid)))
    if error > 1e-12:
        raise AssertionError(f"two-dimensional Gram reduction disagrees with brute force by {error:g}")
    return error


def _load_checkpoint(path: Path, beta: np.ndarray | None, phi: np.ndarray | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as source:
        metadata_raw = str(np.asarray(source["metadata_json"]).item())
        metadata = json.loads(metadata_raw)
        stored = np.asarray(source["chi_grid"], dtype=float)
        source_i = float(source["I"].item())
    checkpoint_beta = np.asarray(metadata["beta_grid"], dtype=float)
    checkpoint_phi = np.asarray(metadata["phi_grid"], dtype=float)
    if stored.shape != EXPECTED_GRID or checkpoint_beta.shape != (EXPECTED_GRID[0],) or checkpoint_phi.shape != (EXPECTED_GRID[1],):
        raise ValueError(f"{path}: unexpected pointer-grid shape")
    if beta is not None and (not np.array_equal(beta, checkpoint_beta) or not np.array_equal(phi, checkpoint_phi)):
        raise ValueError(f"{path}: inconsistent beta or phi grid")
    if not math.isclose(float(metadata["time_value"]), EXPECTED_TIME, abs_tol=1e-12):
        raise ValueError(f"{path}: expected time_value={EXPECTED_TIME:g}")
    return metadata, checkpoint_beta, checkpoint_phi, stored, source_i


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no checkpoint rows to write")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_paths = sorted(checkpoint_dir.glob("*.npz"))
    if len(checkpoint_paths) != EXPECTED_CHECKPOINTS:
        raise ValueError(f"expected {EXPECTED_CHECKPOINTS} checkpoints, found {len(checkpoint_paths)}")
    params_path = run_dir / "params.json"
    snapshots_path = run_dir / "psi_snapshots.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    _validate_params(params)
    run_seeds = params.get("run_seeds")
    if not isinstance(run_seeds, list) or len(run_seeds) != 6:
        raise ValueError("expected six saved run_seeds")
    if snapshots.get("count") != len(run_seeds):
        raise ValueError("psi snapshot metadata does not cover every saved realization")
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_checkpoint_dir = output_dir / "checkpoints"
    exact_checkpoint_dir.mkdir(exist_ok=True)

    beta: np.ndarray | None = None
    phi: np.ndarray | None = None
    gamma_cache: dict[int, np.ndarray] = {}
    grouped_source: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_exact: dict[int, list[np.ndarray]] = defaultdict(list)
    per_realization_source: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    per_realization_exact: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        metadata, checkpoint_beta, checkpoint_phi, stored, source_i = _load_checkpoint(path, beta, phi)
        beta, phi = checkpoint_beta, checkpoint_phi
        run_index = int(metadata["run_index"])
        fragment_size = int(metadata["fragment_size"])
        fragment = [int(site) for site in metadata["fragment"]]
        if len(fragment) != fragment_size:
            raise ValueError(f"{path}: fragment length does not match fragment_size")
        if run_index < 0 or run_index >= len(run_seeds):
            raise ValueError(f"{path}: invalid run_index {run_index}")
        gammas = gamma_cache.setdefault(
            run_index, _local_gammas(params, int(run_seeds[run_index]), float(metadata["time_value"]))
        )
        g00, g11, g01 = _blocks_from_gammas(gammas, fragment)
        exact = _chi_grid(g00, g11, g01, checkpoint_beta, checkpoint_phi)
        exact_i = _mutual_information(g00, g11, g01)
        difference = np.abs(exact - stored)
        difference_index = np.unravel_index(int(np.argmax(difference)), difference.shape)
        source_best = _best(stored, checkpoint_beta, checkpoint_phi)
        exact_best = _best(exact, checkpoint_beta, checkpoint_phi)
        row = {
            "checkpoint": path.name,
            "fragment_size": fragment_size,
            "fragment_id": int(metadata["fragment_id"]),
            "run_index": run_index,
            "seed": int(run_seeds[run_index]),
            "fragment": json.dumps(fragment),
            "max_abs_grid_error_bits": float(np.max(difference)),
            "max_error_beta_index": int(difference_index[0]),
            "max_error_phi_index": int(difference_index[1]),
            "source_z_holevo_bits": float(stored[0, 0]),
            "exact_z_holevo_bits": float(exact[0, 0]),
            "z_abs_error_bits": float(abs(exact[0, 0] - stored[0, 0])),
            "source_mutual_information_bits": source_i,
            "exact_mutual_information_bits": exact_i,
            "mutual_information_abs_error_bits": float(abs(exact_i - source_i)),
            "source_best_beta_index": source_best["beta_index"],
            "source_best_phi_index": source_best["phi_index"],
            "source_best_beta": source_best["beta"],
            "source_best_phi": source_best["phi"],
            "source_best_holevo_bits": source_best["holevo_bits"],
            "source_best_gain_over_z_bits": source_best["gain_over_z_bits"],
            "exact_best_beta_index": exact_best["beta_index"],
            "exact_best_phi_index": exact_best["phi_index"],
            "exact_best_beta": exact_best["beta"],
            "exact_best_phi": exact_best["phi"],
            "exact_best_holevo_bits": exact_best["holevo_bits"],
            "exact_best_gain_over_z_bits": exact_best["gain_over_z_bits"],
        }
        rows.append(row)
        grouped_source[fragment_size].append(stored)
        grouped_exact[fragment_size].append(exact)
        per_realization_source[fragment_size, run_index].append(stored)
        per_realization_exact[fragment_size, run_index].append(exact)
        exact_metadata = {
            **metadata,
            "exact_recalculation": True,
            "exact_recalculation_method": "direct local 2x2 branch exponentials with Gram-reduced S+fragment state",
            "source_checkpoint_sha256": _sha256(path),
        }
        np.savez_compressed(
            exact_checkpoint_dir / path.name,
            metadata_json=np.asarray(json.dumps(exact_metadata, sort_keys=True, separators=(",", ":"))),
            chi_grid=exact,
            I=np.asarray(exact_i),
        )
    assert beta is not None and phi is not None
    _write_csv(output_dir / "pointer_exact_rows.csv", rows)

    aggregate_rows = []
    compatibility_results = {}
    for fragment_size in sorted(grouped_exact):
        source_mean = np.mean(np.stack(grouped_source[fragment_size]), axis=0)
        exact_mean = np.mean(np.stack(grouped_exact[fragment_size]), axis=0)
        aggregate_error = np.abs(exact_mean - source_mean)
        source_best = _best(source_mean, beta, phi)
        exact_best = _best(exact_mean, beta, phi)
        np.savez_compressed(
            output_dir / f"chi_grid_m{fragment_size}.npz",
            beta_grid=beta,
            phi_grid=phi,
            mean_chi_grid=exact_mean,
            mean_chi_z=np.asarray(exact_mean[0, 0]),
        )
        compatibility_results[str(fragment_size)] = {
            "mean_chi_z": float(exact_mean[0, 0]),
            "mean_chi_global_max": exact_best["holevo_bits"],
            "best_beta": exact_best["beta"],
            "best_phi": exact_best["phi"],
            "delta_chi_global": exact_best["gain_over_z_bits"],
        }
        aggregate_rows.append(
            {
                "fragment_size": fragment_size,
                "checkpoint_count": len(grouped_exact[fragment_size]),
                "aggregate_max_abs_grid_error_bits": float(np.max(aggregate_error)),
                "aggregate_z_abs_error_bits": float(abs(exact_mean[0, 0] - source_mean[0, 0])),
                "source_best": source_best,
                "exact_best": exact_best,
                "source_exact_best_location_matches": (
                    source_best["beta_index"] == exact_best["beta_index"]
                    and source_best["phi_index"] == exact_best["phi_index"]
                ),
            }
        )
    per_realization_rows = []
    for (fragment_size, run_index), source_grids in sorted(per_realization_source.items()):
        source_mean = np.mean(np.stack(source_grids), axis=0)
        exact_mean = np.mean(np.stack(per_realization_exact[fragment_size, run_index]), axis=0)
        per_realization_rows.append(
            {
                "fragment_size": fragment_size,
                "run_index": run_index,
                "source_mean_chi_z": float(source_mean[0, 0]),
                "exact_mean_chi_z": float(exact_mean[0, 0]),
                "max_abs_grid_error_bits": float(np.max(np.abs(source_mean - exact_mean))),
            }
        )
    _write_csv(output_dir / "per_realization_summary.csv", per_realization_rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis": "exact local branch-state reconstruction of saved pointer-basis grids",
        "exact_output": True,
        "checkpoint_count": len(rows),
        "beta_count": int(beta.size),
        "phi_count": int(phi.size),
        "fragment_sizes": sorted(grouped_exact),
        "num_realizations": len(run_seeds),
        "num_fragments_per_size": {
            str(size): len(values) // len(run_seeds) for size, values in grouped_exact.items()
        },
        "metadata": {"n_qubits": int(params["nqubits_S"]) + int(params["nqubits_E"]), "system_qubit": 0, "endianness": "big"},
        "results_by_fragment_size": compatibility_results,
        "aggregate_comparison": aggregate_rows,
        "overall_max_abs_checkpoint_grid_error_bits": float(max(row["max_abs_grid_error_bits"] for row in rows)),
        "overall_max_abs_checkpoint_z_error_bits": float(max(row["z_abs_error_bits"] for row in rows)),
        "brute_force_small_environment_grid_max_abs_error": _brute_force_check(),
        "phase_convention": (
            "gamma_j=<e0_j|e1_j> is evaluated from direct 2x2 exponentials. The scalar pi*g_j/2 identity term is retained "
            "in H1, so its relative branch phase enters the phi-grid coherence block."
        ),
        "scope": (
            "This audit reuses the saved realization seeds and physical one-based fragment sites. It checks the finite saved t=60 grid "
            "under the independent local-branch model; it is not a new simulation or a claim about unsaved times or fragments."
        ),
        "inputs": {
            "params": {"path": str(params_path), "sha256": _sha256(params_path)},
            "psi_snapshots": {"path": str(snapshots_path), "sha256": _sha256(snapshots_path)},
            "checkpoint_count": len(checkpoint_paths),
            "checkpoint_bundle_sha256": hashlib.sha256(
                b"".join(path.name.encode("utf-8") + bytes.fromhex(_sha256(path)) for path in checkpoint_paths)
            ).hexdigest(),
        },
        "outputs": {
            "checkpoint_dir": str(exact_checkpoint_dir),
            "rows_csv": "pointer_exact_rows.csv",
            "per_realization_csv": "per_realization_summary.csv",
            "exact_aggregate_grids": [f"chi_grid_m{size}.npz" for size in sorted(grouped_exact)],
        },
    }
    _write_json(output_dir / "pointer_exact_report.json", summary)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("checkpoint_count", "overall_max_abs_checkpoint_grid_error_bits", "overall_max_abs_checkpoint_z_error_bits")}, indent=2))
    print(f"wrote {output_dir / 'pointer_exact_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
