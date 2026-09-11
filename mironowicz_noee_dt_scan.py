import argparse
import csv
import datetime as dt
import importlib
import json
import os
import pathlib
import subprocess
import time

import numpy as np


DEFAULT_MANIFEST = pathlib.Path("experiments/mironowicz_nE17_noee_dt_scan.json")
DEFAULT_REPORT_DIR = pathlib.Path("experiments/results/mironowicz_nE17_noee_dt_scan_v1")


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def _resolved_params(params, timestep, resolve_run_seed_plan_from_params):
    run_params = {**params, "dT": timestep}
    resolved = dict(run_params)
    resolved.update(resolve_run_seed_plan_from_params(run_params))
    return run_params, resolved


def _validate_exact_model(params):
    required = {
        "nqubits_S": 1,
        "psi_S_spec": "x+",
        "psi_E_spec": "bias",
        "H_SE_Special": "Mironowicz_rand",
        "noiseT": 0,
        "p_noise_Gate": 0,
    }
    mismatches = {key: (params.get(key), expected) for key, expected in required.items() if params.get(key) != expected}
    if params.get("H_S") or params.get("H_S_J"):
        mismatches["H_S"] = (params.get("H_S"), "no system self-Hamiltonian")
    if mismatches:
        raise ValueError(f"exact no-E-E comparator assumptions violated: {mismatches}")


def _run_artifact(run_dir, StoredRunResult):
    metadata = _load_json(run_dir / "metadata.json")
    params = _load_json(run_dir / "params.json")
    return StoredRunResult(params, run_dir / "results.npz", metadata)


def _profile_summary(perf):
    profiled_steps = perf.get("evolve_profiled_steps", 0)
    total_steps = perf.get("evolve_steps", 0)
    keys = ("evolve_psi_real", "evolve_psi_imag", "evolve_phi_real", "evolve_phi_imag")
    sampled_seconds = sum(perf.get(key, 0.0) for key in keys)
    estimated_seconds = None
    if profiled_steps:
        estimated_seconds = sampled_seconds * total_steps / profiled_steps
    return {
        "total_steps": int(total_steps),
        "profiled_steps": int(profiled_steps),
        "sampled_evolution_seconds": float(sampled_seconds),
        "estimated_evolution_seconds": None if estimated_seconds is None else float(estimated_seconds),
        "analysis_seconds": float(perf.get("total", 0.0)),
    }


def _local_model(params, seed, hamiltonian, cp):
    model = hamiltonian(params, seed=seed)
    h_se = cp.asnumpy(model.H_SE_matrix)
    h_e = cp.asnumpy(model.H_E_matrix)
    j_se = np.asarray(model.J_SE, dtype=float)
    j_e = np.asarray(model.J_E, dtype=float)
    branch_zero = [coupling * h_e for coupling in j_e]
    branch_one = [field * h_e + coupling * h_se[2:4, 2:4] for coupling, field in zip(j_se, j_e)]
    return branch_zero, branch_one, j_se, j_e


def _spectral_summary(branch_zero, branch_one):
    bounds = []
    for branch in (branch_zero, branch_one):
        eigenvalues = [np.linalg.eigvalsh(local) for local in branch]
        bounds.append((sum(values[0] for values in eigenvalues), sum(values[-1] for values in eigenvalues)))
    minimum = float(min(bound[0] for bound in bounds))
    maximum = float(max(bound[1] for bound in bounds))
    radius = max(abs(minimum), abs(maximum))
    return {
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "spectral_radius": radius,
        "leapfrog_stability_dt_limit": 2.0 / radius,
    }


def _local_eigensystems(branch):
    return [np.linalg.eigh(local) for local in branch]


def _exact_state(time_value, eigensystems, environment_state):
    branches = []
    for branch in eigensystems:
        state = np.array([1.0 + 0.0j])
        for eigenvalues, eigenvectors in branch:
            coefficients = eigenvectors.conj().T @ environment_state
            evolved = eigenvectors @ (np.exp(-1j * time_value * eigenvalues) * coefficients)
            state = np.kron(state, evolved)
        branches.append(state)
    return np.concatenate(branches) / np.sqrt(2.0)


def _normalized_rho_system(state):
    matrix = state.reshape(2, -1)
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho).real


def _entropy(rho):
    eigenvalues = np.clip(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real, 0.0, 1.0)
    nonzero = eigenvalues[eigenvalues > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def _trace_distance(left, right):
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _finite_max_abs(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    mask = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[mask] - right[mask]))) if np.any(mask) else None


def _richardson_estimate(error, timestep, reference_timestep, order=2):
    if error is None or timestep == reference_timestep:
        return 0.0 if error is not None else None
    ratio = (reference_timestep / timestep) ** order
    return error / (1.0 - ratio)


def _tail_summary(times, values, tail_start):
    times = np.asarray(times)
    values = np.asarray(values, dtype=float)
    mask = times >= tail_start
    tail_times = times[mask]
    tail = values[mask]
    slope = np.polyfit(tail_times, tail, 1)[0] if len(tail) > 1 else 0.0
    return {
        "start": float(tail_start),
        "samples": int(len(tail)),
        "mean": float(np.mean(tail)),
        "std": float(np.std(tail)),
        "minimum": float(np.min(tail)),
        "maximum": float(np.max(tail)),
        "linear_slope_per_time": float(slope),
    }


def _compare_runs(manifest, artifacts, run_records, local_model):
    branch_zero, branch_one, j_se, j_e = local_model
    spectral = _spectral_summary(branch_zero, branch_one)
    eigensystems = (_local_eigensystems(branch_zero), _local_eigensystems(branch_one))
    p = float(manifest["params"]["psi_bias"])
    environment_state = np.array([np.sqrt(p), np.sqrt(1.0 - p)], dtype=np.complex128)
    times = np.arange(artifacts[0].S_vn_runAv.shape[0]) * float(manifest["params"]["printT"])

    exact_states = []
    exact_rhos = []
    exact_entropies = []
    for time_value in times:
        state = _exact_state(time_value, eigensystems, environment_state)
        rho = _normalized_rho_system(state)
        exact_states.append(state)
        exact_rhos.append(rho)
        exact_entropies.append(_entropy(rho))

    reference_dt = float(manifest["reference_dt_for_sampled_metrics"])
    reference_index = [float(value) for value in manifest["dT_values"]].index(reference_dt)
    reference = artifacts[reference_index]
    reference_mi = np.asarray(reference.I_S_Ef_fractionsT_runAv)
    reference_holevo = getattr(reference, "Holevo_Z_S_Ef_fractionsT_runAv", None)
    reference_discord = getattr(reference, "Discord_Z_S_Ef_fractionsT_runAv", None)
    rows = []

    for timestep, artifact in zip(manifest["dT_values"], artifacts):
        snapshots = artifact.load_psi_snapshots(mmap_mode="r")
        infidelities = []
        aligned_errors = []
        rho_distances = []
        for index, exact in enumerate(exact_states):
            numerical = snapshots[index, 0].astype(np.complex128)
            numerical += 1j * snapshots[index, 1]
            numerical /= np.linalg.norm(numerical)
            exact_normalized = exact / np.linalg.norm(exact)
            overlap = np.vdot(exact_normalized, numerical)
            fidelity = min(1.0, float(abs(overlap) ** 2))
            infidelities.append(max(0.0, 1.0 - fidelity))
            phase = np.exp(-1j * np.angle(overlap)) if overlap else 1.0
            aligned_errors.append(float(np.linalg.norm(phase * numerical - exact_normalized)))
            rho_distances.append(_trace_distance(_normalized_rho_system(numerical), exact_rhos[index]))

        norms = np.asarray(artifact.norms, dtype=float)
        modified_norms = np.asarray(artifact.modified_norms, dtype=float)
        s_vn = np.asarray(artifact.S_vn_runAv, dtype=float)
        mi = np.asarray(artifact.I_S_Ef_fractionsT_runAv)
        holevo = getattr(artifact, "Holevo_Z_S_Ef_fractionsT_runAv", None)
        discord = getattr(artifact, "Discord_Z_S_Ef_fractionsT_runAv", None)
        mi_error = _finite_max_abs(mi, reference_mi)
        holevo_error = (
            _finite_max_abs(holevo, reference_holevo)
            if holevo is not None and reference_holevo is not None
            else None
        )
        discord_error = (
            _finite_max_abs(discord, reference_discord)
            if discord is not None and reference_discord is not None
            else None
        )
        record = run_records[str(timestep)]
        profile = record.get("profile", {})
        rows.append(
            {
                "dT": float(timestep),
                "steps": int(round(float(manifest["params"]["T"]) / float(timestep))),
                "spectral_radius_times_dT": spectral["spectral_radius"] * float(timestep),
                "max_norm_deviation": float(np.max(np.abs(norms - 1.0))),
                "max_modified_norm_deviation": float(np.max(np.abs(modified_norms - 1.0))),
                "max_state_infidelity": float(np.max(infidelities)),
                "final_state_infidelity": float(infidelities[-1]),
                "max_phase_aligned_state_l2_error": float(np.max(aligned_errors)),
                "max_rhoS_trace_distance": float(np.max(rho_distances)),
                "max_S_vn_abs_error": float(np.max(np.abs(s_vn - exact_entropies))),
                "max_MI_abs_error_vs_reference": mi_error,
                "max_Holevo_abs_error_vs_reference": holevo_error,
                "max_Discord_abs_error_vs_reference": discord_error,
                "estimated_max_MI_abs_error": _richardson_estimate(
                    mi_error, float(timestep), reference_dt
                ),
                "estimated_max_Holevo_abs_error": _richardson_estimate(
                    holevo_error, float(timestep), reference_dt
                ),
                "estimated_max_Discord_abs_error": _richardson_estimate(
                    discord_error, float(timestep), reference_dt
                ),
                "wall_seconds": record.get("wall_seconds"),
                "estimated_evolution_seconds": profile.get("estimated_evolution_seconds"),
                "run_id": record["run_id"],
            }
        )

    half_fragment = int(manifest["params"]["nqubits_E"]) // 2
    coherence = np.asarray([abs(rho[0, 1]) for rho in exact_rhos])
    stability = {
        "exact_system_entropy": _tail_summary(times, exact_entropies, manifest["tail_start"]),
        "exact_system_coherence_abs": _tail_summary(times, coherence, manifest["tail_start"]),
        "reference_half_environment_MI": _tail_summary(
            times,
            reference_mi[half_fragment],
            manifest["tail_start"],
        ),
        "half_environment_fragment_size": half_fragment,
    }
    return rows, stability, spectral, j_se, j_e


def _format_number(value):
    if value is None:
        return ""
    return f"{value:.6e}"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, report):
    columns = [
        ("dT", "dT"),
        ("max_norm_deviation", "max norm dev"),
        ("max_modified_norm_deviation", "modified norm dev"),
        ("max_state_infidelity", "max infidelity"),
        ("max_rhoS_trace_distance", "max D(rhoS)"),
        ("max_S_vn_abs_error", "max entropy err"),
        ("max_MI_abs_error_vs_reference", "max MI err"),
        ("max_Holevo_abs_error_vs_reference", "max Holevo err"),
        ("max_Discord_abs_error_vs_reference", "max discord err"),
        ("estimated_max_MI_abs_error", "estimated MI err"),
        ("estimated_max_Holevo_abs_error", "estimated Holevo err"),
        ("estimated_max_Discord_abs_error", "estimated discord err"),
        ("estimated_evolution_seconds", "evolution s"),
    ]
    lines = [
        f"# {report['study_id']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Exact spectral radius: `{report['spectral']['spectral_radius']:.9g}`; "
        f"leapfrog stability limit: `{report['spectral']['leapfrog_stability_dt_limit']:.9g}`.",
        "",
        "| " + " | ".join(label for _key, label in columns) + " |",
        "| " + " | ".join("---:" for _key, _label in columns) + " |",
    ]
    for row in report["results"]:
        lines.append("| " + " | ".join(_format_number(row[key]) for key, _label in columns) + " |")
    lines.extend(["", "The mutual-information error uses the finest GPU run as its reference; all other errors use exact factorized evolution.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description="Run and analyze the 17-environment-qubit Mironowicz timestep study.")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-dir", type=pathlib.Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true", help="Rerun cases whose complete artifacts already exist.")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import cupy as cp

    from pySL import run_pySL
    from src.export.path_utils import get_data_dir
    from src.export.run_store import StoredRunResult, compute_run_id
    from src.input.hamiltonian import hamiltonian
    from src.input.run_seeds import resolve_run_seed_plan_from_params
    from src.input.validation import validate_params

    manifest = _load_json(args.manifest)
    if "config_module" in manifest:
        config = importlib.import_module(manifest["config_module"]).get_config()
        base_params = {**config["params_default"], **manifest.get("params_overrides", {})}
        manifest = {**manifest, "params": base_params}
    else:
        base_params = manifest["params"]
    _validate_exact_model(base_params)
    subdir = manifest["subdir"]
    index_path = pathlib.Path(get_data_dir(subdir)) / "dt_study_runs.json"
    run_records = _load_json(index_path).get("runs", {}) if index_path.exists() else {}

    artifacts = []
    for timestep in manifest["dT_values"]:
        run_params, expected_params = _resolved_params(base_params, timestep, resolve_run_seed_plan_from_params)
        validate_params(run_params)
        expected_id = compute_run_id(expected_params)
        run_dir = pathlib.Path(get_data_dir(subdir)) / "runs" / expected_id
        complete = (run_dir / "results.npz").exists() and (run_dir / "metadata.json").exists()
        if args.analyze_only and not complete:
            raise FileNotFoundError(f"missing run for dT={timestep}: {run_dir}")
        if not args.analyze_only and (args.force or not complete):
            print(f"\n=== dT={timestep:g} ===", flush=True)
            started = time.perf_counter()
            psi, _model, resolved_params = run_pySL(params=run_params, subdir=subdir)
            wall_seconds = time.perf_counter() - started
            run_id = compute_run_id(resolved_params)
            if run_id != expected_id:
                raise RuntimeError(f"run id mismatch: expected {expected_id}, got {run_id}")
            run_records[str(timestep)] = {
                "run_id": run_id,
                "wall_seconds": wall_seconds,
                "profile": _profile_summary(psi._perf),
            }
            _write_json(index_path, {"study_id": manifest["study_id"], "runs": run_records})
        elif str(timestep) not in run_records:
            run_records[str(timestep)] = {"run_id": expected_id, "wall_seconds": None, "profile": {}}
        artifacts.append(_run_artifact(run_dir, StoredRunResult))

    seed = resolve_run_seed_plan_from_params(base_params)["run_seeds"][0]
    local_model = _local_model(base_params, seed, hamiltonian, cp)
    rows, stability, spectral, j_se, j_e = _compare_runs(manifest, artifacts, run_records, local_model)
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu_name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
    report = {
        "study_id": manifest["study_id"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": manifest["description"],
        "manifest": str(args.manifest.as_posix()),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--short")),
        "gpu": gpu_name,
        "params": base_params,
        "spectral": spectral,
        "J_SE": j_se.tolist(),
        "J_E": j_e.tolist(),
        "post_decoherence_tail": stability,
        "results": rows,
    }
    _write_json(args.report_dir / "report.json", report)
    _write_csv(args.report_dir / "report.csv", rows)
    _write_markdown(args.report_dir / "report.md", report)

    print(f"\nReport: {(args.report_dir / 'report.md').resolve()}")
    for row in rows:
        print(
            f"dT={row['dT']:>7g} norm={row['max_norm_deviation']:.3e} "
            f"infid={row['max_state_infidelity']:.3e} rhoD={row['max_rhoS_trace_distance']:.3e} "
            f"dS={row['max_S_vn_abs_error']:.3e} dMI={row['max_MI_abs_error_vs_reference']:.3e} "
            f"dHolevo={_format_number(row['max_Holevo_abs_error_vs_reference'])} "
            f"dDiscord={_format_number(row['max_Discord_abs_error_vs_reference'])} "
            f"estMI={_format_number(row['estimated_max_MI_abs_error'])}"
        )


if __name__ == "__main__":
    main()
