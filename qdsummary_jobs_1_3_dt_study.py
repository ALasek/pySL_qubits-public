import argparse
import csv
import datetime as dt
import importlib
import json
import pathlib
import subprocess
import time

import numpy as np


DEFAULT_MANIFEST = pathlib.Path("experiments/qdsummary_jobs_1_3_dt_scan.json")
SUMMARY_METRICS = (
    "max_norm_deviation",
    "max_modified_norm_deviation",
    "max_state_infidelity",
    "max_phase_aligned_state_l2_error",
    "max_rhoS_trace_distance",
    "max_S_vn_abs_error",
)


def _load_json(path):
    with pathlib.Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _thresholds(manifest):
    calibration = manifest["acceptance_calibration"]
    report = _load_json(calibration["report"])
    row = next(item for item in report["results"] if item["dT"] == calibration["dT"])
    factor = float(calibration["multiplier"])
    return {metric: factor * float(row[metric]) for metric in calibration["metrics"]}


def _local_model(model, cp):
    h_se = cp.asnumpy(model.H_SE_matrix)
    h_e = cp.asnumpy(model.H_E_matrix)
    j_se = np.asarray(model.J_SE, dtype=float)
    j_e = np.asarray(model.J_E, dtype=float)
    branch_zero = [coupling * h_e for coupling in j_e]
    branch_one = [field * h_e + coupling * h_se[2:4, 2:4] for coupling, field in zip(j_se, j_e)]
    return branch_zero, branch_one


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


def _rho_system(state):
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


def _case_params(base, manifest, seed, bias, timestep):
    params = {
        **base,
        "AverageOverRunsN": 1,
        "seed": int(seed),
        "run_seeds": [int(seed)],
        "Mironowicz_alpha2": float(manifest["alpha2"]),
        "psi_bias": float(bias),
        "dT": float(timestep),
        "T": float(manifest["T"]),
        "printT": float(manifest["sampleT"]),
        "store_psi": False,
        "store_psi_on_disk": False,
        "store_correlators": False,
        "compute_discord": False,
        "plot_mode": "none",
        "save_legacy_rundata": False,
        "log_gpu_memory": False,
        "profile_evolution": False,
    }
    return params


def _exact_samples(eigensystems, bias, times):
    environment_state = np.array([np.sqrt(bias), np.sqrt(1.0 - bias)], dtype=np.complex128)
    states = [_exact_state(time_value, eigensystems, environment_state) for time_value in times]
    rhos = [_rho_system(state) for state in states]
    entropies = [_entropy(rho) for rho in rhos]
    return states, rhos, entropies


def _run_case(params, spectral, exact, hamiltonian, wavefunction, validate_params, cp):
    validate_params(params)
    sample_steps = int(round(params["printT"] / params["dT"]))
    total_steps = int(round(params["T"] / params["dT"]))
    exact_states, exact_rhos, exact_entropies = exact
    state = wavefunction(params, 0)
    state.start_run(0, params["seed"])
    state.buildPsi()
    state.buildPhi()
    model = hamiltonian(params, seed=params["seed"])
    state.precomputeH_cpspCombine(model)

    norm_errors = []
    modified_norm_errors = []
    infidelities = []
    aligned_errors = []
    rho_distances = []
    entropy_errors = []
    sample_index = 0
    started = time.perf_counter()

    for step in range(total_steps + 1):
        if step % sample_steps == 0:
            numerical = cp.asnumpy(state.psi_real + 1j * state.psi_imag).astype(np.complex128)
            physical_norm = float(np.linalg.norm(numerical))
            numerical /= physical_norm
            exact_state = exact_states[sample_index]
            exact_normalized = exact_state / np.linalg.norm(exact_state)
            overlap = np.vdot(exact_normalized, numerical)
            fidelity = min(1.0, float(abs(overlap) ** 2))
            phase = np.exp(-1j * np.angle(overlap)) if overlap else 1.0
            numerical_rho = _rho_system(numerical)
            norm_errors.append(abs(physical_norm - 1.0))
            modified_norm_errors.append(abs(state.modified_staggered_norm(dt=params["dT"]) - 1.0))
            infidelities.append(max(0.0, 1.0 - fidelity))
            aligned_errors.append(float(np.linalg.norm(phase * numerical - exact_normalized)))
            rho_distances.append(_trace_distance(numerical_rho, exact_rhos[sample_index]))
            entropy_errors.append(abs(_entropy(numerical_rho) - exact_entropies[sample_index]))
            sample_index += 1
        if step < total_steps:
            state.evolve_step(model, params["dT"])

    cp.cuda.Stream.null.synchronize()
    wall_seconds = time.perf_counter() - started
    row = {
        "dT": float(params["dT"]),
        "steps": total_steps,
        "spectral_radius_times_dT": spectral["spectral_radius"] * float(params["dT"]),
        "max_norm_deviation": max(norm_errors),
        "max_modified_norm_deviation": max(modified_norm_errors),
        "max_state_infidelity": max(infidelities),
        "max_phase_aligned_state_l2_error": max(aligned_errors),
        "max_rhoS_trace_distance": max(rho_distances),
        "max_S_vn_abs_error": max(entropy_errors),
        "wall_seconds": wall_seconds,
    }
    state.clearMem()
    del state, model
    cp.get_default_memory_pool().free_all_blocks()
    return row


def _key(row):
    return (
        row["phase"],
        row["job"],
        int(row["seed"]),
        float(row["psi_bias"]),
        float(row["dT"]),
    )


def _load_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def _aggregate(rows, thresholds):
    summaries = []
    for timestep in sorted({row["dT"] for row in rows}):
        selected = [row for row in rows if row["dT"] == timestep]
        summary = {"dT": timestep, "cases": len(selected)}
        for metric in SUMMARY_METRICS:
            worst = max(selected, key=lambda row: row[metric])
            summary[metric] = worst[metric]
            summary[f"{metric}_worst_case"] = {
                "job": worst["job"],
                "seed": worst["seed"],
                "psi_bias": worst["psi_bias"],
            }
        summary["accepted"] = all(summary[metric] <= limit for metric, limit in thresholds.items())
        summaries.append(summary)
    return summaries


def _accepted_max(summaries):
    accepted = [summary["dT"] for summary in summaries if summary["accepted"]]
    return max(accepted) if accepted else None


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, report):
    metrics = [
        ("max_norm_deviation", "norm"),
        ("max_state_infidelity", "infidelity"),
        ("max_phase_aligned_state_l2_error", "state L2"),
        ("max_rhoS_trace_distance", "rhoS D"),
        ("max_S_vn_abs_error", "entropy"),
    ]
    lines = [
        f"# {report['study_id']}",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Conservative common dT: `{report['recommended_common_dT']}`.",
        "",
        "Acceptance limits are twice the errors measured in the previously accepted dT=0.025 no-EE study.",
        "These exact-state recommendations gate norm, rhoS, and system entropy; the production choices are further constrained by the separate MI/Holevo/discord reports.",
        "",
    ]
    for phase in ("screen", "validation"):
        lines.extend([f"## {phase.title()}", ""])
        for job, summaries in report[f"{phase}_summary_by_job"].items():
            lines.extend(
                [
                    f"### {job}",
                    "",
                    "| dT | cases | accepted | " + " | ".join(label for _metric, label in metrics) + " |",
                    "| ---: | ---: | :---: | " + " | ".join("---:" for _metric, _label in metrics) + " |",
                ]
            )
            for summary in summaries:
                values = " | ".join(f"{summary[metric]:.6e}" for metric, _label in metrics)
                lines.append(
                    f"| {summary['dT']:.6g} | {summary['cases']} | "
                    f"{'yes' if summary['accepted'] else 'no'} | {values} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Determine an acceptable timestep for QD-summary jobs 1-3.")
    parser.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import os

    os.environ["CUDA_DEVICE"] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import cupy as cp

    from src.input.hamiltonian import hamiltonian
    from src.input.run_seeds import resolve_run_seed_plan_from_params
    from src.input.validation import validate_params
    from src.input.wavefunction import wavefunction

    manifest = _load_json(args.manifest)
    output_dir = pathlib.Path(manifest["output_dir"])
    raw_path = output_dir / "raw_rows.jsonl"
    if args.force and raw_path.exists():
        raw_path.unlink()
    rows = _load_rows(raw_path)
    completed = {_key(row) for row in rows}
    thresholds = _thresholds(manifest)
    times = np.arange(0.0, manifest["T"] + manifest["sampleT"], manifest["sampleT"])

    jobs = []
    for job_spec in manifest["jobs"]:
        config = importlib.import_module(job_spec["config"]).get_config()
        base = config["params_default"]
        seeds = resolve_run_seed_plan_from_params(base)["run_seeds"]
        spectral_by_seed = {}
        eigensystems_by_seed = {}
        for seed in seeds:
            params = _case_params(base, manifest, seed, 0.5, manifest["dT_values"][0])
            model = hamiltonian(params, seed=seed)
            branches = _local_model(model, cp)
            spectral_by_seed[seed] = _spectral_summary(*branches)
            eigensystems_by_seed[seed] = tuple(_local_eigensystems(branch) for branch in branches)
            del model
        worst_seed = max(seeds, key=lambda seed: spectral_by_seed[seed]["spectral_radius"])
        jobs.append((job_spec, base, seeds, worst_seed, spectral_by_seed, eigensystems_by_seed))

    def run_phase(phase, seeds_for_job, biases, timesteps_for_job):
        nonlocal rows, completed
        for job_spec, base, seeds, worst_seed, spectral_by_seed, eigensystems_by_seed in jobs:
            selected_seeds = seeds_for_job(seeds, worst_seed)
            for seed in selected_seeds:
                spectral = spectral_by_seed[seed]
                for bias in biases:
                    exact = _exact_samples(eigensystems_by_seed[seed], bias, times)
                    for timestep in timesteps_for_job(job_spec):
                        identity = (phase, job_spec["id"], int(seed), float(bias), float(timestep))
                        if identity in completed:
                            continue
                        print(
                            f"[{phase}] {job_spec['id']} seed={seed} bias={bias:g} dT={timestep:g}",
                            flush=True,
                        )
                        params = _case_params(base, manifest, seed, bias, timestep)
                        row = {
                            "phase": phase,
                            "job": job_spec["id"],
                            "seed": int(seed),
                            "psi_bias": float(bias),
                            "spectral_radius": spectral["spectral_radius"],
                            "stability_dt_limit": spectral["leapfrog_stability_dt_limit"],
                            **_run_case(
                                params,
                                spectral,
                                exact,
                                hamiltonian,
                                wavefunction,
                                validate_params,
                                cp,
                            ),
                        }
                        row["accepted"] = all(row[metric] <= limit for metric, limit in thresholds.items())
                        _append_row(raw_path, row)
                        rows.append(row)
                        completed.add(identity)
                        print(
                            f"  accepted={row['accepted']} norm={row['max_norm_deviation']:.3e} "
                            f"L2={row['max_phase_aligned_state_l2_error']:.3e} "
                            f"rhoD={row['max_rhoS_trace_distance']:.3e}",
                            flush=True,
                        )

    run_phase(
        "screen",
        lambda _seeds, worst_seed: [worst_seed],
        manifest["screen_bias_values"],
        lambda job_spec: job_spec.get("dT_values", manifest["dT_values"]),
    )
    screen_rows = [row for row in rows if row["phase"] == "screen"]
    screen_summary_by_job = {
        job_spec["id"]: _aggregate(
            [row for row in screen_rows if row["job"] == job_spec["id"]],
            thresholds,
        )
        for job_spec in manifest["jobs"]
    }
    screen_recommendations = {
        job: _accepted_max(summaries) for job, summaries in screen_summary_by_job.items()
    }
    if any(value is None for value in screen_recommendations.values()):
        raise RuntimeError(f"no screened timestep met the limits for: {screen_recommendations}")
    validation_dts = {}
    for job_spec in manifest["jobs"]:
        values = job_spec.get("dT_values", manifest["dT_values"])
        recommended = screen_recommendations[job_spec["id"]]
        larger = [value for value in values if value > recommended]
        validation_dts[job_spec["id"]] = [recommended] + ([min(larger)] if larger else [])

    run_phase(
        "validation",
        lambda seeds, _worst_seed: seeds,
        manifest["validation_bias_values"],
        lambda job_spec: validation_dts[job_spec["id"]],
    )
    validation_rows = [row for row in rows if row["phase"] == "validation"]
    validation_summary_by_job = {
        job_spec["id"]: _aggregate(
            [row for row in validation_rows if row["job"] == job_spec["id"]],
            thresholds,
        )
        for job_spec in manifest["jobs"]
    }
    recommendations = {
        job: _accepted_max(summaries) for job, summaries in validation_summary_by_job.items()
    }
    recommended_common = min(value for value in recommendations.values() if value is not None)

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
        "thresholds": thresholds,
        "recommended_dT_by_job": recommendations,
        "recommended_common_dT": recommended_common,
        "screen_summary_by_job": screen_summary_by_job,
        "validation_summary_by_job": validation_summary_by_job,
        "spectral_by_job_and_seed": {
            job_spec["id"]: {str(seed): spectral_by_seed[seed] for seed in seeds}
            for job_spec, _base, seeds, _worst, spectral_by_seed, _eigensystems in jobs
        },
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", report)
    _write_csv(output_dir / "report.csv", rows)
    _write_markdown(output_dir / "report.md", report)
    print(f"Report: {(output_dir / 'report.md').resolve()}")
    print(f"Recommended dT by job: {recommendations}")
    print(f"Conservative common dT: {recommended_common}")


if __name__ == "__main__":
    main()
