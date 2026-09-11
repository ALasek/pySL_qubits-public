"""Record and verify the tracked pySL solver benchmark suite."""

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

import numpy as np

from src.export.path_utils import get_data_dir, get_repo_root
from src.export.run_store import compute_run_id
from src.input.param_expressions import resolve_parameter_expressions
from src.input.validation import validate_params
from verify_output import _import_runner, main as verify_main


DEFAULT_MANIFEST = os.path.join(get_repo_root(), "benchmarks", "manifest.json")
ARTIFACT_FILES = ("params.json", "metadata.json", "results.npz")
NUMERICS_FILES = (
    "pySL.py",
    "src/Qutils.py",
    "src/sparse_operator.py",
    "src/input/hamiltonian.py",
    "src/input/wavefunction.py",
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")
    os.replace(tmp_path, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numerics_hash():
    digest = hashlib.sha256()
    root = get_repo_root()
    for relative in NUMERICS_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with open(os.path.join(root, relative), "rb") as f:
            digest.update(f.read())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=get_repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_provenance():
    import cupy as cp
    import scipy

    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    return {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--short")),
        "numerics_sha256": _numerics_hash(),
        "numerics_files": list(NUMERICS_FILES),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cupy": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "cuda_driver": int(cp.cuda.runtime.driverGetVersion()),
        "gpu": gpu_name,
    }


def _array_summary(array):
    array = np.asarray(array)
    values = np.abs(array) if np.iscomplexobj(array) else array
    finite = values[np.isfinite(values)]
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "complex_magnitude": bool(np.iscomplexobj(array)),
        "nan_count": int(np.count_nonzero(np.isnan(values))),
    }
    if finite.size:
        final = values.reshape(-1)[-1]
        summary.update(
            min=float(np.min(finite)),
            max=float(np.max(finite)),
            mean=float(np.mean(finite)),
            l2=float(np.linalg.norm(finite)),
            final=float(final) if np.isfinite(final) else None,
        )
    return summary


def summarize_results(path, fields):
    summaries = {}
    with np.load(path, allow_pickle=False) as results:
        for field in fields:
            if field in results.files:
                summaries[field] = _array_summary(results[field])
    return summaries


def _select_cases(manifest, selected):
    selected = set(selected or [])
    cases = [case for case in manifest["cases"] if not selected or case["id"] in selected]
    if selected:
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(missing))}")
    return cases


def _resolved_path(manifest_path, relative):
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(manifest_path)), relative))


def _replace_artifact_dir(source, target, allowed_root, force):
    allowed_root = os.path.abspath(allowed_root)
    if os.path.commonpath([allowed_root, target]) != allowed_root:
        raise ValueError(f"refusing to write benchmark artifact outside {allowed_root}: {target}")
    if os.path.exists(target) and not force:
        raise FileExistsError(f"baseline already exists: {target}; pass --force to replace it")

    staging = f"{target}.tmp"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=True)
    for filename in ARTIFACT_FILES:
        source_path = os.path.join(source, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, os.path.join(staging, filename))
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(staging, target)


def record_suite(manifest_path, selected=None, force=False):
    manifest_path = os.path.abspath(manifest_path)
    manifest = _load_json(manifest_path)
    cases = _select_cases(manifest, selected)
    run_pySL = _import_runner()
    recorded_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for case in cases:
        case_id = case["id"]
        config_path = _resolved_path(manifest_path, case["config"])
        params = resolve_parameter_expressions(_load_json(config_path))
        validate_params(params)
        capture_subdir = os.path.join("_benchmark_capture", case_id)
        print(f"\n=== Recording benchmark: {case_id} ===")
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            _psi, _H, resolved_params = run_pySL(params=params, subdir=capture_subdir)
        finally:
            sys.argv = original_argv
        run_id = compute_run_id(resolved_params)
        source = os.path.join(get_data_dir(capture_subdir), "runs", run_id)
        target = _resolved_path(manifest_path, case["reference"])
        baseline_root = os.path.join(os.path.dirname(manifest_path), "baselines")
        _replace_artifact_dir(source, target, baseline_root, force=force)

        params_path = os.path.join(target, "params.json")
        results_path = os.path.join(target, "results.npz")
        case["baseline"] = {
            "recorded_at": recorded_at,
            "run_id": run_id,
            "params_sha256": _sha256_file(params_path),
            "results_sha256": _sha256_file(results_path),
            "typical_results": summarize_results(results_path, manifest.get("summary_fields", [])),
        }
        print(f"Baseline artifact: {target}")

    manifest["baseline_provenance"] = collect_provenance()
    _write_json(manifest_path, manifest)
    print(f"\nUpdated benchmark manifest: {manifest_path}")
    return manifest


def list_suite(manifest_path):
    manifest = _load_json(manifest_path)
    print(f"{manifest.get('suite_id', os.path.basename(manifest_path))}: {manifest.get('description', '')}")
    for case in manifest.get("cases", []):
        state = "recorded" if case.get("baseline") else "not recorded"
        print(f"  {case['id']:34s} {state:12s} {case.get('description', '')}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record or verify the tracked pySL benchmark suite.")
    parser.add_argument("command", choices=("list", "record", "verify"))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", dest="cases", help="Case id; repeatable.")
    parser.add_argument("--force", action="store_true", help="Replace existing baseline artifacts when recording.")
    parser.add_argument("--profile", default="strict", help="Verification profile. Default: strict.")
    parser.add_argument("--atol", type=float, help="Verification absolute tolerance override.")
    parser.add_argument("--rtol", type=float, help="Verification relative tolerance override.")
    parser.add_argument("--report", help="Verification report path.")
    args = parser.parse_args(argv)

    if args.command == "list":
        list_suite(args.manifest)
        return 0
    if args.command == "record":
        record_suite(args.manifest, selected=args.cases, force=args.force)
        return 0

    verify_args = ["--manifest", args.manifest, "--profile", args.profile]
    for case_id in args.cases or []:
        verify_args.extend(["--case", case_id])
    if args.atol is not None:
        verify_args.extend(["--atol", str(args.atol)])
    if args.rtol is not None:
        verify_args.extend(["--rtol", str(args.rtol)])
    if args.report:
        verify_args.extend(["--report", args.report])
    return verify_main(verify_args)


if __name__ == "__main__":
    sys.exit(main())
