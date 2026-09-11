"""
Rerun pySL references and compare their persisted scientific outputs.

Examples:
  python verify_output.py data/subdir/runs/<run_id> --atol 1e-5
  python verify_output.py --manifest benchmarks/manifest.json --profile strict
  python verify_output.py --manifest benchmarks/manifest.json --profile scientific --report data/benchmark_report.json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

from src.export.path_utils import DEFAULT_RUN_SUBDIR, get_data_dir
from src.export.run_store import StoredRunResult, compute_run_id


DEFAULT_REFERENCE = "data/7cb633820a16669bf386e3de65901102f7029f6d6a99992a9e014af9514b2f34.rundata"

COMPARE_FIELDS = [
    "S_vn_runAv",
    "S_vn_runSTD",
    "S_vn_runSEM",
    "I_S_Ef_fractionsT_runAv",
    "I_S_Ef_fractionsT_STD_runAv",
    "I_S_Ef_fractionsT_runSTD",
    "I_S_Ef_fractionsT_runSEM",
    "Holevo_Z_S_Ef_fractionsT_runAv",
    "Holevo_Z_S_Ef_fractionsT_STD_runAv",
    "Holevo_Z_S_Ef_fractionsT_runSTD",
    "Holevo_Z_S_Ef_fractionsT_runSEM",
    "Discord_Z_S_Ef_fractionsT_runAv",
    "Discord_Z_S_Ef_fractionsT_STD_runAv",
    "Discord_Z_S_Ef_fractionsT_runSTD",
    "Discord_Z_S_Ef_fractionsT_runSEM",
    "TraceDist_fractionsT_runAv",
    "TraceDist_fractionsT_STD_runAv",
    "TraceDist_fractionsT_runSTD",
    "TraceDist_fractionsT_runSEM",
    "rhoS_T",
    "rhoS_purity",
    "C_correlators",
    "C_connected_correlators",
    "C_SE_correlators",
    "corr_spread_radius_T_runAv",
    "corr_offdiag_fraction_T_runAv",
    "corr_total_connected_weight_T_runAv",
    "corr_nn_weight_T_runAv",
    "corr_nn_fraction_T_runAv",
    "norms",
    "modified_norms",
]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _v2_from_run_dir(path):
    params_path = os.path.join(path, "params.json")
    results_path = os.path.join(path, "results.npz")
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.exists(params_path) or not os.path.exists(results_path):
        raise ValueError(f"{path} is not a v2 run directory")
    params = _load_json(params_path)
    metadata = _load_json(metadata_path) if os.path.exists(metadata_path) else {}
    return params, StoredRunResult(params, results_path, metadata)


def _legacy_old_tuple_to_ref(data):
    params, ise, ise_std, s_vn, T, dT, printT = data

    class OldTupleReference:
        pass

    ref = OldTupleReference()
    ref.I_S_Ef_fractionsT_runAv = np.asarray(ise)
    ref.I_S_Ef_fractionsT_STD_runAv = np.asarray(ise_std)
    ref.S_vn_runAv = np.asarray(s_vn)
    ref.T = T
    ref.dT = dT
    ref.printT = printT
    return params, ref


def load_reference(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return _v2_from_run_dir(path)
    if path.endswith(".json") and os.path.basename(path) == "params.json":
        return _v2_from_run_dir(os.path.dirname(path))
    if path.endswith(".npz"):
        return _v2_from_run_dir(os.path.dirname(path))
    if path.endswith(".rundata"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and len(data) == 2:
            return data[0], data[1]
        if isinstance(data, list) and len(data) == 7:
            return _legacy_old_tuple_to_ref(data)
        raise ValueError(f"Unknown .rundata layout in {path}")
    raise ValueError(f"Unsupported reference path: {path}")


def describe_params(params):
    keys = ["seed", "T", "dT", "printT", "nqubits_E", "nqubits_S", "H_SE_Special"]
    parts = [f"{key}={params[key]}" for key in keys if key in params]
    return ", ".join(parts) if parts else "(no common summary params found)"


def compare_arrays_detailed(name, fresh, reference, atol, rtol=0.0):
    a = np.asarray(fresh)
    b = np.asarray(reference)
    detail = {
        "field": name,
        "fresh_shape": list(a.shape),
        "reference_shape": list(b.shape),
        "atol": float(atol),
        "rtol": float(rtol),
    }
    if a.shape != b.shape:
        detail.update(status="fail", reason="shape_mismatch")
        return detail

    try:
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        a_inf = np.isinf(a)
        b_inf = np.isinf(b)
    except TypeError:
        ok = np.array_equal(a, b)
        detail.update(status="ok" if ok else "fail", reason="exact_non_numeric_comparison")
        return detail

    nan_mismatches = int(np.count_nonzero(a_nan != b_nan))
    inf_mismatches = int(np.count_nonzero(a_inf != b_inf))
    matching_inf = a_inf & b_inf
    if np.any(matching_inf):
        inf_mismatches += int(np.count_nonzero(a[matching_inf] != b[matching_inf]))

    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(finite):
        differences = np.abs(a[finite] - b[finite])
        thresholds = atol + rtol * np.abs(b[finite])
        max_abs = float(np.max(differences))
        nonzero_reference = np.abs(b[finite]) > 0
        max_rel = (
            float(np.max(differences[nonzero_reference] / np.abs(b[finite][nonzero_reference])))
            if np.any(nonzero_reference)
            else 0.0
        )
        finite_ok = bool(np.all(differences <= thresholds))
    else:
        max_abs = 0.0
        max_rel = 0.0
        finite_ok = True

    ok = finite_ok and nan_mismatches == 0 and inf_mismatches == 0
    detail.update(
        status="ok" if ok else "fail",
        reason="within_tolerance" if ok else "value_mismatch",
        max_abs=max_abs,
        max_rel=max_rel,
        nan_mismatches=nan_mismatches,
        inf_mismatches=inf_mismatches,
    )
    return detail


def _print_comparison(detail):
    status = detail["status"].upper()
    name = detail["field"]
    if detail.get("reason") == "shape_mismatch":
        print(
            f"  {status:4s}  {name:43s}  "
            f"shape fresh={tuple(detail['fresh_shape'])} reference={tuple(detail['reference_shape'])}"
        )
        return
    if "max_abs" in detail:
        print(
            f"  {status:4s}  {name:43s}  max_abs={detail['max_abs']:.3e} "
            f"max_rel={detail['max_rel']:.3e} nan_mismatch={detail['nan_mismatches']}"
        )
        return
    print(f"  {status:4s}  {name:43s}  {detail.get('reason', '')}")


def compare_field(name, fresh, reference, tol, rtol=0.0, required=False):
    if not hasattr(fresh, name) or not hasattr(reference, name):
        missing = []
        if not hasattr(fresh, name):
            missing.append("fresh")
        if not hasattr(reference, name):
            missing.append("reference")
        status = "FAIL" if required else "SKIP"
        print(f"  {status:4s}  {name:43s}  missing from {', '.join(missing)}")
        return False if required else None

    detail = compare_arrays_detailed(name, getattr(fresh, name), getattr(reference, name), tol, rtol)
    _print_comparison(detail)
    return detail["status"] == "ok"


def compare_arrays(name, fresh, reference, tol, rtol=0.0):
    detail = compare_arrays_detailed(name, fresh, reference, tol, rtol)
    _print_comparison(detail)
    return detail["status"] == "ok"


def _import_runner():
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        from pySL import run_pySL
    finally:
        sys.argv = original_argv
    return run_pySL


def _load_fresh_result(params, subdir):
    run_dir = os.path.join(get_data_dir(subdir), "runs", compute_run_id(params))
    return _v2_from_run_dir(run_dir)[1]


def compare_reference(path, tol=1e-4, rtol=0.0, fields=None, required=False, subdir=None, label=None):
    print(f"\nLoading reference: {path}")
    ref_params, ref_result = load_reference(path)
    print(f"Reference params: {describe_params(ref_params)}")
    print("Running simulation...")
    run_pySL = _import_runner()
    kwargs = {"params": ref_params}
    if subdir is not None:
        kwargs["subdir"] = subdir
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        _psi, _H, resolved_params = run_pySL(**kwargs)
    finally:
        sys.argv = original_argv
    fresh_result = _load_fresh_result(resolved_params, subdir or DEFAULT_RUN_SUBDIR)

    print("\n=== Comparison ===")
    field_details = []
    for field in fields or COMPARE_FIELDS:
        fresh_has = hasattr(fresh_result, field)
        reference_has = hasattr(ref_result, field)
        if not fresh_has or not reference_has:
            missing = []
            if not fresh_has:
                missing.append("fresh")
            if not reference_has:
                missing.append("reference")
            status = "fail" if required else "skip"
            detail = {"field": field, "status": status, "reason": f"missing_from_{'_and_'.join(missing)}"}
            field_details.append(detail)
            print(f"  {status.upper():4s}  {field:43s}  missing from {', '.join(missing)}")
            continue
        detail = compare_arrays_detailed(
            field,
            getattr(fresh_result, field),
            getattr(ref_result, field),
            tol,
            rtol,
        )
        field_details.append(detail)
        _print_comparison(detail)

    compared = [item for item in field_details if item["status"] != "skip"]
    passed = bool(compared) and all(item["status"] == "ok" for item in compared)
    failed = sum(item["status"] == "fail" for item in field_details)
    print(f"\n{'All checks passed.' if passed else str(failed) + ' check(s) FAILED.'}")
    return {
        "case": label,
        "reference": os.path.abspath(path),
        "passed": passed,
        "atol": float(tol),
        "rtol": float(rtol),
        "fields": field_details,
    }


def _manifest_cases(manifest_path, selected=None):
    manifest = _load_json(manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    selected = set(selected or [])
    cases = []
    for case in manifest.get("cases", []):
        case_id = case["id"]
        if selected and case_id not in selected:
            continue
        reference = case.get("reference")
        if not reference:
            raise ValueError(f"benchmark case {case_id!r} has no reference")
        cases.append((case, os.path.abspath(os.path.join(manifest_dir, reference))))
    if selected:
        found = {case["id"] for case, _ in cases}
        missing = selected - found
        if missing:
            raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(missing))}")
    if not cases:
        raise ValueError("benchmark manifest selected no cases")
    return manifest, cases


def compare_manifest(manifest_path, profile="strict", selected=None, atol=None, rtol=None):
    manifest, cases = _manifest_cases(manifest_path, selected=selected)
    profiles = manifest.get("comparison_profiles", {})
    if profile not in profiles:
        raise ValueError(f"unknown comparison profile {profile!r}")
    profile_config = profiles[profile]
    effective_atol = float(profile_config["atol"] if atol is None else atol)
    effective_rtol = float(profile_config["rtol"] if rtol is None else rtol)
    fields = manifest.get("compare_fields", COMPARE_FIELDS)

    print(
        f"Benchmark suite: {manifest.get('suite_id', os.path.basename(manifest_path))}  "
        f"profile={profile} atol={effective_atol:g} rtol={effective_rtol:g}"
    )
    outcomes = []
    for case, reference in cases:
        case_id = case["id"]
        case_fields = case.get("compare_fields", fields)
        outcomes.append(
            compare_reference(
                reference,
                tol=effective_atol,
                rtol=effective_rtol,
                fields=case_fields,
                required=True,
                subdir=os.path.join("_benchmark_verify", case_id),
                label=case_id,
            )
        )
    return {
        "manifest": os.path.abspath(manifest_path),
        "suite_id": manifest.get("suite_id"),
        "profile": profile,
        "atol": effective_atol,
        "rtol": effective_rtol,
        "passed": all(item["passed"] for item in outcomes),
        "cases": outcomes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify pySL output against reference runs.")
    parser.add_argument("references", nargs="*", help="Reference .rundata files or v2 run paths.")
    parser.add_argument("--manifest", help="Tracked benchmark suite manifest.")
    parser.add_argument("--case", action="append", dest="cases", help="Manifest case id; repeatable.")
    parser.add_argument("--profile", default="strict", help="Manifest comparison profile. Default: strict.")
    parser.add_argument("--atol", "--tol", dest="atol", type=float, help="Absolute tolerance override.")
    parser.add_argument("--rtol", type=float, help="Relative tolerance override.")
    parser.add_argument("--report", help="Write a machine-readable comparison report JSON.")
    args = parser.parse_args(argv)

    if args.manifest and args.references:
        parser.error("use either positional references or --manifest, not both")

    if args.manifest:
        report = compare_manifest(
            args.manifest,
            profile=args.profile,
            selected=args.cases,
            atol=args.atol,
            rtol=args.rtol,
        )
    else:
        references = args.references or [DEFAULT_REFERENCE]
        atol = 1e-4 if args.atol is None else args.atol
        rtol = 0.0 if args.rtol is None else args.rtol
        outcomes = [compare_reference(path, tol=atol, rtol=rtol) for path in references]
        report = {
            "manifest": None,
            "profile": None,
            "atol": float(atol),
            "rtol": float(rtol),
            "passed": all(item["passed"] for item in outcomes),
            "cases": outcomes,
        }

    if args.report:
        _write_json(args.report, report)
        print(f"Comparison report: {os.path.abspath(args.report)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
