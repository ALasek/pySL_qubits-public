import datetime
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from contextlib import contextmanager

import numpy as np

from src.export.path_utils import get_data_dir


RUN_STORE_SCHEMA_VERSION = 2
RUN_INDEX_FILENAME = "run_index.jsonl"
RUN_INDEX_LOCK_FILENAME = "run_index.lock"
RESULTS_FILENAME = "results.npz"
PARAMS_FILENAME = "params.json"
METADATA_FILENAME = "metadata.json"
PSI_SNAPSHOTS_DIRNAME = "psi_snapshots"
PSI_SNAPSHOT_METADATA_FILENAME = "psi_snapshots.json"
MIGRATION_REPORT_FILENAME = "migration_report.json"

REQUIRED_ANALYSIS_FIELDS = [
    "I_S_Ef_fractionsT_runAv",
    "printT",
]

RESULT_FIELDS = [
    "fragment_quantile_levels",
    "fragment_sample_members",
    "I_S_Ef_fractionsT",
    "I_S_Ef_fractionsT_STD",
    "I_S_Ef_fractionsT_runAv",
    "I_S_Ef_fractionsT_STD_runAv",
    "I_S_Ef_fractionsT_runSTD",
    "I_S_Ef_fractionsT_runSEM",
    "I_S_Ef_fractionsT_fragmentSTD_runAv",
    "I_S_Ef_fractionsT_fragmentSTD_prop",
    "I_S_Ef_fractionsT_fragmentSEM_runAv",
    "I_S_Ef_fractionsT_Nsamples",
    "I_S_Ef_fractionsT_quantiles",
    "I_S_Ef_fractionsT_quantiles_runAv",
    "I_S_Ef_fractionsT_samples",
    "Holevo_Z_S_Ef_fractionsT",
    "Holevo_Z_S_Ef_fractionsT_STD",
    "Holevo_Z_S_Ef_fractionsT_runAv",
    "Holevo_Z_S_Ef_fractionsT_STD_runAv",
    "Holevo_Z_S_Ef_fractionsT_runSTD",
    "Holevo_Z_S_Ef_fractionsT_runSEM",
    "Holevo_Z_S_Ef_fractionsT_fragmentSTD_runAv",
    "Holevo_Z_S_Ef_fractionsT_fragmentSTD_prop",
    "Holevo_Z_S_Ef_fractionsT_fragmentSEM_runAv",
    "Holevo_Z_S_Ef_fractionsT_Nsamples",
    "Holevo_Z_S_Ef_fractionsT_quantiles",
    "Holevo_Z_S_Ef_fractionsT_quantiles_runAv",
    "Holevo_Z_S_Ef_fractionsT_samples",
    "Discord_Z_S_Ef_fractionsT",
    "Discord_Z_S_Ef_fractionsT_STD",
    "Discord_Z_S_Ef_fractionsT_runAv",
    "Discord_Z_S_Ef_fractionsT_STD_runAv",
    "Discord_Z_S_Ef_fractionsT_runSTD",
    "Discord_Z_S_Ef_fractionsT_runSEM",
    "Discord_Z_S_Ef_fractionsT_fragmentSTD_runAv",
    "Discord_Z_S_Ef_fractionsT_fragmentSTD_prop",
    "Discord_Z_S_Ef_fractionsT_fragmentSEM_runAv",
    "Discord_Z_S_Ef_fractionsT_Nsamples",
    "Discord_Z_S_Ef_fractionsT_quantiles",
    "Discord_Z_S_Ef_fractionsT_quantiles_runAv",
    "Discord_Z_S_Ef_fractionsT_samples",
    "S_vn_runAv",
    "S_vn_runSTD",
    "S_vn_runSEM",
    "branch_probabilities",
    "branch_probabilities_runAv",
    "branch_energy_density",
    "branch_energy_density_runAv",
    "branch_energy_density_std",
    "branch_energy_density_std_runAv",
    "branch_energy_density_gap",
    "branch_energy_density_gap_runAv",
    "branch_energy_density_gap_runSTD",
    "branch_energy_density_gap_runSEM",
    "TraceDist_fractionsT_runAv",
    "TraceDist_fractionsT_STD_runAv",
    "TraceDist_fractionsT_runSTD",
    "TraceDist_fractionsT_runSEM",
    "T",
    "dT",
    "printT",
    "nqubits",
    "nqubits_E",
    "nqubits_S",
    "nqubitsE",
    "nqubitsS",
    "rhoS_T",
    "rhoS_purity",
    "rhoS_BlochEvals",
    "rhoS_BlochVecs",
    "norms",
    "modified_norms",
    "C_correlators",
    "C_connected_correlators",
    "C_SE_correlators",
    "corr_spread_radius_T",
    "corr_spread_radius_T_runAv",
    "corr_spread_radius_T_runSTD",
    "corr_spread_radius_T_runSEM",
    "corr_offdiag_fraction_T",
    "corr_offdiag_fraction_T_runAv",
    "corr_offdiag_fraction_T_runSTD",
    "corr_offdiag_fraction_T_runSEM",
    "corr_total_connected_weight_T",
    "corr_total_connected_weight_T_runAv",
    "corr_total_connected_weight_T_runSTD",
    "corr_total_connected_weight_T_runSEM",
    "corr_nn_weight_T",
    "corr_nn_weight_T_runAv",
    "corr_nn_weight_T_runSTD",
    "corr_nn_weight_T_runSEM",
    "corr_nn_fraction_T",
    "corr_nn_fraction_T_runAv",
    "corr_nn_fraction_T_runSTD",
    "corr_nn_fraction_T_runSEM",
    "corr_commutator_spread_radius_T",
    "corr_commutator_spread_radius_T_runAv",
    "corr_commutator_spread_radius_T_runSTD",
    "corr_commutator_spread_radius_T_runSEM",
    "corr_total_commutator_weight_T",
    "corr_total_commutator_weight_T_runAv",
    "corr_total_commutator_weight_T_runSTD",
    "corr_total_commutator_weight_T_runSEM",
]


def canonicalize_value(value):
    if hasattr(value, "item"):
        return canonicalize_value(value.item())
    if hasattr(value, "tolist"):
        return canonicalize_value(value.tolist())
    if isinstance(value, tuple):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, list):
        return [canonicalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonicalize_value(value[key]) for key in sorted(value)}
    return value


def canonical_params_json(params):
    return json.dumps(canonicalize_value(params), sort_keys=True, separators=(",", ":"))


def compute_run_id(params):
    return hashlib.sha256(canonical_params_json(params).encode("utf-8")).hexdigest()


def legacy_run_filename(params):
    return hashlib.sha256(pickle.dumps(params)).hexdigest() + ".rundata"


def _run_index_path(subdir=""):
    return os.path.join(get_data_dir(subdir), RUN_INDEX_FILENAME)


def _run_index_lock_path(subdir=""):
    return os.path.join(get_data_dir(subdir), RUN_INDEX_LOCK_FILENAME)


def _runs_dir(subdir=""):
    return os.path.join(get_data_dir(subdir), "runs")


def _run_dir(subdir, run_id):
    return os.path.join(_runs_dir(subdir), run_id)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _to_numpy_value(value):
    if hasattr(value, "get"):
        value = value.get()
    if isinstance(value, (str, bytes, dict)):
        raise TypeError("not an array-like result value")
    if isinstance(value, list) and not value:
        return np.asarray(value)
    return np.asarray(value)


def _extract_results_from_psi(psi):
    results = {}
    for field in RESULT_FIELDS:
        if not hasattr(psi, field):
            continue
        value = getattr(psi, field)
        if value is None:
            continue
        try:
            results[field] = _to_numpy_value(value)
        except (TypeError, ValueError):
            continue
    for prefix in ("SBS_qre", "SBS_trace_dist", "SBS_fid", "SBS_fid_runs"):
        values = getattr(psi, prefix, None)
        if values is None:
            continue
        for fragment_size, value in enumerate(values[1:], start=1):
            if value is not None:
                results[f"{prefix}_{fragment_size}"] = np.asarray(value)
    return results


def _extract_results_from_old_tuple(data):
    params, ise, ise_std, s_vn, T, dT, printT = data
    return params, {
        "I_S_Ef_fractionsT_runAv": np.asarray(ise),
        "I_S_Ef_fractionsT_STD_runAv": np.asarray(ise_std),
        "S_vn_runAv": np.asarray(s_vn),
        "T": np.asarray(T),
        "dT": np.asarray(dT),
        "printT": np.asarray(printT),
    }


def _has_minimum_payload(results):
    return all(field in results for field in REQUIRED_ANALYSIS_FIELDS)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_index(subdir=""):
    path = _run_index_path(subdir)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_index(subdir, rows):
    out_dir = get_data_dir(subdir)
    os.makedirs(out_dir, exist_ok=True)
    with open(_run_index_path(subdir), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


@contextmanager
def _index_file_lock(subdir, timeout=120.0, stale_after=600.0):
    out_dir = get_data_dir(subdir)
    os.makedirs(out_dir, exist_ok=True)
    lock_path = _run_index_lock_path(subdir)
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > stale_after:
                    os.unlink(lock_path)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for run index lock: {lock_path}")
            time.sleep(0.05)

    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _upsert_index_row(subdir, row):
    with _index_file_lock(subdir):
        rows = [existing for existing in _read_index(subdir) if existing.get("run_id") != row["run_id"]]
        rows.append(row)
        _write_index(subdir, rows)


def _save_psi_snapshot_files(run_dir, snapshot_records):
    if not snapshot_records:
        return None

    snapshot_dir = os.path.join(run_dir, PSI_SNAPSHOTS_DIRNAME)
    os.makedirs(snapshot_dir, exist_ok=True)
    files = []
    for record in sorted(snapshot_records, key=lambda item: item.get("run_index", 0)):
        src = record.get("path")
        if not src or not os.path.exists(src):
            print(f"[WARN] Missing psi snapshot file, skipping: {src}")
            continue
        run_index = int(record.get("run_index", len(files)))
        filename = f"run_{run_index:05d}_psi_snapshots.npy"
        dest = os.path.join(snapshot_dir, filename)
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.move(src, dest)
        saved_record = {
            key: value
            for key, value in record.items()
            if key not in {"path", "filename"}
        }
        saved_record.update(
            {
                "filename": filename,
                "path": os.path.join(PSI_SNAPSHOTS_DIRNAME, filename).replace(os.sep, "/"),
                "bytes": os.path.getsize(dest),
            }
        )
        files.append(saved_record)

    if not files:
        return None

    metadata = {
        "enabled": True,
        "count": len(files),
        "format": "npy",
        "representation": "physical_psi_real_imag",
        "axis_order": ["time", "component", "basis_index"],
        "component_axis": ["real", "imag"],
        "files": files,
    }
    _write_json(os.path.join(run_dir, PSI_SNAPSHOT_METADATA_FILENAME), canonicalize_value(metadata))
    return metadata


def _save_v2_results(params, results, subdir="", metadata_extra=None, overwrite=True, snapshot_records=None):
    run_id = compute_run_id(params)
    run_dir = _run_dir(subdir, run_id)
    if os.path.exists(run_dir) and overwrite:
        for filename in (PARAMS_FILENAME, METADATA_FILENAME, RESULTS_FILENAME, PSI_SNAPSHOT_METADATA_FILENAME):
            path = os.path.join(run_dir, filename)
            if os.path.isfile(path):
                os.remove(path)
        snapshot_dir = os.path.abspath(os.path.join(run_dir, PSI_SNAPSHOTS_DIRNAME))
        resolved_run_dir = os.path.abspath(run_dir)
        if os.path.commonpath([resolved_run_dir, snapshot_dir]) != resolved_run_dir:
            raise ValueError(f"invalid psi snapshot path outside run directory: {snapshot_dir}")
        if os.path.isdir(snapshot_dir):
            shutil.rmtree(snapshot_dir)
    os.makedirs(run_dir, exist_ok=True)

    params_json = canonicalize_value(params)
    fields = sorted(results)
    psi_snapshot_metadata = _save_psi_snapshot_files(run_dir, snapshot_records)
    metadata = {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "subdir": subdir,
        "run_id": run_id,
        "status": "completed",
        "result_fields": fields,
    }
    if psi_snapshot_metadata is not None:
        metadata["psi_snapshots"] = psi_snapshot_metadata
    metadata.update(metadata_extra or {})

    _write_json(os.path.join(run_dir, PARAMS_FILENAME), params_json)
    _write_json(os.path.join(run_dir, METADATA_FILENAME), metadata)
    np.savez_compressed(os.path.join(run_dir, RESULTS_FILENAME), **results)

    row = {
        "schema_version": RUN_STORE_SCHEMA_VERSION,
        "run_id": run_id,
        "status": metadata["status"],
        "params": params_json,
        "metadata": metadata,
        "paths": {
            "params": os.path.join("runs", run_id, PARAMS_FILENAME),
            "metadata": os.path.join("runs", run_id, METADATA_FILENAME),
            "results": os.path.join("runs", run_id, RESULTS_FILENAME),
        },
    }
    if psi_snapshot_metadata is not None:
        row["paths"]["psi_snapshots"] = os.path.join("runs", run_id, PSI_SNAPSHOTS_DIRNAME)
        row["paths"]["psi_snapshot_metadata"] = os.path.join("runs", run_id, PSI_SNAPSHOT_METADATA_FILENAME)
    _upsert_index_row(subdir, row)
    return row


def save_legacy_rundata(psi, params, subdir=""):
    out_dir = get_data_dir(subdir)
    os.makedirs(out_dir, exist_ok=True)
    fname = legacy_run_filename(params)
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            pickle.dump([params, psi], f)
    return path


def save_run(psi, params, subdir="", write_legacy=True):
    results = _extract_results_from_psi(psi)
    metadata_extra = {
        key: params[key]
        for key in ("seed_input", "seed_base", "run_seeds", "seed_strategy")
        if key in params
    }
    snapshot_records = getattr(psi, "psi_snapshot_records", None)
    row = _save_v2_results(
        params,
        results,
        subdir=subdir,
        metadata_extra=metadata_extra,
        snapshot_records=snapshot_records,
    )
    legacy_path = None
    legacy_error = None
    if write_legacy:
        try:
            legacy_path = save_legacy_rundata(psi, params, subdir=subdir)
        except (MemoryError, pickle.PickleError, AttributeError, TypeError) as exc:
            legacy_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] Skipping legacy .rundata save: {legacy_error}")
    return {
        "run_id": row["run_id"],
        "row": row,
        "legacy_path": legacy_path,
        "legacy_error": legacy_error,
    }


def _float_match(left, right, tol=1e-12):
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def values_match(stored, requested):
    if isinstance(stored, (int, float)) and isinstance(requested, (int, float)):
        return _float_match(stored, requested)
    if isinstance(stored, list) and isinstance(requested, list):
        return len(stored) == len(requested) and all(
            values_match(left, right) for left, right in zip(stored, requested)
        )
    if isinstance(stored, dict) and isinstance(requested, dict):
        return set(stored) == set(requested) and all(
            values_match(stored[key], requested[key]) for key in stored
        )
    return stored == requested


def params_match(params, req_params):
    req_params = canonicalize_value(req_params)
    return all(key in params and values_match(params[key], value) for key, value in req_params.items())


class StoredRunResult:
    def __init__(self, params, results_path, metadata=None):
        self.params = params
        self.metadata = metadata or {}
        self._results_path = results_path
        self._run_dir = os.path.dirname(results_path)
        with np.load(results_path, allow_pickle=True) as loaded:
            for field in loaded.files:
                value = loaded[field]
                if value.shape == ():
                    value = value.item()
                setattr(self, field, value)

        self.nqubitsE = getattr(self, "nqubitsE", getattr(self, "nqubits_E", params.get("nqubits_E", None)))
        self.nqubitsS = getattr(self, "nqubitsS", getattr(self, "nqubits_S", params.get("nqubits_S", None)))
        if not hasattr(self, "nqubits") and self.nqubitsE is not None and self.nqubitsS is not None:
            self.nqubits = self.nqubitsE + self.nqubitsS

    def psi_snapshot_records(self):
        metadata = self.metadata.get("psi_snapshots") or {}
        return list(metadata.get("files", []))

    def _psi_snapshot_record(self, run_index=0):
        records = self.psi_snapshot_records()
        for record in records:
            if int(record.get("run_index", -1)) == int(run_index):
                return record
        if run_index == 0 and len(records) == 1:
            return records[0]
        raise KeyError(f"no psi snapshot record for run_index={run_index}")

    def load_psi_snapshots(self, run_index=0, mmap_mode="r"):
        record = self._psi_snapshot_record(run_index)
        path = os.path.join(self._run_dir, record["path"])
        return np.load(path, mmap_mode=mmap_mode)

    def get_stored_psi(self, t_idx, run_index=0):
        snapshots = self.load_psi_snapshots(run_index=run_index, mmap_mode="r")
        complex_dtype = np.complex64 if snapshots.dtype == np.float32 else np.complex128
        return snapshots[t_idx, 0].astype(complex_dtype) + 1j * snapshots[t_idx, 1].astype(complex_dtype)

    def _time_axis(self, samples):
        T = getattr(self, "T", samples - 1)
        return np.linspace(0, T, samples)

    def _fragment_axis(self, fragments):
        if self.nqubitsE:
            return np.arange(fragments) / self.nqubitsE
        return np.linspace(0, 1, fragments)

    def _plot_suffix(self):
        psi_bias = self.params.get("psi_bias", 0)
        theta = self.params.get("Mironowicz_theta", self.params.get("theta", None))
        if isinstance(theta, (int, float)):
            theta = f"{theta:.2f}"
        return f"_p={psi_bias}_theta={theta}"

    def plot_I_S(self):
        import matplotlib.pyplot as plt

        os.makedirs("figs", exist_ok=True)

        if hasattr(self, "S_vn_runAv"):
            S_vn = np.asarray(self.S_vn_runAv)
            Tplot_s = self._time_axis(S_vn.shape[0])
            title = f"Ent_VnS_N={self.nqubits}"
            plt.plot(Tplot_s, S_vn)
            plt.title(title)
            plt.ylabel("Ent_VnS")
            plt.xlabel("T")
            plt.savefig(os.path.join("figs", title + self._plot_suffix() + ".jpg"), dpi=300)
            plt.show()

        I_SE_tf = np.asarray(self.I_S_Ef_fractionsT_runAv)
        Tplot = self._time_axis(I_SE_tf.shape[1])
        Fplot = self._fragment_axis(I_SE_tf.shape[0])
        X, Y = np.meshgrid(Tplot, Fplot)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, I_SE_tf, cmap="viridis", edgecolor="none")
        title = f"ISE_3Dplot_N={self.nqubits}"
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.title(title)
        ax.set_xlabel("T")
        ax.set_ylabel("Frac")
        ax.set_zlabel("I(S:E)")
        plt.savefig(os.path.join("figs", title + self._plot_suffix() + ".jpg"), dpi=300)
        plt.show()

    def _sample_index(self, T):
        if T < 0:
            return -1
        return T

    def redundancy_fractionScore_at_T(self, T, cutoff=0.5, minfraction=0.4, minTime=0):
        if T < 0:
            return self.redundancy_fractionScore(cutoff, minfraction, minTime=minTime)

        I_SE_tf = self.I_S_Ef_fractionsT_runAv
        fragments = I_SE_tf.shape[0]
        Xdata = np.linspace(0, 1, fragments)
        it = int(T)
        curve = I_SE_tf[:, it]
        curve_max = max(curve)
        global_max = max(I_SE_tf.ravel())
        if curve_max <= cutoff * global_max or it < minTime:
            return [1, -1, -1, -1]

        degree = min(6, max(1, fragments - 1))
        ys = np.poly1d(np.polyfit(Xdata, curve, degree))(Xdata)
        ratios = ys / max(ys)
        frac_index = int(np.argmax(ratios > minfraction)) if np.any(ratios > minfraction) else -1
        return [Xdata[frac_index], it, curve_max, curve_max / global_max]

    def redundancy_fractionScore(self, cutoff=0.5, minfraction=0.4, minTime=0):
        I_SE_tf = self.I_S_Ef_fractionsT_runAv
        times = []
        fracs = []
        fragments = I_SE_tf.shape[0]
        Xdata = np.linspace(0, 1, fragments)

        for it in range(I_SE_tf.shape[1]):
            curve = I_SE_tf[:, it]
            if (max(curve) > cutoff * max(I_SE_tf.ravel())) and it >= minTime:
                degree = min(6, max(1, fragments - 1))
                coeffs = np.polyfit(Xdata, curve, degree)
                poly = np.poly1d(coeffs)
                ys = poly(Xdata)
                frac = np.argmax(ys / max(ys) > minfraction) if np.any(ys > minfraction) else -1
                times.append(it)
                fracs.append(Xdata[frac])

        if fracs:
            best = fracs.index(min(fracs))
            return [min(fracs), times[best], max(curve), max(curve) / max(I_SE_tf.ravel())]
        return [1, -1, -1, -1]

    def redundancy_slope_at_T(self, T, cutoff=0.5, minTime=0):
        if T == -1:
            return self.max_redundancy_slope(cutoff, minTime=minTime)

        it = T
        I_SE_tf = self.I_S_Ef_fractionsT_runAv
        fragments = I_SE_tf.shape[0]
        Xdata = np.linspace(0, 1, fragments)
        curve = I_SE_tf[:, it]

        if (max(curve) > cutoff * max(I_SE_tf.ravel())) and it >= minTime:
            degree = min(6, max(1, fragments - 1))
            coeffs = np.polyfit(Xdata, curve, degree)
            poly_derivative = np.polyder(np.poly1d(coeffs))
            slope = abs(poly_derivative(0.5)) / max(curve)
            return [slope, it, max(curve), max(curve) / max(I_SE_tf.ravel()), coeffs]
        return None

    def max_redundancy_slope(self, cutoff=0.5, minTime=0):
        I_SE_tf = self.I_S_Ef_fractionsT_runAv
        slopes = []
        times = []
        polyfits = []
        fragments = I_SE_tf.shape[0]
        Xdata = np.linspace(0, 1, fragments)

        for it in range(I_SE_tf.shape[1]):
            curve = I_SE_tf[:, it]
            if (max(curve) > cutoff * max(I_SE_tf.ravel())) and it >= minTime:
                degree = min(6, max(1, fragments - 1))
                coeffs = np.polyfit(Xdata, curve, degree)
                poly_derivative = np.polyder(np.poly1d(coeffs))
                slopes.append(abs(poly_derivative(0.5)) / max(curve))
                times.append(it)
                polyfits.append(coeffs)

        if slopes:
            best = slopes.index(min(slopes))
            self.QD_slope = [min(slopes), times[best], max(curve), max(curve) / max(I_SE_tf.ravel()), polyfits[best]]
        else:
            self.QD_slope = [-1, -1, -1, -1, -1]
        return self.QD_slope


def _stored_result_from_row(subdir, row):
    base = get_data_dir(subdir)
    results_path = os.path.join(base, row["paths"]["results"])
    params_path = os.path.join(base, row["paths"]["params"])
    params = _read_json(params_path) if os.path.exists(params_path) else row["params"]
    return [params, StoredRunResult(params, results_path, row.get("metadata", {}))]


def load_v2_runs_matching(req_params, subdir=""):
    matches = []
    seen = set()
    for row in _read_index(subdir):
        run_id = row.get("run_id")
        if run_id in seen:
            continue
        seen.add(run_id)
        if params_match(row.get("params", {}), req_params):
            matches.append(_stored_result_from_row(subdir, row))
    return matches


def _load_legacy_file(path, req_params, old_data=False):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if old_data:
        params = data[0]
        return data if params_match(canonicalize_value(params), req_params) else None
    params, psi = data
    return [params, psi] if params_match(canonicalize_value(params), req_params) else None


def load_legacy_runs_matching(req_params, subdir="", old_data=False):
    out_dir = get_data_dir(subdir)
    if not os.path.exists(out_dir):
        return []
    matches = []
    for fname in os.listdir(out_dir):
        if not fname.endswith(".rundata"):
            continue
        try:
            match = _load_legacy_file(os.path.join(out_dir, fname), req_params, old_data=old_data)
        except Exception as exc:
            print(f"[WARN] Could not load legacy run {fname}: {exc}")
            continue
        if match is not None:
            matches.append(match)
    return matches


def load_runs_matching(req_params, subdir="", allow_legacy=True, old_data=False):
    v2_matches = [] if old_data else load_v2_runs_matching(req_params, subdir=subdir)
    if v2_matches:
        return v2_matches
    if allow_legacy:
        return load_legacy_runs_matching(req_params, subdir=subdir, old_data=old_data)
    return []


def _legacy_payload_from_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, list) and len(data) == 2:
        params, psi = data
        return params, _extract_results_from_psi(psi), "current"
    if isinstance(data, list) and len(data) == 7:
        params, results = _extract_results_from_old_tuple(data)
        return params, results, "old_tuple"
    raise ValueError("unknown legacy rundata layout")


def migrate_legacy_subdir(subdir, dry_run=False, overwrite=False):
    out_dir = get_data_dir(subdir)
    report = {
        "subdir": subdir,
        "dry_run": dry_run,
        "migrated": [],
        "skipped": [],
    }
    if not os.path.exists(out_dir):
        report["skipped"].append({"reason": "missing_subdir", "path": out_dir})
        return report

    for fname in os.listdir(out_dir):
        if not fname.endswith(".rundata"):
            continue
        path = os.path.join(out_dir, fname)
        try:
            params, results, layout = _legacy_payload_from_file(path)
            run_id = compute_run_id(params)
            if not _has_minimum_payload(results):
                missing = [field for field in REQUIRED_ANALYSIS_FIELDS if field not in results]
                report["skipped"].append({"file": fname, "reason": "missing_fields", "missing": missing})
                continue
            if os.path.exists(_run_dir(subdir, run_id)) and not overwrite:
                report["skipped"].append({"file": fname, "run_id": run_id, "reason": "exists"})
                continue
            if not dry_run:
                _save_v2_results(
                    params,
                    results,
                    subdir=subdir,
                    metadata_extra={"migrated_from": fname, "legacy_layout": layout},
                    overwrite=overwrite,
                )
            report["migrated"].append({"file": fname, "run_id": run_id, "layout": layout})
        except Exception as exc:
            report["skipped"].append({"file": fname, "reason": "error", "error": str(exc)})

    if not dry_run:
        _write_json(os.path.join(out_dir, MIGRATION_REPORT_FILENAME), report)
    return report
