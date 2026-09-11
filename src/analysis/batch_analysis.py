import itertools
import datetime
import json
import math
import os
import pickle
import re

import numpy as np

from src.analysis.correlator_metrics import CORRELATOR_METRIC_NAMES
from src.batch_utils import load_batch_manifest
from src.export.path_utils import get_data_dir, get_repo_root


PARAMETER_LABELS = {
    "Mironowicz_theta": "theta",
    "Mironowicz_alpha2": "alpha_2",
    "Mironowicz_h0": "h_0",
    "psi_bias": "p_bias",
    "H_EE_J": "J_EE",
    "Mironowicz_epsilon": "epsilon",
    "Mironowicz_epsilon2": "epsilon_2",
    "Mironowicz_ZZ_to_ZX_epsilon": "epsilon_ZZ_to_ZX",
    "nqubits_E": "N_E",
    "PaperB_field_strength": "W",
    "PaperB_case": "Paper-B storage/control case",
}
PARAMETER_VALUE_LABELS = {
    "PaperB_case": {
        "zz_gaussian_chain": "ZZ chain | Gaussian",
        "zx_uniform_chain": "ZX chain | uniform",
        "zx_gaussian_chain": "ZX chain | Gaussian",
        "zx_quasiperiodic_chain": "ZX chain | quasiperiodic",
        "zx_binary_chain": "ZX chain | binary +/-W",
        "transverse_gaussian_chain": "composite transverse | Gaussian",
        "zx_gaussian_disjoint_all": "disjoint ZX | Gaussian, all sites",
        "zx_gaussian_disjoint_sources": "disjoint ZX | Gaussian, sources",
        "zx_gaussian_disjoint_targets": "disjoint ZX | Gaussian, targets",
    },
}

METRIC_LABELS = {
    "holevo_discord_fhalf": "Z-Holevo and Z-discord nearest F=1/2",
    "holevo_fhalf": "Z-Holevo information nearest F=1/2",
    "qd_slope": "QD slope (screening proxy)",
    "discord_fhalf": "Z-discord nearest F=1/2",
    "holevo_redundancy": "Mean-curve Holevo redundancy R_delta",
    "holevo_redundancy_q": "Mean per-realization quantile-curve estimate R_delta,q",
    "record_time": "Record time",
    "record_formation_time": "Record formation time",
    "record_lifetime": "Observed record lifetime",
    "record_occupancy": "Record-window occupancy",
    "single_site_td_mean": "Mean single-site branch trace distance",
    "single_site_td_q10": "10th-percentile single-site branch trace distance",
    "single_site_td_lifetime": "Single-site record lifetime",
    "single_site_td_occupancy": "Single-site record-window occupancy",
    "branch_energy_density_gap": "Pointer-branch storage-energy density gap",
    "branch_energy_density_width": "Largest pointer-branch storage-energy density width",
    "kappa_typ_f1": "Typical single-site branch-overlap exponent",
    "redundancy_score": "Redundancy score",
    "best_time": "Best time",
    "corr_spread_radius": "Connected correlator spread radius",
    "corr_offdiag_fraction": "Connected off-diagonal fraction",
    "corr_total_connected_weight": "Total connected correlator weight",
    "corr_nn_weight": "Nearest-neighbor connected weight",
    "corr_nn_fraction": "Nearest-neighbor fraction of off-diagonal weight",
    "corr_commutator_spread_radius": "Record-axis commutator-expectation spread radius",
    "corr_total_commutator_weight": "Record-axis commutator-expectation weight",
}

METRIC_FILENAMES = {
    "holevo_discord_fhalf": "holevo_discord_fhalf.jpg",
    "holevo_fhalf": "holevo_fhalf.jpg",
    "qd_slope": "qd_slope.jpg",
    "discord_fhalf": "discord_fhalf.jpg",
    "holevo_redundancy": "holevo_redundancy.jpg",
    "holevo_redundancy_q": "holevo_redundancy_q.jpg",
    "record_time": "record_time.jpg",
    "record_formation_time": "record_formation_time.jpg",
    "record_lifetime": "record_lifetime.jpg",
    "record_occupancy": "record_occupancy.jpg",
    "single_site_td_mean": "single_site_td_mean.jpg",
    "single_site_td_q10": "single_site_td_q10.jpg",
    "single_site_td_lifetime": "single_site_td_lifetime.jpg",
    "single_site_td_occupancy": "single_site_td_occupancy.jpg",
    "branch_energy_density_gap": "branch_energy_density_gap.jpg",
    "branch_energy_density_width": "branch_energy_density_width.jpg",
    "kappa_typ_f1": "kappa_typ_f1.jpg",
    "redundancy_score": "redundancy.jpg",
    "best_time": "best_time.jpg",
    "corr_spread_radius": "corr_spread_radius.jpg",
    "corr_offdiag_fraction": "corr_offdiag_fraction.jpg",
    "corr_total_connected_weight": "corr_connected_weight.jpg",
    "corr_nn_weight": "corr_nn_weight.jpg",
    "corr_nn_fraction": "corr_nn_fraction.jpg",
}

CORE_METRICS = (
    "holevo_fhalf",
    "discord_fhalf",
    "holevo_redundancy",
    "holevo_redundancy_q",
    "record_time",
    "record_formation_time",
    "record_lifetime",
    "record_occupancy",
    "single_site_td_mean",
    "single_site_td_q10",
    "single_site_td_lifetime",
    "single_site_td_occupancy",
    "branch_energy_density_gap",
    "branch_energy_density_width",
    "kappa_typ_f1",
    "qd_slope",
    "redundancy_score",
    "best_time",
)
DEFAULT_METRICS = CORE_METRICS + CORRELATOR_METRIC_NAMES
COMPOSITE_METRICS = {
    "holevo_discord_fhalf": ("holevo_fhalf", "discord_fhalf"),
}


def parameter_label(name):
    return PARAMETER_LABELS.get(name, name)


def metric_label(name):
    return METRIC_LABELS.get(name, name)


def load_manifest_or_config(subdir=None, config=None):
    if config is not None:
        manifest = {
            "subdir": config["subdir"],
            "sweep": config["sweep"],
            "sweep_keys": list(config["sweep"].keys()),
            "analysis": config.get("analysis", {}),
            "params_default": config.get("params_default", {}),
        }
    else:
        manifest = load_batch_manifest(subdir or "")

    if subdir is not None:
        manifest["subdir"] = subdir
    return manifest


def select_axes(manifest, x=None, y=None, filters=None):
    analysis = manifest.get("analysis", {})
    sweep = manifest["sweep"]
    x_axis = x or analysis.get("x")
    y_axis = y or analysis.get("y")

    if x_axis is None:
        candidates = [key for key, values in sweep.items() if len(values) > 1]
        if not candidates:
            candidates = list(sweep.keys())
        x_axis = candidates[0]

    if x_axis not in sweep:
        raise ValueError(f"x axis '{x_axis}' is not present in manifest sweep")
    if y_axis is not None and y_axis not in sweep:
        raise ValueError(f"y axis '{y_axis}' is not present in manifest sweep")
    if y_axis == x_axis:
        raise ValueError("x and y axes must be different")

    for key, values in sweep.items():
        if key not in {x_axis, y_axis} and len(values) > 1 and key not in (filters or {}):
            raise ValueError(
                f"sweep key '{key}' has {len(values)} values but is not selected as an axis; "
                "analyze a 1D/2D slice by choosing axes whose remaining sweep keys are singletons"
            )

    return x_axis, y_axis


def _axis_values(manifest, axis):
    return list(manifest["sweep"][axis])


def _jsonable_value(value):
    if hasattr(value, "tolist"):
        return _jsonable_value(value.tolist())
    if hasattr(value, "item"):
        return _jsonable_value(value.item())
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(value[key]) for key in sorted(value)}
    return value


def _fixed_default_filter(manifest):
    sweep_keys = set(manifest.get("sweep", {}))
    return {
        key: value
        for key, value in manifest.get("params_default", {}).items()
        if key not in sweep_keys
    }


def _make_filter(manifest, x_axis, x_value, y_axis=None, y_value=None, filters=None):
    req = _fixed_default_filter(manifest)
    req.update(filters or {})
    for key, values in manifest["sweep"].items():
        if key == x_axis:
            req[key] = x_value
        elif key == y_axis:
            req[key] = y_value
        elif len(values) == 1:
            req[key] = values[0]
    return req


def _default_loader(filter_dict, subdir, old_data=False):
    from src.export.runDataLoad import runDataLoad

    return runDataLoad(filter_dict, subdir, old_data).matchdata


def _psi_from_match(match, old_data=False):
    if old_data:
        from src.input.wavefunction import wavefunction

        psi = wavefunction(match[0], 0)
        psi.I_S_Ef_fractionsT_runAv = match[1]
        psi.I_S_Ef_fractionsT_STD_runAv = match[2]
        psi.S_vn_runAv = match[3]
        psi.T = match[4]
        psi.dT = match[5]
        psi.printT = match[6]
        return psi
    return match[1]


def _sample_index(psi, tsample):
    if tsample is None or tsample < 0:
        return -1
    return int(tsample / psi.printT)


def _series_scalar(series, index):
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        return math.nan
    if index < 0 or index >= arr.shape[0] or not math.isfinite(arr[index]):
        return math.nan
    return float(arr[index])


def _fragment_fraction_index(values, target_fraction=0.5, nqubits_E=None):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    if nqubits_E:
        fractions = np.arange(arr.shape[0], dtype=float) / float(nqubits_E)
    else:
        fractions = np.linspace(0, 1, arr.shape[0])
    return int(np.nanargmin(np.abs(fractions - target_fraction)))


def fraction_metric_at(psi, attr_name, time_index, target_fraction=0.5):
    values = getattr(psi, attr_name, None)
    if values is None:
        return math.nan
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return math.nan
    if time_index < 0 or time_index >= arr.shape[1]:
        return math.nan
    fragment_index = _fragment_fraction_index(
        arr,
        target_fraction=target_fraction,
        nqubits_E=getattr(psi, "nqubitsE", getattr(psi, "nqubits_E", None)),
    )
    if fragment_index is None:
        return math.nan
    value = arr[fragment_index, time_index]
    return float(value) if math.isfinite(float(value)) else math.nan


def fraction_metric_series(psi, attr_name, target_fraction=0.5):
    values = getattr(psi, attr_name, None)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        return None
    fragment_index = _fragment_fraction_index(
        arr,
        target_fraction=target_fraction,
        nqubits_E=getattr(psi, "nqubitsE", getattr(psi, "nqubits_E", None)),
    )
    return None if fragment_index is None else arr[fragment_index]


def _pointer_entropy_series(psi, length):
    rho_series = getattr(psi, "rhoS_T", None)
    if rho_series is None:
        return np.full(length, np.log(2), dtype=float)
    rho_series = np.asarray(rho_series)
    if rho_series.ndim != 3 or rho_series.shape[0] != length:
        return np.full(length, np.log(2), dtype=float)
    probabilities = np.real(np.diagonal(rho_series, axis1=1, axis2=2))
    totals = np.sum(probabilities, axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        totals,
        out=np.zeros_like(probabilities, dtype=float),
        where=totals > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, -probabilities * np.log(probabilities), 0.0)
    return np.sum(terms, axis=1)


def _best_record_index(psi, set_min_time):
    series = fraction_metric_series(psi, "Holevo_Z_S_Ef_fractionsT_runAv")
    if series is None:
        return -1
    start = max(0, int(math.ceil(set_min_time / psi.printT)))
    eligible = np.asarray(series, dtype=float).copy()
    eligible[:start] = np.nan
    return -1 if not np.any(np.isfinite(eligible)) else int(np.nanargmax(eligible))


def _holevo_redundancy(psi, time_index, delta, quantile=None):
    values = getattr(psi, "Holevo_Z_S_Ef_fractionsT_runAv", None)
    if values is None:
        return math.nan
    values = np.asarray(values, dtype=float)
    curve = values[:, time_index]
    if quantile is not None:
        quantile_values = getattr(psi, "Holevo_Z_S_Ef_fractionsT_quantiles_runAv", None)
        levels = getattr(psi, "fragment_quantile_levels", None)
        if quantile_values is None or levels is None:
            return math.nan
        levels = np.asarray(levels, dtype=float)
        quantile_values = np.asarray(quantile_values, dtype=float)
        if levels.size == 0 or quantile_values.ndim != 3:
            return math.nan
        curve = quantile_values[int(np.argmin(np.abs(levels - quantile))), :, time_index]
    n_environment = int(getattr(psi, "nqubitsE", getattr(psi, "nqubits_E", curve.size - 1)))
    pointer_entropy = _pointer_entropy_series(psi, values.shape[1])[time_index]
    if not math.isfinite(pointer_entropy) or pointer_entropy <= 1e-12:
        return math.nan
    threshold = (1.0 - delta) * pointer_entropy
    for fragment_size in range(1, min(curve.size, n_environment + 1)):
        if math.isfinite(curve[fragment_size]) and curve[fragment_size] >= threshold:
            return n_environment / fragment_size
    return math.nan


def _record_window_metrics(psi, delta, set_min_time):
    series = fraction_metric_series(psi, "Holevo_Z_S_Ef_fractionsT_runAv")
    if series is None:
        return math.nan, math.nan, math.nan
    series = np.asarray(series, dtype=float)
    entropy = _pointer_entropy_series(psi, series.size)
    qualifies = np.isfinite(series) & (entropy > 1e-12) & (series >= (1.0 - delta) * entropy)
    start = max(0, int(math.ceil(set_min_time / psi.printT)))
    qualifies[:start] = False
    indices = np.flatnonzero(qualifies)
    if indices.size == 0:
        return math.nan, 0.0, 0.0
    formation = int(indices[0])
    end = formation
    while end + 1 < qualifies.size and qualifies[end + 1]:
        end += 1
    lifetime = (end - formation) * psi.printT
    occupancy = float(np.mean(qualifies[start:])) if start < qualifies.size else math.nan
    return formation * psi.printT, lifetime, occupancy


def _single_site_trace_distance_series(psi, quantile=None):
    values = getattr(psi, "SBS_trace_dist_1", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return None
    values = np.where(np.isfinite(values), np.clip(values, 0.0, 1.0), np.nan)
    if quantile is None:
        return np.nanmean(values, axis=0)
    return np.nanquantile(values, quantile, axis=0)


def _single_site_record_window_metrics(psi, delta, set_min_time, quantile=0.1):
    series = _single_site_trace_distance_series(psi, quantile=quantile)
    if series is None:
        return math.nan, math.nan
    qualifies = np.isfinite(series) & (series >= 1.0 - delta)
    start = max(0, int(math.ceil(set_min_time / psi.printT)))
    qualifies[:start] = False
    indices = np.flatnonzero(qualifies)
    if indices.size == 0:
        return 0.0, 0.0
    formation = int(indices[0])
    end = formation
    while end + 1 < qualifies.size and qualifies[end + 1]:
        end += 1
    lifetime = (end - formation) * psi.printT
    occupancy = float(np.mean(qualifies[start:])) if start < qualifies.size else math.nan
    return lifetime, occupancy


def _branch_energy_metrics(psi):
    gap = getattr(psi, "branch_energy_density_gap_runAv", math.nan)
    widths = np.asarray(getattr(psi, "branch_energy_density_std_runAv", []), dtype=float)
    width = float(np.nanmax(widths)) if np.any(np.isfinite(widths)) else math.nan
    return float(gap), width


def _kappa_typ_f1(psi, time_index):
    fidelities = getattr(psi, "SBS_fid_runs_1", getattr(psi, "SBS_fid_1", None))
    if fidelities is None:
        return math.nan
    arr = np.asarray(fidelities, dtype=float)
    if arr.ndim not in (2, 3) or time_index < 0 or time_index >= arr.shape[1]:
        return math.nan
    values = arr[:, time_index] if arr.ndim == 2 else arr[:, time_index, :]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(-0.5 * np.mean(np.log(np.clip(values, 1e-300, 1.0))))


def _correlator_metrics_for_psi(psi, index):
    metrics = {}
    for name in CORRELATOR_METRIC_NAMES:
        series = getattr(psi, f"{name}_T_runAv", None)
        metrics[name] = math.nan if series is None else _series_scalar(series, index)
    return metrics


def _nan_grid(rows, cols):
    return [[math.nan for _ in range(cols)] for _ in range(rows)]


def _bool_grid(rows, cols, value=False):
    return [[value for _ in range(cols)] for _ in range(rows)]


def _transpose(grid):
    return [list(row) for row in zip(*grid)]


def _grid_has_finite_value(grid):
    for row in grid:
        for value in row:
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if finite:
                return True
    return False


def compute_metrics_for_psi(
    psi,
    tsample=-1,
    set_min_time=30,
    redundancy_min_fraction=0.4,
    holevo_delta=0.1,
    holevo_quantile=0.1,
):
    min_time = set_min_time / psi.printT
    index = _sample_index(psi, tsample)
    metric_index = index if index >= 0 else _best_record_index(psi, set_min_time)
    slope_scan = None
    if metric_index < 0:
        slope_scan = psi.redundancy_slope_at_T(-1, 0.5, minTime=min_time)
        if slope_scan is not None and slope_scan[0] != -1:
            metric_index = int(slope_scan[1])

    if metric_index < 0:
        return {
            **{metric: math.nan for metric in CORE_METRICS},
            "bad": True,
            **_correlator_metrics_for_psi(psi, -1),
        }

    slope = psi.redundancy_slope_at_T(metric_index, 0.5, minTime=min_time)
    frac = psi.redundancy_fractionScore_at_T(
        metric_index,
        cutoff=0.5,
        minfraction=redundancy_min_fraction,
        minTime=min_time,
    )
    corr_metrics = _correlator_metrics_for_psi(psi, metric_index)
    holevo_fhalf = fraction_metric_at(
        psi,
        "Holevo_Z_S_Ef_fractionsT_runAv",
        metric_index,
        target_fraction=0.5,
    )
    discord_fhalf = fraction_metric_at(
        psi,
        "Discord_Z_S_Ef_fractionsT_runAv",
        metric_index,
        target_fraction=0.5,
    )

    redundancy_score = math.nan
    if frac is not None and frac[0] not in (0, -1):
        redundancy_score = 1 / frac[0]
    formation_time, lifetime, occupancy = _record_window_metrics(psi, holevo_delta, set_min_time)
    single_site_mean_series = _single_site_trace_distance_series(psi)
    single_site_q10_series = _single_site_trace_distance_series(psi, quantile=0.1)
    single_site_lifetime, single_site_occupancy = _single_site_record_window_metrics(
        psi,
        holevo_delta,
        set_min_time,
    )
    branch_energy_gap, branch_energy_width = _branch_energy_metrics(psi)
    qd_slope = math.nan if slope is None or slope[0] == -1 else slope[0]

    return {
        "holevo_fhalf": holevo_fhalf,
        "discord_fhalf": discord_fhalf,
        "holevo_redundancy": _holevo_redundancy(psi, metric_index, holevo_delta),
        "holevo_redundancy_q": _holevo_redundancy(
            psi,
            metric_index,
            holevo_delta,
            quantile=holevo_quantile,
        ),
        "record_time": metric_index * psi.printT,
        "record_formation_time": formation_time,
        "record_lifetime": lifetime,
        "record_occupancy": occupancy,
        "single_site_td_mean": _series_scalar(single_site_mean_series, metric_index),
        "single_site_td_q10": _series_scalar(single_site_q10_series, metric_index),
        "single_site_td_lifetime": single_site_lifetime,
        "single_site_td_occupancy": single_site_occupancy,
        "branch_energy_density_gap": branch_energy_gap,
        "branch_energy_density_width": branch_energy_width,
        "kappa_typ_f1": _kappa_typ_f1(psi, metric_index),
        "qd_slope": qd_slope,
        "redundancy_score": redundancy_score,
        "best_time": metric_index * psi.printT,
        "bad": not (math.isfinite(holevo_fhalf) or math.isfinite(qd_slope)),
        **corr_metrics,
    }


def analyze_batch(
    manifest,
    x=None,
    y=None,
    tsample=None,
    set_min_time=None,
    redundancy_min_fraction=None,
    holevo_delta=None,
    holevo_quantile=None,
    filters=None,
    old_data=False,
    loader=None,
):
    analysis = manifest.get("analysis", {})
    x_axis, y_axis = select_axes(manifest, x=x, y=y, filters=filters)
    tsample = analysis.get("Tsample", -1) if tsample is None else tsample
    set_min_time = analysis.get("setMinTime", 30) if set_min_time is None else set_min_time
    redundancy_min_fraction = (
        analysis.get("redundancyminfraction", 0.4)
        if redundancy_min_fraction is None
        else redundancy_min_fraction
    )
    holevo_delta = analysis.get("holevo_delta", 0.1) if holevo_delta is None else holevo_delta
    holevo_quantile = analysis.get("holevo_quantile", 0.1) if holevo_quantile is None else holevo_quantile
    if not 0 <= holevo_delta <= 1:
        raise ValueError(f"holevo_delta must be between 0 and 1, got {holevo_delta!r}")
    if not 0 <= holevo_quantile <= 1:
        raise ValueError(f"holevo_quantile must be between 0 and 1, got {holevo_quantile!r}")
    loader = loader or _default_loader

    x_values = _axis_values(manifest, x_axis)
    y_values = _axis_values(manifest, y_axis) if y_axis else [None]
    rows = len(x_values)
    cols = len(y_values)
    matched_count = 0
    missing_count = 0
    ambiguous_count = 0
    matched_params = []
    results = {
        "bad": _bool_grid(rows, cols),
        "missing": _bool_grid(rows, cols),
    }
    for metric in DEFAULT_METRICS:
        results[metric] = _nan_grid(rows, cols)
        results[f"{metric}_bad"] = _bool_grid(rows, cols, value=True)

    for xi, yi in itertools.product(range(rows), range(cols)):
        req = _make_filter(
            manifest,
            x_axis,
            x_values[xi],
            y_axis=y_axis,
            y_value=y_values[yi],
            filters=filters,
        )
        matches = loader(req, manifest.get("subdir", ""), old_data=old_data)
        if not matches:
            missing_count += 1
            results["missing"][xi][yi] = True
            results["bad"][xi][yi] = True
            continue
        matched_count += 1
        if len(matches) > 1:
            ambiguous_count += 1
            print(f"[WARN] {len(matches)} matches for {req}; using the first")

        matched_params.append(matches[0][0])
        psi = _psi_from_match(matches[0], old_data=old_data)
        metrics = compute_metrics_for_psi(
            psi,
            tsample=tsample,
            set_min_time=set_min_time,
            redundancy_min_fraction=redundancy_min_fraction,
            holevo_delta=holevo_delta,
            holevo_quantile=holevo_quantile,
        )
        for key in DEFAULT_METRICS:
            results[key][xi][yi] = metrics[key]
            results[f"{key}_bad"][xi][yi] = not math.isfinite(metrics[key])
        results["bad"][xi][yi] = metrics["bad"]

    return {
        "manifest": manifest,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_values": x_values,
        "y_values": y_values if y_axis else None,
        "Tsample": tsample,
        "setMinTime": set_min_time,
        "redundancyminfraction": redundancy_min_fraction,
        "holevo_delta": holevo_delta,
        "holevo_quantile": holevo_quantile,
        "filters": filters or {},
        "old_data": old_data,
        "matched_runs": matched_count,
        "missing_runs": missing_count,
        "ambiguous_matches": ambiguous_count,
        "matched_params": matched_params,
        "results": results,
    }


def _figure_dir(subdir):
    out_dir = os.path.join(get_repo_root(), "data", "figs", subdir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _safe_part(value):
    text = str(value)
    text = text.replace(os.sep, "_").replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")
    return text[:80] or "unnamed"


def _compact_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _analysis_params(batch_result):
    manifest = batch_result["manifest"]
    params = dict(manifest.get("params_default", {}))
    for key, values in manifest.get("sweep", {}).items():
        if len(values) == 1:
            params[key] = values[0]
    params.update(batch_result.get("filters", {}))
    return params


def _analysis_sample_token(tsample):
    if tsample is None:
        return "default"
    try:
        if float(tsample) < 0:
            return "best"
    except (TypeError, ValueError):
        pass
    return f"ts{_compact_value(tsample)}"


def _analysis_name(batch_result):
    params = _analysis_params(batch_result)
    parts = []
    n_system = params.get("nqubits_S", 1)
    n_environment = params.get("nqubits_E")
    if n_environment is not None:
        parts.append(f"N{_compact_value(n_system + n_environment)}")
        parts.append(f"NE{_compact_value(n_environment)}")
    if "T" in params:
        parts.append(f"T{_compact_value(params['T'])}")
    parts.extend(
        [
            _analysis_sample_token(batch_result["Tsample"]),
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S"),
        ]
    )
    return _safe_part("_".join(parts))


def _analysis_bundle_dir(batch_result, analysis_name=None):
    out_dir = os.path.join(
        _figure_dir(batch_result["manifest"].get("subdir", "")),
        _safe_part(analysis_name or _analysis_name(batch_result)),
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable_value(data), f, indent=2, sort_keys=True)


def _effective_config(batch_result):
    manifest = batch_result["manifest"]
    return {
        "subdir": manifest.get("subdir", ""),
        "x": batch_result["x_axis"],
        "y": batch_result["y_axis"],
        "Tsample": batch_result["Tsample"],
        "setMinTime": batch_result["setMinTime"],
        "redundancyminfraction": batch_result["redundancyminfraction"],
        "holevo_delta": batch_result.get("holevo_delta", 0.1),
        "holevo_quantile": batch_result.get("holevo_quantile", 0.1),
        "filters": batch_result.get("filters", {}),
        "old_data": batch_result.get("old_data", False),
        "params_default": manifest.get("params_default", {}),
        "sweep": manifest.get("sweep", {}),
        "analysis": manifest.get("analysis", {}),
    }


def _write_readme(batch_result, out_dir, saved_metrics):
    manifest = batch_result["manifest"]
    filters = batch_result.get("filters", {})
    figure_files = {
        metric: METRIC_FILENAMES.get(metric, f"{_safe_part(metric)}.jpg")
        for metric in saved_metrics
    }
    interpretation = []
    if any("holevo" in metric or "discord" in metric for metric in saved_metrics):
        interpretation.extend(
            [
                "- Holevo and discord use natural-log units; one fully available qubit record has information ln(2).",
                "- Fixed-time Holevo/discord and quantile-curve Holevo redundancy are the primary record diagnostics.",
            ]
        )
    if any("lifetime" in metric or "occupancy" in metric for metric in saved_metrics):
        interpretation.append(
            "- Lifetime and occupancy use samples at or after setMinTime and report persistence in that window."
        )
    if any(metric in saved_metrics for metric in ("qd_slope", "redundancy_score")):
        interpretation.append(
            "- QD slope and the legacy redundancy score are screening proxies, not persistence observables."
        )
    if batch_result["Tsample"] == -1:
        interpretation.append(
            "- Tsample=-1 performs a post-cutoff maximum scan and should not be interpreted as late-time survival."
        )
    lines = [
        f"# {manifest.get('subdir', '')}",
        "",
        "## Analysis",
        f"- x: {batch_result['x_axis']}",
        f"- y: {batch_result['y_axis'] or 'none'}",
        f"- Tsample: {batch_result['Tsample']}",
        f"- setMinTime: {batch_result['setMinTime']}",
        f"- redundancyminfraction: {batch_result['redundancyminfraction']}",
        f"- holevo_delta: {batch_result.get('holevo_delta', 0.1)}",
        f"- holevo_quantile: {batch_result.get('holevo_quantile', 0.1)}",
        f"- matched runs: {batch_result['matched_runs']}",
        f"- missing runs: {batch_result['missing_runs']}",
        f"- ambiguous matches: {batch_result['ambiguous_matches']}",
        *(["", "## Interpretation", *interpretation] if interpretation else []),
        "",
        "## Filters",
    ]
    if filters:
        lines.extend(f"- {key}: {value}" for key, value in sorted(filters.items()))
    else:
        lines.append("- none beyond fixed manifest/config parameters")
    lines.extend(
        [
            "",
            "## Saved Metrics",
            *(f"- {metric}: {figure_files[metric]}" for metric in saved_metrics),
            "",
            "Files in this folder:",
            "- batch_manifest.json: manifest/config used to find runs",
            "- effective_config.json: analysis settings after CLI/local overrides",
            "- analysis_manifest.json: compact summary of this export",
            "- matched_params_summary.json: fixed and varied params from matched runs",
            "- matched_run_params.json: exact params for every matched run",
        ]
    )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _unique_values(values):
    seen = {}
    for value in values:
        key = json.dumps(_jsonable_value(value), sort_keys=True)
        seen.setdefault(key, value)
    return list(seen.values())


def _summarize_values(values):
    unique = _unique_values(values)
    summary = {"count": len(unique)}
    if len(unique) <= 100:
        summary["values"] = unique
        return summary

    numeric = []
    for value in unique:
        if not isinstance(value, (int, float)):
            return summary
        numeric.append(float(value))
    summary["min"] = min(numeric)
    summary["max"] = max(numeric)
    return summary


def _matched_params_summary(matched_params):
    if not matched_params:
        return {"common": {}, "varied": {}, "matched_runs": 0}

    keys = sorted(set().union(*(params.keys() for params in matched_params)))
    common = {}
    varied = {}
    missing = object()
    for key in keys:
        values = [params.get(key, missing) for params in matched_params]
        if any(value is missing for value in values):
            present_values = [value for value in values if value is not missing]
            varied[key] = {
                "count": len(_unique_values(present_values)) + 1,
                "missing_in_some_runs": True,
            }
            continue
        unique = _unique_values(values)
        if len(unique) == 1:
            common[key] = unique[0]
        else:
            varied[key] = _summarize_values(values)
    return {"common": common, "varied": varied, "matched_runs": len(matched_params)}


def _write_analysis_bundle_metadata(batch_result, out_dir, saved_metrics, paths):
    manifest = batch_result["manifest"]
    matched_params = batch_result.get("matched_params", [])
    saved_figures = [os.path.basename(path) for path in paths]
    figure_files_by_metric = {
        metric: os.path.basename(path)
        for metric, path in zip(saved_metrics, paths)
    }
    analysis_manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "subdir": manifest.get("subdir", ""),
        "output_dir": out_dir,
        "x": batch_result["x_axis"],
        "y": batch_result["y_axis"],
        "Tsample": batch_result["Tsample"],
        "setMinTime": batch_result["setMinTime"],
        "redundancyminfraction": batch_result["redundancyminfraction"],
        "holevo_delta": batch_result.get("holevo_delta", 0.1),
        "holevo_quantile": batch_result.get("holevo_quantile", 0.1),
        "filters": batch_result.get("filters", {}),
        "matched_runs": batch_result["matched_runs"],
        "missing_runs": batch_result["missing_runs"],
        "ambiguous_matches": batch_result["ambiguous_matches"],
        "saved_metrics": saved_metrics,
        "saved_figures": saved_figures,
        "figure_files_by_metric": figure_files_by_metric,
    }
    _write_json(os.path.join(out_dir, "batch_manifest.json"), manifest)
    _write_json(os.path.join(out_dir, "effective_config.json"), _effective_config(batch_result))
    _write_json(os.path.join(out_dir, "analysis_manifest.json"), analysis_manifest)
    _write_json(os.path.join(out_dir, "matched_run_params.json"), matched_params)
    _write_json(
        os.path.join(out_dir, "matched_params_summary.json"),
        _matched_params_summary(matched_params),
    )
    _write_readme(batch_result, out_dir, saved_metrics)


def _metric_filename(metric_name):
    return METRIC_FILENAMES.get(metric_name, f"{_safe_part(metric_name)}.jpg")


def _plot_axis_values(values, parameter_name=None):
    try:
        return [float(value) for value in values], None
    except (TypeError, ValueError):
        labels = PARAMETER_VALUE_LABELS.get(parameter_name, {})
        return list(range(len(values))), [labels.get(value, str(value)) for value in values]


def _set_category_ticks(axis, positions, labels, *, vertical=False):
    if labels is None:
        return
    if vertical:
        axis.set_yticks(positions, labels)
    else:
        axis.set_xticks(positions, labels, rotation=30, ha="right")


def _plot_1d(batch_result, metric_name, out_dir, plot_pane=None):
    import matplotlib.pyplot as plt

    x_values, x_labels = _plot_axis_values(batch_result["x_values"], batch_result["x_axis"])
    data = [row[0] for row in batch_result["results"][metric_name]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, data, marker="o")
    ax.set_xlabel(parameter_label(batch_result["x_axis"]))
    ax.set_ylabel(metric_label(metric_name))
    ax.set_title(
        f"{metric_label(metric_name)} vs {parameter_label(batch_result['x_axis'])} "
        f"(T={batch_result['Tsample']})"
    )
    ax.grid(True, alpha=0.3)
    _set_category_ticks(ax, x_values, x_labels)
    fig.tight_layout()
    fname = _metric_filename(metric_name)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    if plot_pane is not None:
        plot_pane.push(fig)
    plt.close(fig)
    return path


def _plot_2d(batch_result, metric_name, out_dir, plot_pane=None):
    import matplotlib.pyplot as plt

    x_values, x_labels = _plot_axis_values(batch_result["x_values"], batch_result["x_axis"])
    y_values, y_labels = _plot_axis_values(batch_result["y_values"], batch_result["y_axis"])
    data = batch_result["results"][metric_name]
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(x_values, y_values, _transpose(data), cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=ax, label=metric_label(metric_name))
    ax.set_xlabel(parameter_label(batch_result["x_axis"]))
    ax.set_ylabel(parameter_label(batch_result["y_axis"]))
    _set_category_ticks(ax, x_values, x_labels)
    _set_category_ticks(ax, y_values, y_labels, vertical=True)
    ax.set_title(
        f"{metric_label(metric_name)}: {parameter_label(batch_result['x_axis'])} "
        f"vs {parameter_label(batch_result['y_axis'])} (T={batch_result['Tsample']})"
    )

    bad = batch_result["results"].get(f"{metric_name}_bad", batch_result["results"]["bad"])
    for xi, row in enumerate(bad):
        for yi, is_bad in enumerate(row):
            if is_bad:
                ax.scatter(x_values[xi], y_values[yi], marker="x", color="red", s=60)

    fig.tight_layout()
    fname = _metric_filename(metric_name)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300)
    if plot_pane is not None:
        plot_pane.push(fig)
    plt.close(fig)
    return path


def _plot_metric_pair(batch_result, metric_name, out_dir, plot_pane=None):
    import matplotlib.pyplot as plt

    component_metrics = COMPOSITE_METRICS[metric_name]
    component_titles = ("Z-Holevo (F=1/2)", "Z-discord (F=1/2)")
    x_values, x_labels = _plot_axis_values(batch_result["x_values"], batch_result["x_axis"])
    y_values = batch_result["y_values"]
    if y_values is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
        for axis, component, title in zip(axes, component_metrics, component_titles):
            data = [row[0] for row in batch_result["results"][component]]
            axis.plot(x_values, data, marker="o")
            axis.set_xlabel(parameter_label(batch_result["x_axis"]))
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.set_ylim(0, np.log(2) * 1.02)
            _set_category_ticks(axis, x_values, x_labels)
        axes[0].set_ylabel("Information (nats)")
    else:
        y_positions, y_labels = _plot_axis_values(y_values, batch_result["y_axis"])
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(15, 5.5),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        meshes = []
        for axis, component, title in zip(axes, component_metrics, component_titles):
            mesh = axis.pcolormesh(
                x_values,
                y_positions,
                _transpose(batch_result["results"][component]),
                cmap="viridis",
                shading="auto",
                vmin=0,
                vmax=np.log(2),
            )
            meshes.append(mesh)
            axis.set_xlabel(parameter_label(batch_result["x_axis"]))
            axis.set_title(title)
            _set_category_ticks(axis, x_values, x_labels)
            _set_category_ticks(axis, y_positions, y_labels, vertical=True)
            bad = batch_result["results"].get(
                f"{component}_bad",
                batch_result["results"]["bad"],
            )
            for xi, row in enumerate(bad):
                for yi, is_bad in enumerate(row):
                    if is_bad:
                        axis.scatter(x_values[xi], y_positions[yi], marker="x", color="red", s=60)
        axes[0].set_ylabel(parameter_label(batch_result["y_axis"]))
        fig.colorbar(meshes[0], ax=axes.ravel().tolist(), label="Information (nats)", shrink=0.9)
    fig.suptitle(f"Pointer-record information at F=1/2 (T={batch_result['Tsample']})")
    path = os.path.join(out_dir, _metric_filename(metric_name))
    fig.savefig(path, dpi=300)
    if plot_pane is not None:
        plot_pane.push(fig)
    plt.close(fig)
    return path


def save_metric_plots(batch_result, metrics=None, plot_pane=None, analysis_name=None):
    metrics = metrics or DEFAULT_METRICS
    unknown = [
        metric
        for metric in metrics
        if metric not in batch_result["results"] and metric not in COMPOSITE_METRICS
    ]
    if unknown:
        raise ValueError(f"unknown analysis metrics: {unknown}")
    metrics = [
        metric
        for metric in metrics
        if (
            metric in COMPOSITE_METRICS
            and all(
                component in batch_result["results"]
                and _grid_has_finite_value(batch_result["results"][component])
                for component in COMPOSITE_METRICS[metric]
            )
        )
        or (
            metric in batch_result["results"]
            and _grid_has_finite_value(batch_result["results"][metric])
        )
    ]
    out_dir = _analysis_bundle_dir(batch_result, analysis_name=analysis_name)
    batch_result["analysis_bundle_dir"] = out_dir
    paths = []
    for metric_name in metrics:
        if metric_name in COMPOSITE_METRICS:
            paths.append(_plot_metric_pair(batch_result, metric_name, out_dir, plot_pane=plot_pane))
        elif batch_result["y_axis"] is None:
            paths.append(_plot_1d(batch_result, metric_name, out_dir, plot_pane=plot_pane))
        else:
            paths.append(_plot_2d(batch_result, metric_name, out_dir, plot_pane=plot_pane))
    _write_analysis_bundle_metadata(batch_result, out_dir, metrics, paths)
    return paths


def load_pickled_run(path):
    with open(path, "rb") as f:
        return pickle.load(f)
