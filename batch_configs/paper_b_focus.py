"""Focused staged write-store suite for candidate Paper B."""

import os

import numpy as np

from src.input.default_params import build_default_params
from src.input.validation import PAPER_B_CASES


PROFILE_ENV = "PYSL_PAPER_B_PROFILE"
WRITE_TIME = 10.0
LATE_WINDOW_START = 20.0
PAPER_B_PLOT_METRICS = [
    "holevo_discord_fhalf",
    "holevo_redundancy_q",
    "holevo_redundancy",
    "record_lifetime",
    "record_occupancy",
    "single_site_td_q10",
    "single_site_td_lifetime",
    "single_site_td_occupancy",
    "kappa_typ_f1",
    "branch_energy_density_gap",
    "corr_commutator_spread_radius",
    "corr_total_commutator_weight",
    "qd_slope",
]

_PROFILES = {
    "smoke": {
        "AverageOverRunsN": 1,
        "nqubits_E": [6],
        "dT": 0.02,
        "T": 30.0,
        "printT": 2.0,
        "fragment_sample_count": 4,
        "field_strengths": [0.0, 0.6],
    },
    "focused": {
        "AverageOverRunsN": 3,
        "nqubits_E": [8, 10],
        "dT": 0.01,
        "T": 60.0,
        "printT": 2.0,
        "fragment_sample_count": 8,
        "field_strengths": [0.0, 0.2, 0.6],
    },
}


def profile_name():
    profile = os.environ.get(PROFILE_ENV, "smoke").strip().lower()
    if profile not in _PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {tuple(_PROFILES)}, got {profile!r}")
    return profile


def get_config():
    profile = profile_name()
    settings = _PROFILES[profile]
    params = build_default_params()
    params.update(
        {
            "AverageOverRunsN": settings["AverageOverRunsN"],
            "QREmaxFragSize": 1,
            "nqubits_E": settings["nqubits_E"][0],
            "psi_S_spec": "x+",
            "psi_E_spec": "bias",
            "psi_bias": 0.5,
            "dT": settings["dT"],
            "T": settings["T"],
            "printT": settings["printT"],
            "H_SE_Special": "PaperB_staged",
            "H_SE_J": 0.1,
            "H_SE_p": "Const",
            "Mironowicz_theta": np.pi / 2,
            "Mironowicz_H_E": "Z",
            "H_EE_J": 0.1,
            "H_EE_p": "Const",
            "PaperB_case": PAPER_B_CASES[0],
            "PaperB_field_strength": 0.0,
            "PaperB_field_normalization": "nominal",
            "evolution_protocol": "staged_write_store",
            "staged_write_time": WRITE_TIME,
            "store_correlators": True,
            "correlator_axis": "X",
            "correlator_sources": [0],
            "correlator_reference_time": WRITE_TIME,
            "compute_discord": True,
            "fragment_sample_count": settings["fragment_sample_count"],
            "fragment_reuse_samples": True,
            "fragment_half_only": True,
            "fragment_quantiles": [0.1, 0.5],
            "store_psi": False,
            "store_psi_on_disk": False,
            "H_precompute_mode": "combined_cpu",
            "seed": 1337,
            "plot_mode": "none",
            "save_figures": "off",
            "save_legacy_rundata": False,
        }
    )
    sweep = {
        "PaperB_field_strength": settings["field_strengths"],
        "PaperB_case": list(PAPER_B_CASES),
        "nqubits_E": settings["nqubits_E"],
    }
    analysis = {
        "x": "PaperB_field_strength",
        "y": "PaperB_case",
        "Tsample": settings["T"],
        "setMinTime": LATE_WINDOW_START,
        "redundancyminfraction": 0.4,
        "holevo_delta": 0.1,
        "holevo_quantile": 0.1,
        "metrics": PAPER_B_PLOT_METRICS,
        "plotData": False,
    }
    return {
        "subdir": f"Paper_B_{profile}_FocusedWriteStore",
        "params_default": params,
        "sweep": sweep,
        "analysis": analysis,
    }


_CONFIG = get_config()
subdir = _CONFIG["subdir"]
params_default = _CONFIG["params_default"]
sweep = _CONFIG["sweep"]
analysis = _CONFIG["analysis"]
