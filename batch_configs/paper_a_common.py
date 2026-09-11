"""Shared profile controls for the Paper-A simulation suite."""

import math
import os

import numpy as np

from src.input.default_params import build_default_params


PROFILE_ENV = "PYSL_PAPER_A_PROFILE"
H_SE_J = 0.1

_PROFILES = {
    "smoke": {
        "AverageOverRunsN": 1,
        "nqubits_E": 6,
        "dT": 0.01,
        "T": 8,
        "printT": 2,
        "fragment_sample_count": 4,
        "setMinTime": 0,
    },
    "final": {
        "AverageOverRunsN": 12,
        "nqubits_E": 16,
        "dT": 0.002,
        "T": 40,
        "printT": 1,
        "fragment_sample_count": 32,
        "setMinTime": 5,
    },
}


def profile_name():
    profile = os.environ.get(PROFILE_ENV, "smoke").strip().lower()
    if profile not in _PROFILES:
        raise ValueError(f"{PROFILE_ENV} must be one of {tuple(_PROFILES)}, got {profile!r}")
    return profile


def profile_settings():
    return dict(_PROFILES[profile_name()])


def build_params(*, T=None, extra=None):
    settings = profile_settings()
    params = build_default_params()
    params.pop("save_figures", None)
    params.update(
        {
            "H_SE_Special": "Mironowicz_rand",
            "H_SE_J": H_SE_J,
            "H_EE_J": 0.0,
            "QREmaxFragSize": 1,
            "seed": 1337,
            "fragment_reuse_samples": True,
            "fragment_half_only": True,
            "fragment_quantiles": [0.1, 0.5],
            "compute_discord": True,
            "store_psi": False,
            "store_psi_on_disk": False,
            "store_correlators": False,
            "save_legacy_rundata": False,
            "plot_mode": "none",
            **{key: value for key, value in settings.items() if key != "setMinTime"},
        }
    )
    if T is not None:
        params["T"] = T
    if extra:
        params.update(extra)
    return params


def analysis(x, y=None, *, set_min_time=None):
    settings = profile_settings()
    result = {
        "x": x,
        "Tsample": -1,
        "setMinTime": settings["setMinTime"] if set_min_time is None else set_min_time,
        "redundancyminfraction": 0.4,
        "holevo_delta": 0.1,
        "holevo_quantile": 0.1,
        "plotData": False,
    }
    if y is not None:
        result["y"] = y
    return result


def beta_to_p(beta_values):
    values = [math.cos(float(beta) / 2.0) ** 2 for beta in beta_values]
    return [0.0 if abs(value) < 1e-15 else 1.0 if abs(value - 1.0) < 1e-15 else value for value in values]


def normalized_grid(start, stop, count):
    return H_SE_J * np.linspace(start, stop, count)


def scaled_values(values):
    return H_SE_J * np.asarray(values, dtype=float)
