"""Focused higher-N state-vector validation for Paper A."""

import math
import os

from batch_configs.paper_a_common import H_SE_J, build_params


PROFILE_ENV = "PYSL_PAPER_A_HIGH_N_PROFILE"


def profile_name():
    profile = os.environ.get(PROFILE_ENV, "smoke").strip().lower()
    if profile not in ("smoke", "final"):
        raise ValueError(f"{PROFILE_ENV} must be 'smoke' or 'final', got {profile!r}")
    return profile


def validation_cases():
    return [
        {
            "validation_case": "no_field_lambda_half",
            "validation_family": "fragment_scaling",
            "psi_bias": 0.5,
            "Mironowicz_theta": math.pi / 4.0,
            "Mironowicz_h0": 0.0,
            "Mironowicz_alpha2": 0.0,
        },
        {
            "validation_case": "aligned_uniform_rescue",
            "validation_family": "field_profile",
            "field_profile": "uniform",
            "field_strength_ratio": 0.25,
            "psi_bias": 0.5,
            "Mironowicz_theta": 0.0,
            "Mironowicz_h0": 0.25 * H_SE_J,
            "Mironowicz_alpha2": 0.0,
        },
        {
            "validation_case": "aligned_random_rescue",
            "validation_family": "field_profile",
            "field_profile": "random",
            "field_strength_ratio": 0.5,
            "psi_bias": 0.5,
            "Mironowicz_theta": 0.0,
            "Mironowicz_h0": 0.0,
            "Mironowicz_alpha2": 0.5 * H_SE_J,
        },
        {
            "validation_case": "aligned_uniform_strong",
            "validation_family": "field_profile",
            "field_profile": "uniform",
            "field_strength_ratio": 30.0,
            "psi_bias": 0.5,
            "Mironowicz_theta": 0.0,
            "Mironowicz_h0": 30.0 * H_SE_J,
            "Mironowicz_alpha2": 0.0,
        },
        {
            "validation_case": "aligned_random_strong",
            "validation_family": "field_profile",
            "field_profile": "random",
            "field_strength_ratio": 30.0,
            "psi_bias": 0.5,
            "Mironowicz_theta": 0.0,
            "Mironowicz_h0": 0.0,
            "Mironowicz_alpha2": 30.0 * H_SE_J,
        },
    ]


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_HigherNValidation"
params_default = build_params(
    T=60 if _final else 2,
    extra={
        "AverageOverRunsN": 8 if _final else 1,
        "nqubits_E": 20,
        "dT": 0.002 if _final else 0.01,
        "printT": 1,
        "fragment_sample_count": 32 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)
cases = validation_cases()
analysis = {
    "purpose": "targeted N_E=20 cross-check, not a replacement for the exact factorized scaling analysis",
    "no_field_late_window": [20.0, 40.0] if _final else [0.0, 2.0],
    "field_late_window": [30.0, 60.0] if _final else [0.0, 2.0],
    "record_holevo_fraction": 0.9,
    "record_discord_fraction": 0.1,
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
