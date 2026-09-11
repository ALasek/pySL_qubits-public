"""Focused N_E=24 fixed-time validation for Paper A."""

import math

from batch_configs.paper_a_common import build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_N24FixedTimeValidation"
params_default = build_params(
    T=36 if _final else 0.02,
    extra={
        "AverageOverRunsN": 8 if _final else 1,
        "nqubits_E": 24,
        "dT": 0.002 if _final else 0.01,
        "printT": 36 if _final else 0.02,
        "fragment_sample_count": 32 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)
cases = [
    {
        "validation_case": "no_field_lambda_half",
        "validation_family": "fixed_time_scaling",
        "lambda_target": 0.5,
        "lambda_geometry": "p_half",
        "psi_bias": 0.5,
        "Mironowicz_theta": math.pi / 4.0,
    }
]
analysis = {
    "purpose": "targeted N_E=24 full-state check at the common finite-size time",
    "primary_time": 36.0 if _final else 0.02,
    "reference_environment_sizes": [16, 20],
    "record_holevo_fraction": 0.9,
    "record_discord_fraction": 0.1,
    "claim_boundary": "implementation and finite-size consistency check, not a thermodynamic fit",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
