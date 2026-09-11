"""Paired half-step convergence checks for the retained Paper-A regimes."""

import math

from batch_configs.paper_a_common import H_SE_J, build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_TimestepConvergence"
params_default = build_params(
    T=60 if _final else 2,
    extra={
        "AverageOverRunsN": 4 if _final else 1,
        "nqubits_E": 16 if _final else 6,
        "dT": 0.001 if _final else 0.005,
        "printT": 1,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)
cases = [
    {
        "convergence_case": "no_field_lambda_half",
        "reference_batch": "Paper_A_final_SubmissionMatchedLambda",
        "reference_dT": 0.002,
        "T": 40 if _final else 2,
        "psi_bias": 0.5,
        "Mironowicz_theta": math.pi / 4.0,
        "Mironowicz_h0": 0.0,
        "Mironowicz_alpha2": 0.0,
    },
    {
        "convergence_case": "aligned_random_rescue",
        "reference_batch": "Paper_A_final_FieldProfileControl",
        "reference_dT": 0.002,
        "field_profile": "random",
        "field_strength_ratio": 0.5,
        "psi_bias": 0.5,
        "Mironowicz_theta": 0.0,
        "Mironowicz_h0": 0.0,
        "Mironowicz_alpha2": 0.5 * H_SE_J,
    },
    {
        "convergence_case": "aligned_random_strong",
        "reference_batch": "Paper_A_final_FieldProfileControl",
        "reference_dT": 0.002,
        "field_profile": "random",
        "field_strength_ratio": 30.0,
        "psi_bias": 0.5,
        "Mironowicz_theta": 0.0,
        "Mironowicz_h0": 0.0,
        "Mironowicz_alpha2": 30.0 * H_SE_J,
    },
]
analysis = {
    "purpose": "paired dT=0.001 validation against retained dT=0.002 Paper-A production runs",
    "reference_realizations": 4 if _final else 1,
    "no_field_window": [20.0, 40.0] if _final else [0.0, 2.0],
    "field_window": [30.0, 60.0] if _final else [0.0, 2.0],
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
