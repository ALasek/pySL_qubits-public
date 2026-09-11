"""Direct full-fragment Holevo plateau run for Paper A."""

import math

from batch_configs.paper_a_common import analysis, build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_HolevoPlateauFullFragments"
params_default = build_params(
    extra={
        "AverageOverRunsN": 24 if _final else 1,
        "fragment_half_only": False,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "nqubits_E": 16 if _final else 6,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    }
)
cases = [
    {
        "lambda_target": 0.75,
        "lambda_geometry": "p_half",
        "psi_bias": 0.5,
        "Mironowicz_theta": math.pi / 3.0,
    }
]
analysis = {
    **analysis("lambda_target", set_min_time=5 if _final else 0),
    "purpose": "direct full-fragment Holevo plateau through m=N_E",
    "primary_time": 36.0 if _final else 8.0,
    "late_window": [20.0, 40.0] if _final else [0.0, 8.0],
    "record_holevo_fraction": 0.9,
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
