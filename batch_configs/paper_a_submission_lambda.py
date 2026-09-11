import math

import numpy as np

from batch_configs.paper_a_common import analysis, build_params, profile_name


def alignment_lambda(p, theta):
    dot = 2.0 * math.sqrt(p * (1.0 - p)) * math.cos(theta)
    dot += (2.0 * p - 1.0) * math.sin(theta)
    return float(np.clip(1.0 - dot * dot, 0.0, 1.0))


def matched_lambda_cases(target_count):
    cases = []
    for target in np.linspace(0.0, 1.0, target_count):
        target = float(target)
        root = math.sqrt(target)
        geometries = [
            ("theta_zero", (1.0 - root) / 2.0, 0.0),
            ("p_half", 0.5, math.asin(root)),
            ("p_zero", 0.0, math.acos(root)),
        ]
        if math.isclose(target, 0.0, abs_tol=1e-15):
            geometries = [geometries[0], geometries[2]]
        elif math.isclose(target, 1.0, abs_tol=1e-15):
            geometries[2] = ("diagonal", math.cos(3.0 * math.pi / 8.0) ** 2, math.pi / 4.0)
        for family, p, theta in geometries:
            cases.append(
                {
                    "lambda_target": target,
                    "lambda_geometry": family,
                    "psi_bias": p,
                    "Mironowicz_theta": theta,
                }
            )
    return cases


_profile = profile_name()
subdir = f"Paper_A_{_profile}_SubmissionMatchedLambda"
_final = _profile == "final"
params_default = build_params(
    extra={
        "AverageOverRunsN": 24 if _final else 1,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "nqubits_E": 16 if _final else 6,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    }
)
cases = matched_lambda_cases(13 if _final else 3)
analysis = {
    **analysis("lambda_target", set_min_time=5 if _final else 0),
    "primary_time_rule": "late_window_median",
    "late_window": [20.0, 40.0] if _final else [0.0, 8.0],
    "record_sustained_samples": 3,
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
