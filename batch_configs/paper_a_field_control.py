import math

from batch_configs.paper_a_common import H_SE_J, build_params, profile_name


def field_cases(primary_ratios, control_ratios):
    cases = []
    geometries = (
        ("aligned", 0.5, 0.0, primary_ratios),
        ("longitudinal", 0.5, math.pi / 2.0, control_ratios),
    )
    for geometry, bias, theta, ratios in geometries:
        for ratio in ratios:
            strength = H_SE_J * float(ratio)
            profiles = ("none",) if ratio == 0 else ("uniform", "random")
            for field_profile in profiles:
                cases.append(
                    {
                        "field_geometry": geometry,
                        "field_profile": field_profile,
                        "field_strength_ratio": float(ratio),
                        "psi_bias": bias,
                        "Mironowicz_theta": theta,
                        "Mironowicz_h0": strength if field_profile == "uniform" else 0.0,
                        "Mironowicz_alpha2": strength if field_profile == "random" else 0.0,
                    }
                )
    return cases


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FieldProfileControl"
params_default = build_params(
    T=60 if _final else 8,
    extra={
        "AverageOverRunsN": 24 if _final else 1,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
    },
)
cases = field_cases(
    primary_ratios=(0, 0.25, 0.5, 1, 2, 5, 10, 20, 30) if _final else (0, 0.5, 5),
    control_ratios=(0, 0.5, 2, 10, 30) if _final else (0, 5),
)
analysis = {
    "primary_time_rule": "late_window_median",
    "late_window": [30.0, 60.0] if _final else [0.0, 8.0],
    "record_sustained_samples": 3,
    "record_holevo_fraction": 0.9,
    "record_discord_fraction": 0.1,
    "strength_scale": "field_strength_ratio = field RMS / H_SE_J; uniform uses h0, random uses Gaussian width",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
