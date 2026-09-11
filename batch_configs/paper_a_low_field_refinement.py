from batch_configs.paper_a_common import build_params, profile_name
from batch_configs.paper_a_field_control import field_cases


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_LowFieldRefinement"
params_default = build_params(
    T=60 if _final else 8,
    extra={
        "AverageOverRunsN": 24 if _final else 1,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [0.1, 0.5, 0.9],
        "store_fragment_samples": True,
        "Mironowicz_alpha2": 0.0,
        "Mironowicz_h0": 0.0,
        # The low-field onset must not be hidden by the old 0.01-nat cutoff.
        "clampISE": 0.0,
    },
)
cases = field_cases(primary_ratios=(0.05, 0.10, 0.15, 0.20), control_ratios=())
analysis = {
    "purpose": "resolve the aligned-field onset below h_rms/g=0.25",
    "reference_batch": "Paper_A_final_FieldProfileControl",
    "primary_time_rule": "late_window_median",
    "late_window": [30.0, 60.0] if _final else [0.0, 8.0],
    "primary_metrics": ["mean_holevo", "mean_threshold_redundancy"],
    "robustness_metrics": ["q10_holevo", "late_window_minimum_q10_holevo"],
    "record_holevo_fraction": 0.9,
    "strength_scale": "field_strength_ratio = field RMS / H_SE_J; uniform uses h0, random uses Gaussian width",
    "reference_difference": "clampISE=0 replaces the original 0.01-nat information cutoff",
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "cases": cases,
        "analysis": analysis,
    }
