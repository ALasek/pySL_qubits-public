"""Nonzero-mean Gaussian fields at fixed distribution RMS for Paper A."""

import math

from batch_configs.paper_a_common import H_SE_J, build_params, profile_name


_profile = profile_name()
_final = _profile == "final"
subdir = f"Paper_A_{_profile}_FieldMeanWidth"
RATIOS = (0, .05, .10, .15, .20, .25, .5, 1, 2, 5, 10, 20, 30)
ETAS = (0, .25, .5, .75, 1)
params_default = build_params(
    T=60 if _final else .02,
    extra={
        "AverageOverRunsN": 24 if _final else 1,
        "nqubits_E": 16 if _final else 6,
        "dT": .002,
        "printT": 1 if _final else .02,
        "fragment_sample_count": 64 if _final else 4,
        "fragment_quantiles": [.1, .25, .5, .9],
        "store_fragment_samples": True,
        "clampISE": 0.,
        "psi_bias": .5,
        "Mironowicz_theta": 0.,
    },
)
cases = [
    {
        "field_geometry": "aligned",
        "field_profile": "none" if ratio == 0 else "uniform" if eta == 0 else "random" if eta == 1 else "shifted_gaussian",
        "field_strength_ratio": float(ratio),
        "field_disorder_fraction": float(eta),
        "Mironowicz_h0": H_SE_J * ratio * math.sqrt(1 - eta**2),
        "Mironowicz_alpha2": H_SE_J * ratio * eta,
    }
    for ratio in (RATIOS if _final else (0, .2, 30))
    for eta in ((0,) if ratio == 0 else ETAS)
]
analysis = {
    "purpose": "separate field magnitude spread from RMS strength in the uniform/Gaussian crossover",
    "reference_batches": ["Paper_A_final_FieldProfileControl", "Paper_A_final_LowFieldRefinement"],
    "field_distribution": "h_j=h_rms*(sqrt(1-eta^2)+eta*xi_j), xi_j independent standard Gaussian",
    "rms_convention": "distribution RMS; finite realizations are not renormalized",
    "primary_time_rule": "late_window_median",
    "late_window": [30., 60.] if _final else [0., .02],
    "record_holevo_fraction": .9,
    "fragment_quantile": .1,
    "primary_metrics": ["mean_holevo", "q10_redundancy", "minimum_q10_holevo", "persistent_realization_fraction"],
    "aggregation": "fragment statistics and R=N_E/m within realization; time median of defined R; then average defined realizations and report counts",
    "persistence": "minimum over sampled window of half-environment q10 Holevo; threshold .9 bits within each realization",
    "paired_sampling": "same coupling seeds, standardized field draws, and fragment selections across cases",
    "sign_control": "At psi_bias=.5 and theta=0, independent field sign flips preserve branch-overlap magnitudes; verified with exact propagators in tests/test_paper_a_field_mean_width.py",
}


def get_config():
    return {"subdir": subdir, "params_default": params_default, "cases": cases, "analysis": analysis}
