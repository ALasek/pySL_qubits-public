"""Paper-B field-structure mechanism-discrimination suite.

The controls preserve exact field RMS while separately changing quasiperiodic
ordering, signs, and magnitude diversity. Gaussian fields provide the
continuous-disorder reference.

Run with:
    python pySLbatch.py --config batch_configs.paper_b_mechanism_discrimination --workers 1 --gpu-ids 0
"""

import numpy as np

from src.input.default_params import build_default_params
from src.input.validation import PAPER_B_MECHANISM_CASES


WRITE_TIME = 10.0
LATE_WINDOW_START = 20.0
EE_COUPLING = 0.1
FIELD_TO_EE_RATIOS = (3.0, 4.0, 5.0)
FIELD_STRENGTHS = tuple(round(ratio * EE_COUPLING, 12) for ratio in FIELD_TO_EE_RATIOS)
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


def get_config():
    params = build_default_params()
    params.update(
        {
            "AverageOverRunsN": 5,
            "QREmaxFragSize": 1,
            "nqubits_E": 8,
            "psi_S_spec": "x+",
            "psi_E_spec": "bias",
            "psi_bias": 0.5,
            "dT": 0.01,
            "T": 60.0,
            "printT": 2.0,
            "H_SE_Special": "PaperB_staged",
            "H_SE_J": 0.1,
            "H_SE_p": "Const",
            "Mironowicz_theta": np.pi / 2,
            "Mironowicz_H_E": "Z",
            "H_EE_J": EE_COUPLING,
            "H_EE_p": "Const",
            "PaperB_case": PAPER_B_MECHANISM_CASES[0],
            "PaperB_field_strength": FIELD_STRENGTHS[0],
            "PaperB_field_normalization": "exact_rms",
            "evolution_protocol": "staged_write_store",
            "staged_write_time": WRITE_TIME,
            "store_correlators": True,
            "correlator_axis": "X",
            "correlator_sources": [0],
            "correlator_reference_time": WRITE_TIME,
            "compute_discord": True,
            "fragment_sample_count": 32,
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
        "PaperB_field_strength": list(FIELD_STRENGTHS),
        "PaperB_case": list(PAPER_B_MECHANISM_CASES),
        "nqubits_E": [8, 10],
    }
    analysis = {
        "x": "PaperB_field_strength",
        "y": "PaperB_case",
        "Tsample": 60.0,
        "setMinTime": LATE_WINDOW_START,
        "redundancyminfraction": 0.4,
        "holevo_delta": 0.1,
        "holevo_quantile": 0.1,
        "metrics": PAPER_B_PLOT_METRICS,
        "plotData": False,
    }
    return {
        "subdir": "Paper_B_MechanismDiscrimination_ZX_ExactRMS",
        "params_default": params,
        "sweep": sweep,
        "analysis": analysis,
    }


_CONFIG = get_config()
subdir = _CONFIG["subdir"]
params_default = _CONFIG["params_default"]
sweep = _CONFIG["sweep"]
analysis = _CONFIG["analysis"]
