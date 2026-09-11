import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_preprod_Stab_pi4_p_alpha2_dt0025_N1"

params_default = build_replacement_params(
    {
        "AverageOverRunsN": 1,
        "dT": 0.0025,
        "Mironowicz_theta": np.pi / 4,
        "seed": 3212275105,
    }
)

sweep = {
    "psi_bias": np.linspace(0.0, 0.5, 20),
    "Mironowicz_alpha2": np.linspace(0.0, 3.0, 20),
}

analysis = replacement_analysis("psi_bias", "Mironowicz_alpha2")


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
