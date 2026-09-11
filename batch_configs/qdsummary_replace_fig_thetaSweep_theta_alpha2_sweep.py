import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_thetaSweep_theta_alpha2_randfix_V2"

params_default = build_replacement_params(
    {
        "psi_bias": 0.5,
    }
)

sweep = {
    "Mironowicz_theta": np.linspace(0.0, np.pi / 2, 20),
    "Mironowicz_alpha2": np.linspace(0.0, 0.07, 20),
}

analysis = replacement_analysis("Mironowicz_theta", "Mironowicz_alpha2")


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
