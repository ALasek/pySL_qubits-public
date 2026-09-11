import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_Stab_pi8_p_alpha2_randfix_V2"

params_default = build_replacement_params(
    {
        "dT": 1 / 1200,
        "Mironowicz_theta": np.pi / 8,
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
