import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_Epsi1QD_epsilon1_randfix"

params_default = build_replacement_params(
    {
        "H_SE_Special": "Mironowicz_rand_mod_epsilon",
        "psi_bias": 0.0,
        "Mironowicz_theta": 0.0,
        "Mironowicz_epsilon2": 0.0,
    }
)

sweep = {
    "Mironowicz_epsilon": np.linspace(0.0, 0.5, 21),
}

analysis = replacement_analysis("Mironowicz_epsilon")


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
