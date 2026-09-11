import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_Epsi2QD_epsilon2_randfix"

params_default = build_replacement_params(
    {
        "H_SE_Special": "Mironowicz_rand_mod_epsilon",
        "psi_bias": 0.0,
        "Mironowicz_theta": 0.0,
        "Mironowicz_epsilon": 0.0,
    }
)

sweep = {
    "Mironowicz_epsilon2": np.linspace(0.0, 0.5, 21),
}

analysis = replacement_analysis("Mironowicz_epsilon2")


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
