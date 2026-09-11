import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_MBL3_longtime_transverse_HEE_alpha2_randfix_V2"

params_default = build_replacement_params(
    {
        "H_SE_Special": "Mironowicz_rand_EE_transverse",
        "psi_bias": 0.25,
        "Mironowicz_theta": np.pi / 2,
        "H_EE_p": "Const",
        "AverageOverRunsN": 1,
        "T": 500,
        "printT": 250,
        "store_correlators": True,
    }
)

sweep = {
    "H_EE_J": np.linspace(0.0, 1.0, 20),
    "Mironowicz_alpha2": np.linspace(0.0, 3.5, 20),
}

analysis = replacement_analysis(
    "H_EE_J",
    "Mironowicz_alpha2",
    tsample=500,
    set_min_time=20,
)


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
