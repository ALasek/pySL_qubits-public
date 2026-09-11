import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_MBL_Sweep2_TEST_lt"

params_default = build_batch_params_default()
params_default.update(
    {
        "H_SE_Special": "Mironowicz_rand_EE",
        "Mironowicz_theta": np.pi / 2,
        "H_EE_p": "Const",
        "psi_bias": 0.25,
        "T": 500,
        "printT": 20,
        "plot_mode": "none",
    }
)

sweep = {
    "Mironowicz_alpha2": np.linspace(0, 3.5, 6),
    "H_EE_J": np.linspace(0, 1, 6),
}

analysis = {
    "x": "H_EE_J",
    "y": "Mironowicz_alpha2",
    "Tsample": 500,
    "setMinTime": 20,
    "redundancyminfraction": 0.4,
    "plotData": False,
}


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
