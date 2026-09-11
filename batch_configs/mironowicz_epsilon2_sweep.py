import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_rand_epsi2Sweep"

params_default = build_batch_params_default()
params_default.update(
    {
        "H_SE_Special": "Mironowicz_rand_mod_epsilon",
        "plot_mode": "none",
    }
)

sweep = {
    "Mironowicz_epsilon2": np.linspace(0, 0.5, 21),
}

analysis = {
    "x": "Mironowicz_epsilon2",
    "y": None,
    "Tsample": -1,
    "setMinTime": 30,
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
