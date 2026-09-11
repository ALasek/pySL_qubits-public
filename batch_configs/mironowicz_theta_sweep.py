import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_rand_ThetaSweep"

params_default = build_batch_params_default()
params_default.update(
    {
        "H_EE_J": 0.1,
        "plot_mode": "none",
    }
)

sweep = {
    "Mironowicz_theta": np.arange(0, np.pi / 2 + 0.02, 0.02),
    "Mironowicz_alpha2": [0, 0.01, 0.05],
}

analysis = {
    "x": "Mironowicz_theta",
    "y": "Mironowicz_alpha2",
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
