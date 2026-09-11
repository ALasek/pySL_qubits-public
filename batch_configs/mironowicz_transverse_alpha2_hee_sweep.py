import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_transverse_alpha2_HEE_sweep"

params_default = build_batch_params_default()
params_default.update(
    {
        "H_SE_Special": "Mironowicz_rand_EE_transverse",
        "Mironowicz_theta": 0,
        "H_EE_p": "Const",
        "psi_E_spec": "rand",
        "seed": 1337,
        "nqubits_E": 13,
        "dT": 0.0005,
        "T": 60,
        "printT": 2,
        "plot_mode": "none",
    }
)

sweep = {
    "Mironowicz_alpha2": np.linspace(0, 3.5, 20),
    "H_EE_J": np.linspace(0, 1, 20),
}

analysis = {
    "x": "H_EE_J",
    "y": "Mironowicz_alpha2",
    "Tsample": -1,
    "setMinTime": 10,
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
