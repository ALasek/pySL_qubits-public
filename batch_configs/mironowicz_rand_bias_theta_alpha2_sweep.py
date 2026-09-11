import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_rand_bias035_theta_alpha2_sweep_randfix_LT"

params_default = build_batch_params_default()
params_default.update(
    {
        "AverageOverRunsN": 4,
        "QREmaxFragSize": 4,
        "psi_bias": 0.35,
        "seed": 1337,
        "nqubits_E": 15,
        "dT": 0.00200,
        "T": 120,
        "printT": 1,
        "plot_mode": "none",
        "H_precompute_mode": "combined_cpu", 
        "H_precompute_group_size": 8,
        "store_psi": False,
        "store_psi_on_disk": False,
        "store_correlators": False,
    }
)

sweep = {
    "Mironowicz_theta": np.linspace(0, np.pi / 4, 20),
    "Mironowicz_alpha2": np.linspace(0, 2.0, 20),
}

analysis = {
    "x": "Mironowicz_theta",
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
