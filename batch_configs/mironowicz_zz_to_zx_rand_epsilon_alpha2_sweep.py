import numpy as np

from batch_configs.defaults import build_batch_params_default


subdir = "Mironowicz_ZZ_to_ZX_rand_epsilon_alpha2_sweep"

params_default = build_batch_params_default()
params_default.update(
    {
        "H_SE_Special": "Mironowicz_rand_EE_ZZ_to_ZX",
        "H_EE_J": 0.5,
        "H_EE_p": "Const",
        "nqubits_E": 13,
        "dT": 0.0005,
        "T": 60,
        "printT": 2,
        "psi_E_spec": "rand",
        "plot_mode": "none",
    }
)

sweep = {
    "Mironowicz_alpha2": np.linspace(0, 3, 20),
    "Mironowicz_ZZ_to_ZX_epsilon": np.linspace(np.pi/2 -0.1, np.pi/2 + 0.1, 20),
}

analysis = {
    "x": "Mironowicz_ZZ_to_ZX_epsilon",
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
