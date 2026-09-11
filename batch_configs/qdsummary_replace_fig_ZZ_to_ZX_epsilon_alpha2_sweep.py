import numpy as np

from batch_configs.qdsummary_replacement_common import (
    build_replacement_params,
    replacement_analysis,
)

subdir = "qdsummary_replace_fig_ZZ_to_ZX_epsilon_alpha2_randfix_V2"

params_default = build_replacement_params(
    {
        "H_SE_Special": "Mironowicz_rand_EE_ZZ_to_ZX",
        "psi_E_spec": "rand",
        "H_EE_J": 0.5,
        "H_EE_p": "Const",
        "store_correlators": True,
    }
)

sweep = {
    "Mironowicz_ZZ_to_ZX_epsilon": np.linspace(np.pi / 2 - 0.1, np.pi / 2 + 0.1, 20),
    "Mironowicz_alpha2": np.linspace(0.0, 3.0, 20),
}

analysis = replacement_analysis("Mironowicz_ZZ_to_ZX_epsilon", "Mironowicz_alpha2")


def get_config():
    return {
        "subdir": subdir,
        "params_default": params_default,
        "sweep": sweep,
        "analysis": analysis,
    }
