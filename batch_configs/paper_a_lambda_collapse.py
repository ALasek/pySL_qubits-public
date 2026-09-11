import numpy as np

from batch_configs.paper_a_common import analysis, beta_to_p, build_params, profile_name


subdir = f"Paper_A_{profile_name()}_LambdaCollapse"
_beta = [np.pi, 5 * np.pi / 6, 3 * np.pi / 4, 2 * np.pi / 3, np.pi / 2]

params_default = build_params(extra={"Mironowicz_alpha2": 0.0, "Mironowicz_h0": 0.0})
sweep = {
    "psi_bias": beta_to_p(_beta),
    "Mironowicz_theta": [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2],
    "nqubits_E": [6] if profile_name() == "smoke" else [8, 12, 16],
}
analysis = analysis("psi_bias", "Mironowicz_theta")


def get_config():
    return {"subdir": subdir, "params_default": params_default, "sweep": sweep, "analysis": analysis}
