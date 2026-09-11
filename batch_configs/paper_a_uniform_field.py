import numpy as np

from batch_configs.paper_a_common import analysis, build_params, normalized_grid, profile_name


_profile = profile_name()
subdir = f"Paper_A_{_profile}_UniformField"
_h0 = normalized_grid(-1, 1, 3) if _profile == "smoke" else normalized_grid(-4, 4, 17)

params_default = build_params(T=60 if _profile == "final" else None, extra={"Mironowicz_alpha2": 0.0})
sweep = {
    "psi_bias": [0.25, 0.5],
    "Mironowicz_theta": [0.0, np.pi / 4, np.pi / 2],
    "Mironowicz_h0": _h0,
}
analysis = analysis("Mironowicz_h0", "Mironowicz_theta")


def get_config():
    return {"subdir": subdir, "params_default": params_default, "sweep": sweep, "analysis": analysis}
