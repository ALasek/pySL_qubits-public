import numpy as np

from batch_configs.paper_a_common import analysis, build_params, profile_name, scaled_values


_profile = profile_name()
subdir = f"Paper_A_{_profile}_DisorderField"
_W = (
    scaled_values([0, 1, 3])
    if _profile == "smoke"
    else scaled_values([0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 25, 30])
)

params_default = build_params(T=60 if _profile == "final" else None, extra={"Mironowicz_h0": 0.0})
sweep = {
    "psi_bias": [0.25, 0.5],
    "Mironowicz_theta": [0.0, np.pi / 4, np.pi / 2],
    "Mironowicz_alpha2": _W,
}
analysis = analysis("Mironowicz_alpha2", "Mironowicz_theta")


def get_config():
    return {"subdir": subdir, "params_default": params_default, "sweep": sweep, "analysis": analysis}
