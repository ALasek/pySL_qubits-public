import numpy as np

from batch_configs.paper_a_common import analysis, build_params, profile_name, scaled_values


_profile = profile_name()
subdir = f"Paper_A_{_profile}_StabilityMap"
_count = 3 if _profile == "smoke" else 11
_W = (
    scaled_values([0, 1, 3])
    if _profile == "smoke"
    else scaled_values([0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 20, 30])
)

params_default = build_params(extra={"Mironowicz_h0": 0.0})
sweep = {
    "psi_bias": np.linspace(0.0, 0.5, _count),
    "Mironowicz_alpha2": _W,
    "Mironowicz_theta": [0.0, np.pi / 4, np.pi / 2],
}
analysis = analysis("psi_bias", "Mironowicz_alpha2")


def get_config():
    return {"subdir": subdir, "params_default": params_default, "sweep": sweep, "analysis": analysis}
