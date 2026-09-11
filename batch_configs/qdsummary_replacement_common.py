from src.input.default_params import build_default_params


QUALITY_OVERRIDES = {
    "AverageOverRunsN": 3,
    "seed": 1337,
    "nqubits_E": 17,
    "dT": 0.002,
    "T": 30,
    "printT": 1,
    "plot_mode": "none",
    "H_precompute_mode": "grouped_cpu",
    "H_precompute_group_size": 8,
}


def build_replacement_params(extra=None):
    params = build_default_params()
    params.pop("save_figures", None)
    params.update(
        {
            "QREmaxFragSize": 0,
            "store_psi": False,
            "store_psi_on_disk": False,
            "store_correlators": False,
            "save_legacy_rundata": False,
            **QUALITY_OVERRIDES,
        }
    )
    if extra:
        params.update(extra)
    return params


def replacement_analysis(x, y=None, *, tsample=-1, set_min_time=10):
    analysis = {
        "plot_type": "qd_slope",
        "x": x,
        "Tsample": tsample,
        "setMinTime": set_min_time,
        "redundancyminfraction": 0.4,
        "plotData": False,
    }
    if y is not None:
        analysis["y"] = y
    return analysis
