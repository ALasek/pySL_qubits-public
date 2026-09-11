from src.input.default_params import build_default_params


def build_batch_params_default():
    params = build_default_params()
    params.pop("save_figures", None)
    params.update(
        {
            "QREmaxFragSize": 0,
            "dT": 0.00025,
            "seed": 1337,
            "H_noise": [[2.0, "Z"]],
            "noiseStd": 3,
            "store_correlators": False,
        }
    )
    return params
