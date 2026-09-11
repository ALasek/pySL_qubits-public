import json
import os


def load_local_config(defaults, path, label):
    config, _keys = load_local_config_with_keys(defaults, path, label)
    return config


def load_local_config_with_keys(defaults, path, label):
    config = defaults.copy()
    if not os.path.exists(path):
        print(f"Using default {label} config; no local config found at {path}")
        return config, set()

    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    unknown = sorted(set(user_config) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown {label} config keys: {unknown}")

    config.update(user_config)
    print(f"Loaded {label} config from {path}")
    explicit_keys = {key for key, value in user_config.items() if value is not None}
    return config, explicit_keys


def apply_config_defaults(config, defaults, protected_keys=(), keys=None):
    result = config.copy()
    protected_keys = set(protected_keys)
    keys_to_apply = defaults if keys is None else keys
    for key in keys_to_apply:
        if key in defaults and key not in protected_keys:
            result[key] = defaults[key]
    return result
