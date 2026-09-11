import math

from src.input.run_seeds import coerce_seed_int


_ABS_TOL = 1e-9

PAPER_B_CASES = (
    "zz_gaussian_chain",
    "zx_uniform_chain",
    "zx_gaussian_chain",
    "zx_quasiperiodic_chain",
    "transverse_gaussian_chain",
    "zx_gaussian_disjoint_all",
    "zx_gaussian_disjoint_sources",
    "zx_gaussian_disjoint_targets",
)

PAPER_B_MATCHED_CASES = (
    "zx_uniform_chain",
    "zx_gaussian_chain",
    "zx_quasiperiodic_chain",
    "zx_binary_chain",
)
PAPER_B_MECHANISM_CASES = (
    "zx_quasiperiodic_chain",
    "zx_quasiperiodic_shuffled_chain",
    "zx_quasiperiodic_signscrambled_chain",
    "zx_two_magnitude_chain",
    "zx_gaussian_chain",
)
PAPER_B_SUPPORTED_CASES = tuple(
    dict.fromkeys(PAPER_B_CASES + PAPER_B_MATCHED_CASES + PAPER_B_MECHANISM_CASES)
)
PAPER_B_FIELD_NORMALIZATIONS = ("nominal", "exact_rms")


def _is_int_like(value):
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and math.isclose(numeric, round(numeric), rel_tol=0.0, abs_tol=_ABS_TOL)


def _require_positive_int(name, value):
    if not _is_int_like(value) or int(round(float(value))) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(round(float(value)))


def _require_non_negative_int(name, value):
    if not _is_int_like(value) or int(round(float(value))) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(round(float(value)))


def _require_positive_number(name, value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return numeric


def _require_non_negative_number(name, value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    return numeric


def _require_finite_number(name, value):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return numeric


def _require_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value


def _require_step_multiple(name, value, dT):
    steps = _require_non_negative_number(name, value) / dT
    rounded = round(steps)
    if not math.isclose(steps, rounded, rel_tol=0.0, abs_tol=_ABS_TOL):
        raise ValueError(f"{name} must be an integer multiple of dT={dT}, got {value!r}")
    return int(rounded)


def derive_time_grid(params):
    dT = _require_positive_number("dT", params["dT"])
    T = _require_non_negative_number("T", params["T"])
    printT = _require_positive_number("printT", params["printT"])

    total_steps = _require_step_multiple("T", T, dT)
    print_interval_steps = _require_step_multiple("printT", printT, dT)

    noise_interval_steps = None
    if _require_non_negative_number("noiseT", params.get("noiseT", 0)) > 0:
        noise_interval_steps = _require_step_multiple("noiseT", params["noiseT"], dT)
        if noise_interval_steps <= 0:
            raise ValueError(f"noiseT must span at least one time step, got {params['noiseT']!r}")

    return {
        "total_steps": total_steps,
        "print_interval_steps": print_interval_steps,
        "print_sample_count": total_steps // print_interval_steps + 1,
        "noise_interval_steps": noise_interval_steps,
    }


def validate_params(params):
    dtypepbits = params.get("dtypepbits")
    if dtypepbits not in (32, 64):
        raise ValueError(f"dtypepbits must be 32 or 64, got {dtypepbits!r}")

    plot_mode = params.get("plot_mode", "auto")
    if plot_mode not in ("auto", "external_blocking", "external_nonblocking", "plot_pane", "none"):
        raise ValueError(
            "plot_mode must be one of 'auto', 'external_blocking', 'external_nonblocking', 'plot_pane', or 'none', "
            f"got {plot_mode!r}"
        )

    save_figures = params.get("save_figures", "off")
    if save_figures not in ("off", "latest", "run", "both"):
        raise ValueError(
            "save_figures must be one of 'off', 'latest', 'run', or 'both', "
            f"got {save_figures!r}"
        )

    h_precompute_mode = params.get("H_precompute_mode", "combined_cpu")
    if h_precompute_mode not in ("combined_cpu", "grouped_cpu", "sum_terms"):
        raise ValueError(
            "H_precompute_mode must be one of 'combined_cpu', 'grouped_cpu', or 'sum_terms', "
            f"got {h_precompute_mode!r}"
        )

    _require_positive_int("AverageOverRunsN", params["AverageOverRunsN"])
    nqubits_S = _require_positive_int("nqubits_S", params["nqubits_S"])
    if nqubits_S != 1:
        raise ValueError(f"nqubits_S must be 1; multi-system-qubit evolution is not supported, got {nqubits_S}")
    nqubits_E = _require_positive_int("nqubits_E", params["nqubits_E"])

    qre_frag_size = _require_non_negative_int("QREmaxFragSize", params["QREmaxFragSize"])
    if qre_frag_size > nqubits_E:
        raise ValueError(
            f"QREmaxFragSize must be less than or equal to nqubits_E={nqubits_E}, got {qre_frag_size}"
        )

    _require_non_negative_int("profile_evolution_max_steps", params.get("profile_evolution_max_steps", 200))
    _require_positive_int("H_precompute_group_size", params.get("H_precompute_group_size", 8))
    _require_bool("store_psi", params.get("store_psi", False))
    _require_bool("store_psi_on_disk", params.get("store_psi_on_disk", False))
    _require_bool("store_correlators", params.get("store_correlators", True))
    _require_bool("log_gpu_memory", params.get("log_gpu_memory", True))
    _require_bool("save_legacy_rundata", params.get("save_legacy_rundata", False))
    _require_bool("compute_discord", params.get("compute_discord", True))
    _require_bool("fragment_reuse_samples", params.get("fragment_reuse_samples", False))
    store_fragment_samples = _require_bool("store_fragment_samples", params.get("store_fragment_samples", False))
    _require_bool("fragment_half_only", params.get("fragment_half_only", False))

    fragment_sample_count = params.get("fragment_sample_count")
    if fragment_sample_count is not None:
        _require_positive_int("fragment_sample_count", fragment_sample_count)
    if store_fragment_samples and fragment_sample_count is None:
        raise ValueError("store_fragment_samples requires a finite fragment_sample_count")
    if store_fragment_samples and not params.get("fragment_reuse_samples", False):
        raise ValueError("store_fragment_samples requires fragment_reuse_samples=True")

    fragment_quantiles = params.get("fragment_quantiles", [])
    if not isinstance(fragment_quantiles, (list, tuple)):
        raise ValueError(f"fragment_quantiles must be a list or tuple, got {fragment_quantiles!r}")
    for quantile in fragment_quantiles:
        numeric = _require_finite_number("fragment_quantiles entry", quantile)
        if numeric < 0 or numeric > 1:
            raise ValueError(f"fragment_quantiles entries must be between 0 and 1, got {quantile!r}")

    _require_finite_number("Mironowicz_h0", params.get("Mironowicz_h0", 0.0))

    if params.get("psi_E_spec") == "bias":
        psi_bias = _require_non_negative_number("psi_bias", params["psi_bias"])
        if psi_bias > 1:
            raise ValueError(f"psi_bias must be between 0 and 1, got {psi_bias!r}")

    p_noise_gate = _require_non_negative_number("p_noise_Gate", params.get("p_noise_Gate", 0))
    if p_noise_gate > 1:
        raise ValueError(f"p_noise_Gate must be between 0 and 1, got {p_noise_gate!r}")
    if p_noise_gate > 0:
        _require_positive_int("noiseT_Gate", params.get("noiseT_Gate", 0))

    derive_time_grid(params)

    protocol = params.get("evolution_protocol", "simultaneous")
    if protocol not in ("simultaneous", "staged_write_store"):
        raise ValueError(
            "evolution_protocol must be 'simultaneous' or 'staged_write_store', "
            f"got {protocol!r}"
        )
    if protocol == "staged_write_store":
        write_time = _require_positive_number("staged_write_time", params.get("staged_write_time", 0))
        if write_time >= float(params["T"]):
            raise ValueError(f"staged_write_time must be less than T={params['T']}, got {write_time}")
        _require_step_multiple("staged_write_time", write_time, float(params["dT"]))
        _require_step_multiple("staged_write_time", write_time, float(params["printT"]))
        if _require_non_negative_number("noiseT", params.get("noiseT", 0)) > 0 or p_noise_gate > 0:
            raise ValueError("staged_write_store does not support stochastic Hamiltonian or gate noise")

    correlator_axis = str(params.get("correlator_axis", "Z")).upper()
    if correlator_axis not in ("X", "Y", "Z"):
        raise ValueError(f"correlator_axis must be 'X', 'Y', or 'Z', got {correlator_axis!r}")
    reference_time = _require_non_negative_number(
        "correlator_reference_time", params.get("correlator_reference_time", 0)
    )
    if reference_time > float(params["T"]):
        raise ValueError(f"correlator_reference_time must not exceed T={params['T']}, got {reference_time}")
    _require_step_multiple("correlator_reference_time", reference_time, float(params["dT"]))
    _require_step_multiple("correlator_reference_time", reference_time, float(params["printT"]))

    correlator_sources = params.get("correlator_sources", "all")
    if correlator_sources != "all":
        if not isinstance(correlator_sources, (list, tuple)) or not correlator_sources:
            raise ValueError("correlator_sources must be 'all' or a non-empty list of environment indices")
        normalized_sources = [_require_non_negative_int("correlator_sources entry", value) for value in correlator_sources]
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ValueError(f"correlator_sources must not contain duplicates, got {correlator_sources!r}")
        if any(value >= nqubits_E for value in normalized_sources):
            raise ValueError(
                f"correlator_sources entries must be less than nqubits_E={nqubits_E}, got {correlator_sources!r}"
            )

    if params.get("H_SE_Special") == "PaperB_staged":
        case = params.get("PaperB_case")
        if case not in PAPER_B_SUPPORTED_CASES:
            raise ValueError(f"PaperB_case must be one of {PAPER_B_SUPPORTED_CASES}, got {case!r}")
        _require_non_negative_number("PaperB_field_strength", params.get("PaperB_field_strength", 0))
        normalization = params.get("PaperB_field_normalization", "nominal")
        if normalization not in PAPER_B_FIELD_NORMALIZATIONS:
            raise ValueError(
                "PaperB_field_normalization must be one of "
                f"{PAPER_B_FIELD_NORMALIZATIONS}, got {normalization!r}"
            )
        if protocol != "staged_write_store":
            raise ValueError("H_SE_Special='PaperB_staged' requires evolution_protocol='staged_write_store'")

    if params.get("seed") != "TIME":
        coerce_seed_int(params.get("seed"))

    return params
