import math
import time

import numpy as np


def coerce_seed_int(seed):
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer or 'TIME'")
    if isinstance(seed, float) and (not math.isfinite(seed) or not seed.is_integer()):
        raise ValueError("seed must be an integer or 'TIME'")
    try:
        numeric = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer or 'TIME'") from exc
    if numeric < 0:
        raise ValueError("seed must be non-negative")
    return numeric


def resolve_run_seed_plan(seed, run_count):
    if run_count <= 0:
        raise ValueError(f"run_count must be positive, got {run_count!r}")

    seed_input = seed
    time_seed = seed == "TIME"
    base_seed = time.time_ns() if time_seed else coerce_seed_int(seed)

    if run_count == 1:
        run_seeds = [base_seed]
    else:
        sequence = np.random.SeedSequence(base_seed)
        run_seeds = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in sequence.spawn(run_count)
        ]

    if run_count == 1:
        strategy = "time_ns" if time_seed else "fixed"
    else:
        strategy = "SeedSequence.spawn"

    return {
        "seed_input": seed_input,
        "seed_base": base_seed,
        "run_seeds": run_seeds,
        "seed_strategy": strategy,
    }


def resolve_run_seed_plan_from_params(params):
    run_count = int(round(float(params["AverageOverRunsN"])))
    if "run_seeds" in params:
        run_seeds = [coerce_seed_int(seed) for seed in params["run_seeds"]]
        if len(run_seeds) != run_count:
            raise ValueError(
                f"run_seeds length must match AverageOverRunsN={run_count}, got {len(run_seeds)}"
            )
        return {
            "seed_input": params.get("seed_input", params["seed"]),
            "seed_base": params.get("seed_base", run_seeds[0]),
            "run_seeds": run_seeds,
            "seed_strategy": params.get("seed_strategy", "provided"),
        }

    return resolve_run_seed_plan(params["seed"], run_count)
