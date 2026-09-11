import importlib
import math
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from src.input.validation import validate_params


def config(profile):
    with patch.dict("os.environ", {"PYSL_PAPER_A_PROFILE": profile}):
        module = importlib.import_module("batch_configs.paper_a_field_mean_width")
        return importlib.reload(module).get_config()


def test_grid_rms_and_endpoints():
    final = config("final")
    assert len(final["cases"]) == 61
    assert len(config("smoke")["cases"]) == 11
    p = final["params_default"]
    assert (p["nqubits_E"], p["AverageOverRunsN"], p["T"], p["printT"]) == (16, 24, 60, 1)
    assert p["clampISE"] == 0 and p["fragment_sample_count"] == 64
    signatures = set()
    for case in final["cases"]:
        validate_params({**p, **case})
        h, w = case["Mironowicz_h0"], case["Mironowicz_alpha2"]
        rms = p["H_SE_J"] * case["field_strength_ratio"]
        eta = case["field_disorder_fraction"]
        assert math.isclose(h*h+w*w, rms*rms, abs_tol=1e-15)
        if eta == 0:
            assert w == 0 and h == rms
        if eta == 1:
            assert h == 0 and w == rms
        assert (h,w) not in signatures
        signatures.add((h,w))


def test_independent_field_signs_preserve_fragment_overlaps():
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1., -1.])
    initial = np.array([1., 1.]) / np.sqrt(2)
    rng = np.random.default_rng(20260908)
    couplings = rng.normal(0, .1, 16)
    signs = rng.choice([-1, 1], 16)
    for strength in (.005, .02, .05, .2, 1., 3.):
        for time in (0., 1., 30., 36., 60.):
            overlaps = []
            for fields in (np.full(16,strength), strength*signs):
                local = []
                for g,h in zip(couplings,fields):
                    u0 = expm(-1j*time*h*z)
                    u1 = expm(-1j*time*(h*z-math.pi*g*x/2))
                    local.append(abs(np.vdot(u0@initial,u1@initial))**2)
                overlaps.append(np.array(local))
            np.testing.assert_allclose(*overlaps, atol=2e-12, rtol=2e-12)
            for size in (1, 4, 8, 16):
                np.testing.assert_allclose(np.prod(overlaps[0][:size]), np.prod(overlaps[1][:size]), atol=2e-12)
