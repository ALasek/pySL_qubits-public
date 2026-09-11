import math

import numpy as np

from eigensolver.sz_eigensolver import build_H_numpy, check_sz_conservation, sz_sectors


_DTYPES = {
    "float32": np.float32,
    "float64": np.float64,
    "complex64": np.complex64,
    "complex128": np.complex128,
}


def adjacent_gap_ratio(eigenvalues, trim_fraction=0.25, gap_tol=1e-12):
    evals = np.sort(np.asarray(eigenvalues, dtype=float))
    if not 0 <= trim_fraction < 0.5:
        raise ValueError(f"trim_fraction must be in [0, 0.5), got {trim_fraction}")

    trim = int(len(evals) * trim_fraction)
    if trim:
        evals = evals[trim:-trim]

    gaps = np.diff(evals)
    if len(gaps) < 2:
        return {
            "mean_r": math.nan,
            "std_r": math.nan,
            "n_ratios": 0,
            "ratios": np.asarray([], dtype=float),
        }

    left = gaps[:-1]
    right = gaps[1:]
    denom = np.maximum(left, right)
    keep = denom > gap_tol
    ratios = np.minimum(left[keep], right[keep]) / denom[keep]

    if len(ratios) == 0:
        return {
            "mean_r": math.nan,
            "std_r": math.nan,
            "n_ratios": 0,
            "ratios": ratios,
        }

    return {
        "mean_r": float(np.mean(ratios)),
        "std_r": float(np.std(ratios)),
        "n_ratios": int(len(ratios)),
        "ratios": ratios,
    }


def parse_sector_mode(sector):
    if isinstance(sector, int):
        return sector
    if sector is None:
        return "auto"
    try:
        return int(sector)
    except (TypeError, ValueError):
        return str(sector).lower()


def choose_sector_k(nqubits, sector="middle"):
    sector = parse_sector_mode(sector)
    if sector in ("auto", "middle", "central"):
        return nqubits // 2
    if sector == "largest":
        sectors = sz_sectors(nqubits)
        return max(sectors, key=lambda k: len(sectors[k]))
    if isinstance(sector, int):
        if not 0 <= sector <= nqubits:
            raise ValueError(f"sector k must be between 0 and {nqubits}, got {sector}")
        return sector
    raise ValueError(f"unsupported sector mode for conserved Hamiltonian: {sector!r}")


def sector_info(nqubits, sector="middle"):
    k = choose_sector_k(nqubits, sector)
    indices = sz_sectors(nqubits)[k]
    return {
        "sector_k": int(k),
        "sector_sz": float(nqubits / 2 - k),
        "sector_dim": int(len(indices)),
        "indices": indices,
    }


def _is_real_matrix(matrix, real_tol=1e-12):
    data = matrix.data if hasattr(matrix, "toarray") else np.asarray(matrix)
    if data.size == 0:
        return True
    return bool(np.max(np.abs(np.imag(data))) <= real_tol)


def _is_hermitian(matrix, tol=1e-8):
    delta = matrix - matrix.conj().T
    if hasattr(delta, "toarray"):
        if delta.nnz == 0:
            return True
        return bool(np.max(np.abs(delta.data)) <= tol)
    return bool(np.allclose(matrix, matrix.conj().T, atol=tol))


def _resolve_dtype(matrix, dtype="auto", real_tol=1e-12):
    is_real = _is_real_matrix(matrix, real_tol=real_tol)
    if dtype in (None, "auto"):
        return np.dtype(np.float32 if is_real else np.complex64), is_real

    key = str(dtype).lower()
    if key not in _DTYPES:
        raise ValueError(f"dtype must be one of auto, {sorted(_DTYPES)}, got {dtype!r}")

    resolved = np.dtype(_DTYPES[key])
    if resolved.kind == "f" and not is_real:
        raise ValueError(f"dtype {resolved} requested for complex Hamiltonian")
    return resolved, is_real


def _as_dense_for_eig(matrix, dtype="auto", real_tol=1e-12):
    resolved, is_real = _resolve_dtype(matrix, dtype=dtype, real_tol=real_tol)
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if resolved.kind == "f":
        dense = np.real(dense)
    return dense.astype(resolved, copy=False), str(resolved), is_real


def _resolve_backend(backend):
    backend = str(backend).lower()
    if backend in ("cpu", "gpu"):
        return backend
    if backend != "auto":
        raise ValueError(f"backend must be 'cpu', 'gpu', or 'auto', got {backend!r}")
    try:
        import cupy as cp

        return "gpu" if cp.cuda.runtime.getDeviceCount() > 0 else "cpu"
    except Exception:
        return "cpu"


def _eigvalsh(matrix, dtype="auto", backend="cpu", real_tol=1e-12):
    dense, dtype_name, is_real = _as_dense_for_eig(matrix, dtype=dtype, real_tol=real_tol)
    resolved_backend = _resolve_backend(backend)
    if resolved_backend == "gpu":
        import cupy as cp

        dense_gpu = cp.asarray(dense)
        evals = cp.linalg.eigvalsh(dense_gpu).get()
        del dense_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        evals = np.linalg.eigvalsh(dense)
    return np.sort(np.asarray(evals, dtype=float)), dtype_name, is_real, resolved_backend


def _full_dim_exceeds(dim, max_full_dim):
    return max_full_dim is not None and max_full_dim > 0 and dim > max_full_dim


def eigenvalues_for_params(
    params,
    sector="auto",
    seed=None,
    tol=1e-10,
    max_full_dim=None,
    dtype="auto",
    backend="cpu",
    real_tol=1e-12,
):
    nqubits = params["nqubits_S"] + params["nqubits_E"]
    dim = 2 ** nqubits
    sector = parse_sector_mode(sector)
    conserved, message = check_sz_conservation(params, tol=tol)
    use_full = (not conserved and sector in ("auto", "full")) or sector == "full"

    if not conserved and sector not in ("auto", "full"):
        raise ValueError(f"sector={sector!r} requested, but Sz is not conserved: {message}")

    if use_full and _full_dim_exceeds(dim, max_full_dim):
        raise ValueError(
            f"full diagonalization dim={dim} exceeds max_full_dim={max_full_dim}; "
            "reduce nqubits_E, use an Sz-conserved model, or remove/raise --max-full-dim"
        )

    H = build_H_numpy(params, seed=seed)

    if conserved and not use_full:
        info = sector_info(nqubits, sector)
        idx = info["indices"]
        block = H[idx[:, None], idx[None, :]]
        if not _is_hermitian(block, tol=max(tol, 1e-8)):
            raise ValueError(f"Sz sector k={info['sector_k']} block is not Hermitian")
        evals, dtype_name, is_real, resolved_backend = _eigvalsh(
            block,
            dtype=dtype,
            backend=backend,
            real_tol=real_tol,
        )
        return {
            "eigenvalues": evals,
            "conserved": True,
            "conservation_message": message,
            "sector_k": info["sector_k"],
            "sector_sz": info["sector_sz"],
            "sector_dim": info["sector_dim"],
            "full_dim": int(dim),
            "is_real": is_real,
            "eig_dtype": dtype_name,
            "backend": resolved_backend,
        }

    if not _is_hermitian(H, tol=max(tol, 1e-8)):
        raise ValueError("Hamiltonian is not Hermitian")
    evals, dtype_name, is_real, resolved_backend = _eigvalsh(
        H,
        dtype=dtype,
        backend=backend,
        real_tol=real_tol,
    )

    return {
        "eigenvalues": evals,
        "conserved": conserved,
        "conservation_message": message,
        "sector_k": None,
        "sector_sz": None,
        "sector_dim": int(dim),
        "full_dim": int(dim),
        "is_real": is_real,
        "eig_dtype": dtype_name,
        "backend": resolved_backend,
    }


def level_stats_for_params(
    params,
    sector="auto",
    seed=None,
    trim_fraction=0.25,
    tol=1e-10,
    max_full_dim=None,
    dtype="auto",
    backend="cpu",
    real_tol=1e-12,
):
    spectrum = eigenvalues_for_params(
        params,
        sector=sector,
        seed=seed,
        tol=tol,
        max_full_dim=max_full_dim,
        dtype=dtype,
        backend=backend,
        real_tol=real_tol,
    )
    gap_stats = adjacent_gap_ratio(spectrum["eigenvalues"], trim_fraction=trim_fraction)
    return {
        "mean_r": gap_stats["mean_r"],
        "std_r": gap_stats["std_r"],
        "n_ratios": gap_stats["n_ratios"],
        "n_levels": int(len(spectrum["eigenvalues"])),
        "conserved": spectrum["conserved"],
        "conservation_message": spectrum["conservation_message"],
        "sector_k": spectrum["sector_k"],
        "sector_sz": spectrum["sector_sz"],
        "sector_dim": spectrum["sector_dim"],
        "full_dim": spectrum["full_dim"],
        "is_real": spectrum["is_real"],
        "eig_dtype": spectrum["eig_dtype"],
        "backend": spectrum["backend"],
    }
