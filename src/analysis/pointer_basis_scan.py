import csv
import hashlib
import itertools
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np

from src.export.path_utils import get_data_dir, get_repo_root
from src.export.run_store import StoredRunResult, load_runs_matching, params_match


CHECKPOINT_VERSION = 2


def _as_float(value):
    if hasattr(value, "get"):
        value = value.get()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _to_numpy(value):
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if hasattr(value, "get"):
        return _jsonable(value.get())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def _canonical_json(data):
    return json.dumps(_jsonable(data), sort_keys=True, separators=(",", ":"))


def _checkpoint_filename(metadata):
    digest = hashlib.sha256(_canonical_json(metadata).encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def checkpoint_path(checkpoint_dir, metadata):
    return os.path.join(checkpoint_dir, _checkpoint_filename(metadata))


def load_checkpoint(path, metadata):
    if not os.path.exists(path):
        return None
    expected = _canonical_json(metadata)
    try:
        with np.load(path, allow_pickle=False) as loaded:
            stored = str(loaded["metadata_json"].item())
            if stored != expected:
                return None
            return {
                "I": float(loaded["I"].item()),
                "chi_grid": np.asarray(loaded["chi_grid"], dtype=float),
            }
    except (OSError, KeyError, ValueError):
        return None


def save_checkpoint(path, metadata, chi_grid, I):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        metadata_json=np.asarray(_canonical_json(metadata)),
        chi_grid=np.asarray(chi_grid, dtype=float),
        I=np.asarray(float(I)),
    )
    os.replace(tmp_path, path)


def parse_filter(value):
    key, sep, raw = value.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError("filters must use KEY=VALUE")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _load_v2_runs_by_id(subdir, run_ids, filters=None):
    run_ids = set(run_ids or [])
    index_path = os.path.join(get_data_dir(subdir), "run_index.jsonl")
    if not os.path.exists(index_path):
        return []
    base = get_data_dir(subdir)
    matches = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") not in run_ids:
                continue
            params = row.get("params", {})
            if filters and not params_match(params, filters):
                continue
            params_path = os.path.join(base, row["paths"]["params"])
            if os.path.exists(params_path):
                with open(params_path, encoding="utf-8") as params_file:
                    params = json.load(params_file)
            results_path = os.path.join(base, row["paths"]["results"])
            matches.append([params, StoredRunResult(params, results_path, row.get("metadata", {}))])
    return matches


def array_module_for_device(device):
    if device == "cpu":
        return np
    if device == "cuda":
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("device='cuda' requires CuPy") from exc
        return cp
    raise ValueError(f"unsupported device: {device}")


def complex_dtype(dtype):
    if dtype in {"complex64", np.complex64}:
        return np.complex64
    if dtype in {"complex128", np.complex128}:
        return np.complex128
    raise ValueError("dtype must be complex64 or complex128")


def make_basis_grid(beta_count, phi_count):
    if beta_count < 2:
        raise ValueError("beta_count must be at least 2")
    if phi_count < 1:
        raise ValueError("phi_count must be at least 1")
    beta = np.linspace(0.0, math.pi, beta_count)
    beta[0] = 0.0
    beta[-1] = math.pi
    phi = np.linspace(0.0, 2.0 * math.pi, phi_count, endpoint=False)
    return beta, phi


def _all_combinations(environment_qubits, m):
    return [tuple(item) for item in itertools.combinations(environment_qubits, m)]


def sample_fragments(
    environment_qubits,
    fragment_sizes,
    num_fragments,
    seed=1234,
    mode="random",
    explicit_fragments=None,
):
    environment_qubits = list(environment_qubits)
    rng = random.Random(seed)
    out = {}
    if mode == "explicit":
        if explicit_fragments is None:
            raise ValueError("fragment_mode='explicit' requires explicit_fragments")
        for m in fragment_sizes:
            out[int(m)] = [tuple(fragment) for fragment in explicit_fragments[str(int(m))]]
        return out

    if mode == "random":
        for m in fragment_sizes:
            m = int(m)
            if m < 0 or m > len(environment_qubits):
                raise ValueError(f"invalid fragment size {m}")
            combos = _all_combinations(environment_qubits, m)
            if len(combos) <= num_fragments:
                out[m] = combos
            else:
                out[m] = rng.sample(combos, num_fragments)
        return out

    if mode == "prefix-permutation":
        permutations = []
        for _ in range(num_fragments):
            perm = list(environment_qubits)
            rng.shuffle(perm)
            permutations.append(perm)
        for m in fragment_sizes:
            m = int(m)
            if m < 0 or m > len(environment_qubits):
                raise ValueError(f"invalid fragment size {m}")
            out[m] = [tuple(perm[:m]) for perm in permutations]
        return out

    raise ValueError(f"unsupported fragment mode: {mode}")


def flat_to_qubit_tensor(psi, n_qubits, endianness="big"):
    xp = np if isinstance(psi, np.ndarray) else __import__("cupy")
    tensor = psi.reshape((2,) * n_qubits)
    if endianness == "big":
        return tensor
    if endianness == "little":
        return xp.transpose(tensor, tuple(reversed(range(n_qubits))))
    raise ValueError("endianness must be 'big' or 'little'")


def branch_matrices_from_psi(psi, n_qubits, system_qubit, fragment, endianness="big"):
    xp = np if isinstance(psi, np.ndarray) else __import__("cupy")
    fragment = list(fragment)
    occupied = {system_qubit, *fragment}
    rest = [q for q in range(n_qubits) if q not in occupied]
    tensor = flat_to_qubit_tensor(psi, n_qubits, endianness=endianness)
    tensor = xp.transpose(tensor, tuple([system_qubit] + fragment + rest))
    reshaped = tensor.reshape(2, 2 ** len(fragment), -1)
    return reshaped[0], reshaped[1]


def compute_blocks(A0, A1):
    G00 = A0 @ A0.conj().T
    G11 = A1 @ A1.conj().T
    G01 = A0 @ A1.conj().T
    return G00, G11, G01


def entropy_psd(mat, trace=None, eps=1e-12):
    xp = np if isinstance(mat, np.ndarray) else __import__("cupy")
    herm = (mat + mat.conj().T) * 0.5
    evals = xp.linalg.eigvalsh(herm).real
    evals = xp.maximum(evals, 0)
    norm = evals.sum() if trace is None else trace
    norm_value = _as_float(norm)
    if norm_value <= eps:
        return 0.0
    probs = evals / norm
    probs = probs[probs > eps]
    if probs.size == 0:
        return 0.0
    return _as_float(-(probs * xp.log2(probs)).sum())


def entropy_from_amplitude_matrix(P, eps=1e-12):
    if P.shape[0] <= P.shape[1]:
        gram = P @ P.conj().T
    else:
        gram = P.conj().T @ P
    return entropy_psd(gram, eps=eps)


def entropy_psd_batch(mats, traces=None, eps=1e-12):
    xp = np if isinstance(mats, np.ndarray) else __import__("cupy")
    herm = (mats + mats.conj().swapaxes(-1, -2)) * 0.5
    evals = xp.linalg.eigvalsh(herm).real
    evals = xp.maximum(evals, 0)
    traces = evals.sum(axis=-1) if traces is None else traces
    traces = xp.asarray(traces).real
    safe_traces = xp.where(traces > eps, traces, 1.0)
    probs = evals / safe_traces[..., None]
    terms = xp.zeros_like(probs)
    mask = probs > eps
    terms[mask] = probs[mask] * xp.log2(probs[mask])
    entropy = -terms.sum(axis=-1)
    entropy = xp.where(traces > eps, entropy, 0.0)
    return _to_numpy(entropy)


def mutual_information_from_branches(A0, A1, G00, G11, G01, eps=1e-12):
    xp = np if isinstance(A0, np.ndarray) else __import__("cupy")
    G10 = G01.conj().T
    rho_s = xp.empty((2, 2), dtype=G00.dtype)
    rho_s[0, 0] = xp.trace(G00)
    rho_s[0, 1] = xp.trace(G01)
    rho_s[1, 0] = xp.trace(G10)
    rho_s[1, 1] = xp.trace(G11)
    rho_s = (rho_s + rho_s.conj().T) * 0.5
    rho_f = G00 + G11
    P = xp.concatenate([A0, A1], axis=0)
    value = (
        entropy_psd(rho_s, eps=eps)
        + entropy_psd(rho_f, eps=eps)
        - entropy_from_amplitude_matrix(P, eps=eps)
    )
    return max(0.0, value)


def chi_for_basis_from_blocks(G00, G11, G01, beta, phi, S_F=None, eps=1e-12):
    xp = np if isinstance(G00, np.ndarray) else __import__("cupy")
    G10 = G01.conj().T
    S_F = entropy_psd(G00 + G11, eps=eps) if S_F is None else S_F
    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    phase = complex(math.cos(phi), math.sin(phi))
    x_phi = phase * G01 + phase.conjugate() * G10
    rho_plus = (c * c) * G00 + (s * s) * G11 + c * s * x_phi
    rho_minus = (s * s) * G00 + (c * c) * G11 - c * s * x_phi
    p_plus = xp.trace(rho_plus).real
    p_minus = xp.trace(rho_minus).real
    chi = S_F
    if _as_float(p_plus) > eps:
        chi -= _as_float(p_plus) * entropy_psd(rho_plus, trace=p_plus, eps=eps)
    if _as_float(p_minus) > eps:
        chi -= _as_float(p_minus) * entropy_psd(rho_minus, trace=p_minus, eps=eps)
    return max(0.0, chi)


def default_basis_batch_size(fragment_size):
    if fragment_size <= 4:
        return 512
    if fragment_size <= 6:
        return 256
    if fragment_size <= 8:
        return 32
    if fragment_size <= 10:
        return 8
    return 1


def _resolve_basis_batch_size(basis_batch_size, fragment_dim):
    if basis_batch_size in (None, "auto"):
        fragment_size = int(round(math.log2(fragment_dim))) if fragment_dim > 0 else 0
        return default_basis_batch_size(fragment_size)
    return max(1, int(basis_batch_size))


def scan_basis_grid(G00, G11, G01, beta_grid, phi_grid, eps=1e-12, basis_batch_size="auto"):
    xp = np if isinstance(G00, np.ndarray) else __import__("cupy")
    S_F = entropy_psd(G00 + G11, eps=eps)
    pairs = [(bi, pi, float(beta), float(phi)) for bi, beta in enumerate(beta_grid) for pi, phi in enumerate(phi_grid)]
    out = np.empty((len(beta_grid), len(phi_grid)), dtype=float)
    batch_size = _resolve_basis_batch_size(basis_batch_size, G00.shape[0])
    G10 = G01.conj().T

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        betas = xp.asarray([item[2] for item in chunk], dtype=float)
        phis = xp.asarray([item[3] for item in chunk], dtype=float)
        c = xp.cos(betas / 2.0)[:, None, None]
        s = xp.sin(betas / 2.0)[:, None, None]
        phase = xp.exp(1j * phis)[:, None, None]
        x_phi = phase * G01[None, :, :] + phase.conj() * G10[None, :, :]
        rho_plus = (c * c) * G00[None, :, :] + (s * s) * G11[None, :, :] + c * s * x_phi
        rho_minus = (s * s) * G00[None, :, :] + (c * c) * G11[None, :, :] - c * s * x_phi
        p_plus = xp.trace(rho_plus, axis1=-2, axis2=-1).real
        p_minus = xp.trace(rho_minus, axis1=-2, axis2=-1).real
        S_plus = entropy_psd_batch(rho_plus, traces=p_plus, eps=eps)
        S_minus = entropy_psd_batch(rho_minus, traces=p_minus, eps=eps)
        chi = S_F - _to_numpy(p_plus) * S_plus - _to_numpy(p_minus) * S_minus
        chi = np.maximum(0.0, chi)
        for local_i, (bi, pi, _beta, _phi) in enumerate(chunk):
            out[bi, pi] = float(chi[local_i])
    return out


def angle_from_z(beta):
    return float(math.acos(abs(math.cos(float(beta)))))


def _sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def _sem_grid(grids, shape):
    if len(grids) <= 1:
        return np.zeros(shape, dtype=float)
    stack = np.stack(grids, axis=0)
    return np.nanstd(stack, axis=0, ddof=1) / math.sqrt(len(grids))


def _mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan
    return float(np.nanmean(values))


class FragmentAccumulator:
    def __init__(self, beta_grid, phi_grid):
        self.beta_grid = beta_grid
        self.phi_grid = phi_grid
        self.sum_chi_grid = np.zeros((len(beta_grid), len(phi_grid)), dtype=float)
        self.count = 0
        self.chi_z = []
        self.I = []
        self.local_max = []
        self.local_angle = []
        self.local_rows = []

    def add(self, chi_grid, I, realization_id, time_index, time_value, fragment_size, fragment_id, save_local):
        flat_idx = int(np.nanargmax(chi_grid))
        beta_idx, phi_idx = np.unravel_index(flat_idx, chi_grid.shape)
        chi_z = float(chi_grid[0, 0])
        chi_local = float(chi_grid[beta_idx, phi_idx])
        local_angle = angle_from_z(self.beta_grid[beta_idx])
        self.sum_chi_grid += chi_grid
        self.count += 1
        self.chi_z.append(chi_z)
        self.I.append(float(I))
        self.local_max.append(chi_local)
        self.local_angle.append(local_angle)
        if save_local:
            self.local_rows.append(
                {
                    "realization_id": realization_id,
                    "time_index": time_index,
                    "time_value": time_value,
                    "fragment_size": fragment_size,
                    "fragment_id": fragment_id,
                    "chi_z": chi_z,
                    "I": float(I),
                    "D_z": max(0.0, float(I) - chi_z),
                    "chi_local_max": chi_local,
                    "beta_local_best": float(self.beta_grid[beta_idx]),
                    "phi_local_best": float(self.phi_grid[phi_idx]),
                    "angle_local_from_z": local_angle,
                }
            )

    def mean_grid(self):
        if self.count == 0:
            return np.full_like(self.sum_chi_grid, np.nan)
        return self.sum_chi_grid / self.count


class RealizationAccumulator:
    def __init__(self, fragment_sizes, beta_grid, phi_grid):
        self.by_m = {int(m): FragmentAccumulator(beta_grid, phi_grid) for m in fragment_sizes}

    def add(self, m, *args, **kwargs):
        self.by_m[int(m)].add(*args, **kwargs)


@dataclass
class SnapshotSource:
    params: dict
    result: object
    record: dict

    @property
    def run_index(self):
        return int(self.record.get("run_index", 0))

    @property
    def run_id(self):
        return self.result.metadata.get("run_id", "unknown")

    @property
    def realization_id(self):
        seed = self.record.get("seed")
        suffix = f":seed{seed}" if seed is not None else ""
        return f"{self.run_id}:run{self.run_index}{suffix}"

    @property
    def times(self):
        times = self.record.get("sample_times")
        if times is not None:
            return np.asarray(times, dtype=float)
        shape = self.record.get("shape")
        if shape:
            n_times = int(shape[0])
        else:
            snapshots = self.result.load_psi_snapshots(run_index=self.run_index, mmap_mode="r")
            n_times = snapshots.shape[0]
        printT = float(self.params.get("printT", 1.0))
        return np.arange(n_times, dtype=float) * printT


def discover_snapshot_sources(subdir, filters=None, run_ids=None, max_realizations=None):
    run_ids = set(run_ids or [])
    if run_ids:
        matches = _load_v2_runs_by_id(subdir, run_ids, filters=filters)
    else:
        matches = load_runs_matching(filters or {}, subdir=subdir, allow_legacy=False)
    sources = []
    for params, result in matches:
        records = result.psi_snapshot_records() if hasattr(result, "psi_snapshot_records") else []
        for record in records:
            sources.append(SnapshotSource(params=params, result=result, record=record))
            if max_realizations is not None and len(sources) >= max_realizations:
                return sources
    if run_ids:
        found = {source.run_id for source in sources}
        missing = sorted(run_ids - found)
        if missing:
            raise ValueError(f"no v2 runs with psi snapshots found for run_id(s): {missing}")
    return sources


def infer_metadata_from_sources(sources, system_qubit=0, environment_qubits=None, endianness="big"):
    if not sources:
        raise ValueError("no snapshot sources found")
    params = sources[0].params
    n_qubits = params.get("nqubits")
    if n_qubits is None:
        n_qubits = int(params.get("nqubits_S", 1)) + int(params.get("nqubits_E"))
    else:
        n_qubits = int(n_qubits)
    if environment_qubits is None:
        environment_qubits = [q for q in range(n_qubits) if q != system_qubit]
    return {
        "n_qubits": n_qubits,
        "system_qubit": int(system_qubit),
        "environment_qubits": list(environment_qubits),
        "endianness": endianness,
    }


def select_time_indices(times, time_indices=None, times_requested=None):
    if time_indices:
        selected = []
        for idx in time_indices:
            idx = int(idx)
            if idx < 0:
                idx += len(times)
            if idx < 0 or idx >= len(times):
                raise IndexError(f"time index {idx} is out of bounds for {len(times)} snapshots")
            selected.append(idx)
        return selected
    if times_requested:
        selected = []
        times = np.asarray(times, dtype=float)
        for target in times_requested:
            selected.append(int(np.argmin(np.abs(times - float(target)))))
        return sorted(set(selected))
    return list(range(len(times)))


def snapshot_frame_to_complex(frame, dtype):
    dtype = complex_dtype(dtype)
    if np.iscomplexobj(frame):
        return np.asarray(frame, dtype=dtype)
    if frame.ndim == 2 and frame.shape[0] == 2:
        return frame[0].astype(dtype) + 1j * frame[1].astype(dtype)
    raise ValueError("expected complex vector or pySL snapshot frame with shape (2, dim)")


def normalize_psi(psi, xp):
    norm = xp.linalg.norm(psi)
    norm_value = _as_float(norm)
    if norm_value == 0:
        raise ValueError("wavefunction has zero norm")
    return psi / norm


def sample_checkpoint_metadata(
    source,
    time_index,
    time_value,
    fragment_size,
    fragment_id,
    fragment,
    scan_metadata,
):
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": source.run_id,
        "run_generation": source.result.metadata.get("created_at"),
        "run_index": source.run_index,
        "time_index": int(time_index),
        "time_value": float(time_value),
        "fragment_size": int(fragment_size),
        "fragment_id": int(fragment_id),
        "fragment": list(fragment),
        **scan_metadata,
    }


def analyze_pointer_basis_scan(
    subdir,
    out_dir,
    filters=None,
    run_ids=None,
    fragment_sizes=(4, 6, 8),
    num_fragments=50,
    fragment_seed=1234,
    fragment_mode="random",
    explicit_fragments=None,
    beta_count=31,
    phi_count=64,
    device="cuda",
    dtype="complex64",
    eig_eps=1e-12,
    time_indices=None,
    times_requested=None,
    max_realizations=None,
    normalize=True,
    save_local_samples=False,
    bootstrap=0,
    plot=False,
    basis_batch_size="auto",
    checkpoint=True,
    force_recompute_checkpoints=False,
    system_qubit=0,
    environment_qubits=None,
    endianness="big",
    verbose=False,
):
    xp = array_module_for_device(device)
    beta_grid, phi_grid = make_basis_grid(beta_count, phi_count)
    sources = discover_snapshot_sources(
        subdir,
        filters=filters,
        run_ids=run_ids,
        max_realizations=max_realizations,
    )
    metadata = infer_metadata_from_sources(
        sources,
        system_qubit=system_qubit,
        environment_qubits=environment_qubits,
        endianness=endianness,
    )
    fragments_by_m = sample_fragments(
        metadata["environment_qubits"],
        fragment_sizes,
        num_fragments,
        seed=fragment_seed,
        mode=fragment_mode,
        explicit_fragments=explicit_fragments,
    )
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    scan_metadata = {
        "n_qubits": metadata["n_qubits"],
        "system_qubit": metadata["system_qubit"],
        "endianness": metadata["endianness"],
        "beta_count": int(beta_count),
        "phi_count": int(phi_count),
        "beta_grid": beta_grid.tolist(),
        "phi_grid": phi_grid.tolist(),
        "dtype": str(dtype),
        "eig_eps": float(eig_eps),
        "basis_batch_size": str(basis_batch_size),
        "normalize_psi": bool(normalize),
    }

    global_accum = {int(m): FragmentAccumulator(beta_grid, phi_grid) for m in fragment_sizes}
    realization_grids = {int(m): [] for m in fragment_sizes}
    realization_rows = []
    local_sample_rows = []
    checkpoint_hits = 0
    checkpoint_writes = 0

    for source_i, source in enumerate(sources):
        if verbose:
            print(f"[{source_i + 1}/{len(sources)}] {source.realization_id}")
        snapshots = source.result.load_psi_snapshots(run_index=source.run_index, mmap_mode="r")
        source_times = source.times
        selected = select_time_indices(source_times, time_indices=time_indices, times_requested=times_requested)
        r_accum = RealizationAccumulator(fragment_sizes, beta_grid, phi_grid)

        for time_index in selected:
            psi = None
            for m in fragment_sizes:
                for fragment_id, fragment in enumerate(fragments_by_m[int(m)]):
                    checkpoint_metadata = sample_checkpoint_metadata(
                        source,
                        time_index,
                        source_times[time_index],
                        m,
                        fragment_id,
                        fragment,
                        scan_metadata,
                    )
                    cached = None
                    cache_path = None
                    if checkpoint:
                        cache_path = checkpoint_path(checkpoint_dir, checkpoint_metadata)
                        if not force_recompute_checkpoints:
                            cached = load_checkpoint(cache_path, checkpoint_metadata)

                    if cached is not None:
                        I = cached["I"]
                        chi_grid = cached["chi_grid"]
                        checkpoint_hits += 1
                    else:
                        if psi is None:
                            psi_np = snapshot_frame_to_complex(snapshots[time_index], dtype=dtype)
                            psi = xp.asarray(psi_np)
                            if normalize:
                                psi = normalize_psi(psi, xp)
                        A0, A1 = branch_matrices_from_psi(
                            psi,
                            metadata["n_qubits"],
                            metadata["system_qubit"],
                            fragment,
                            endianness=metadata["endianness"],
                        )
                        G00, G11, G01 = compute_blocks(A0, A1)
                        I = mutual_information_from_branches(A0, A1, G00, G11, G01, eps=eig_eps)
                        chi_grid = scan_basis_grid(
                            G00,
                            G11,
                            G01,
                            beta_grid,
                            phi_grid,
                            eps=eig_eps,
                            basis_batch_size=basis_batch_size,
                        )
                        if checkpoint and cache_path is not None:
                            save_checkpoint(cache_path, checkpoint_metadata, chi_grid, I)
                            checkpoint_writes += 1
                    row_args = (
                        chi_grid,
                        I,
                        source.realization_id,
                        int(time_index),
                        float(source_times[time_index]),
                        int(m),
                        int(fragment_id),
                    )
                    global_accum[int(m)].add(*row_args, save_local=save_local_samples)
                    r_accum.add(int(m), *row_args, save_local=False)

            if device == "cuda":
                xp.get_default_memory_pool().free_all_blocks()
            if psi is not None:
                del psi

        del snapshots
        for m, accum in r_accum.by_m.items():
            if accum.count == 0:
                continue
            realization_grids[int(m)].append(accum.mean_grid())
            realization_rows.append(
                {
                    "realization_id": source.realization_id,
                    "fragment_size": m,
                    "num_times": len(selected),
                    "num_fragments": len(fragments_by_m[m]),
                    "mean_chi_z": _mean(accum.chi_z),
                    "mean_I": _mean(accum.I),
                    "mean_D_z": max(0.0, _mean(accum.I) - _mean(accum.chi_z)),
                    "mean_chi_local_max": _mean(accum.local_max),
                    "mean_local_angle_from_z": _mean(accum.local_angle),
                }
            )

    results_by_fragment_size = {}
    for m, accum in global_accum.items():
        mean_grid = accum.mean_grid()
        best_flat = int(np.nanargmax(mean_grid)) if np.isfinite(mean_grid).any() else 0
        best_beta_i, best_phi_i = np.unravel_index(best_flat, mean_grid.shape)
        mean_chi_z = _mean(accum.chi_z)
        mean_I = _mean(accum.I)
        mean_chi_global_max = float(mean_grid[best_beta_i, best_phi_i])
        r_rows = [row for row in realization_rows if int(row["fragment_size"]) == int(m)]
        results_by_fragment_size[str(m)] = {
            "sample_count": accum.count,
            "mean_chi_z": mean_chi_z,
            "sem_chi_z": _sem([row["mean_chi_z"] for row in r_rows]),
            "mean_I": mean_I,
            "sem_I": _sem([row["mean_I"] for row in r_rows]),
            "mean_D_z": max(0.0, mean_I - mean_chi_z),
            "sem_D_z": _sem([row["mean_D_z"] for row in r_rows]),
            "mean_chi_global_max": mean_chi_global_max,
            "delta_chi_global": max(0.0, mean_chi_global_max - mean_chi_z),
            "best_beta": float(beta_grid[best_beta_i]),
            "best_phi": float(phi_grid[best_phi_i]),
            "angle_from_z": angle_from_z(beta_grid[best_beta_i]),
            "mean_D_global": max(0.0, mean_I - mean_chi_global_max),
            "mean_chi_local_max": _mean(accum.local_max),
            "delta_chi_local": max(0.0, _mean(accum.local_max) - mean_chi_z),
            "mean_local_angle_from_z": _mean(accum.local_angle),
            "std_local_angle_from_z": float(np.nanstd(accum.local_angle)) if accum.local_angle else math.nan,
        }

    summary = {
        "subdir": subdir,
        "filters": filters or {},
        "run_ids": list(run_ids or []),
        "metadata": metadata,
        "fragment_sizes": [int(m) for m in fragment_sizes],
        "num_realizations": len(sources),
        "num_fragments_per_size": {str(m): len(fragments_by_m[int(m)]) for m in fragment_sizes},
        "beta_count": beta_count,
        "phi_count": phi_count,
        "device": device,
        "dtype": str(dtype),
        "basis_batch_size": basis_batch_size,
        "normalize_psi": bool(normalize),
        "checkpoint": {
            "enabled": bool(checkpoint),
            "dir": checkpoint_dir if checkpoint else None,
            "hits": int(checkpoint_hits),
            "writes": int(checkpoint_writes),
            "force_recompute": bool(force_recompute_checkpoints),
        },
        "results_by_fragment_size": results_by_fragment_size,
    }
    if bootstrap:
        summary["bootstrap"] = bootstrap_realization_rows(realization_rows, bootstrap, seed=fragment_seed)

    write_pointer_outputs(
        out_dir,
        summary,
        global_accum,
        realization_rows,
        local_sample_rows if save_local_samples else None,
        beta_grid,
        phi_grid,
        realization_grids,
        plot=plot,
    )
    return summary


def bootstrap_realization_rows(realization_rows, bootstrap, seed=1234):
    rng = np.random.default_rng(seed)
    ids = sorted({row["realization_id"] for row in realization_rows})
    if len(ids) <= 1:
        return {"samples": int(bootstrap), "note": "not enough realizations"}
    by_id = {rid: [row for row in realization_rows if row["realization_id"] == rid] for rid in ids}
    out = {}
    for m in sorted({int(row["fragment_size"]) for row in realization_rows}):
        metrics = {
            "mean_chi_z": [],
            "mean_I": [],
            "mean_D_z": [],
            "mean_chi_local_max": [],
        }
        for _ in range(int(bootstrap)):
            sampled = rng.choice(ids, size=len(ids), replace=True)
            rows = [row for rid in sampled for row in by_id[rid] if int(row["fragment_size"]) == m]
            for key in metrics:
                metrics[key].append(_mean([row[key] for row in rows]))
        out[str(m)] = {
            key: {
                "mean": _mean(values),
                "ci95": [
                    float(np.nanpercentile(values, 2.5)),
                    float(np.nanpercentile(values, 97.5)),
                ],
            }
            for key, values in metrics.items()
        }
    return {"samples": int(bootstrap), "by_fragment_size": out}


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_pointer_outputs(
    out_dir,
    summary,
    global_accum,
    realization_rows,
    local_sample_rows,
    beta_grid,
    phi_grid,
    realization_grids,
    plot=False,
):
    write_json(os.path.join(out_dir, "summary.json"), summary)
    write_csv(
        os.path.join(out_dir, "per_realization_summary.csv"),
        realization_rows,
        [
            "realization_id",
            "fragment_size",
            "num_times",
            "num_fragments",
            "mean_chi_z",
            "mean_I",
            "mean_D_z",
            "mean_chi_local_max",
            "mean_local_angle_from_z",
        ],
    )
    if local_sample_rows is not None:
        for accum in global_accum.values():
            local_sample_rows.extend(accum.local_rows)
        write_csv(
            os.path.join(out_dir, "local_best_samples.csv"),
            local_sample_rows,
            [
                "realization_id",
                "time_index",
                "time_value",
                "fragment_size",
                "fragment_id",
                "chi_z",
                "I",
                "D_z",
                "chi_local_max",
                "beta_local_best",
                "phi_local_best",
                "angle_local_from_z",
            ],
        )

    for m, accum in global_accum.items():
        mean_grid = accum.mean_grid()
        mean_I = _mean(accum.I)
        mean_chi_z = _mean(accum.chi_z)
        np.savez_compressed(
            os.path.join(out_dir, f"chi_grid_m{m}.npz"),
            beta_grid=beta_grid,
            phi_grid=phi_grid,
            mean_chi_grid=mean_grid,
            sem_chi_grid=_sem_grid(realization_grids.get(int(m), []), mean_grid.shape),
            mean_D_grid=np.maximum(0.0, mean_I - mean_grid),
            mean_I=mean_I,
            mean_chi_z=mean_chi_z,
        )
        if plot:
            make_pointer_plots(out_dir, int(m), accum, beta_grid, phi_grid)
    write_pointer_readme(out_dir, summary)


def write_pointer_readme(out_dir, summary):
    lines = [
        "# Pointer basis scan",
        "",
        "This analysis scans projective measurement bases on the system qubit and computes",
        "Holevo information from saved pySL full-state snapshots.",
        "",
        "It performs a quenched average over realizations, times, and sampled fragments.",
        "Wavefunctions are never averaged.",
        "",
        "## Interpretation",
        "- delta_chi_global near zero with angle_from_z near zero means the Z pointer basis is already optimal.",
        "- Large local improvement but small global improvement means basis choices are sample-dependent.",
        "- Low chi_z and low D_z means there is no useful record; low discord alone is not QD evidence.",
        "- chi_z approximately equal to I and H(S) indicates a strong classical pointer record.",
        "",
        "## pySL inputs",
        f"- subdir: {summary['subdir']}",
        f"- filters: {summary['filters']}",
        f"- snapshots analyzed: {summary['num_realizations']}",
        f"- metadata: {summary['metadata']}",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_pointer_plots(out_dir, m, accum, beta_grid, phi_grid):
    import matplotlib.pyplot as plt

    mean_grid = accum.mean_grid()
    mean_chi_z = _mean(accum.chi_z)
    plot_specs = [
        ("chi_grid", mean_grid, "mean chi"),
        ("delta_chi_grid", mean_grid - mean_chi_z, "mean chi - mean chi_z"),
    ]
    for name, data, label in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 5))
        mesh = ax.pcolormesh(phi_grid, beta_grid, data, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel("phi")
        ax.set_ylabel("beta")
        ax.set_title(f"{label}, fragment size m={m}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_m{m}.png"), dpi=200)
        plt.close(fig)

    if accum.local_angle:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(accum.local_angle, bins=30)
        ax.set_xlabel("local best angle from z")
        ax.set_ylabel("count")
        ax.set_title(f"Local best angle, m={m}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"local_best_angle_hist_m{m}.png"), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["chi_z", "chi_global_max", "chi_local_max", "I"]
    mean_grid = accum.mean_grid()
    values = [
        _mean(accum.chi_z),
        float(np.nanmax(mean_grid)),
        _mean(accum.local_max),
        _mean(accum.I),
    ]
    ax.bar(labels, values)
    ax.set_ylabel("bits")
    ax.set_title(f"Pointer scan summary, m={m}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"summary_bars_m{m}.png"), dpi=200)
    plt.close(fig)


def default_output_dir(subdir, analysis_name=None):
    name = analysis_name or "pointer_basis_scan"
    return os.path.join(get_repo_root(), "data", "figs", subdir, name)
