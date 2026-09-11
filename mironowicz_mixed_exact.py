"""
CLI README

Standalone exact-density-matrix scanner for the two-environment-qubit model in
Mironowicz et al., "Non-Perfect Propagation of Information to a Noisy
Environment with Self-Evolution".

This is not integrated with pySL. It uses dense NumPy/SciPy matrices for the
three-qubit state S,E1,E2, evolves to t=1 by exp(-i H t), traces out E2, and
analyzes rho(S,E1).

Common commands:
  python mironowicz_mixed_exact.py --self-test

  python mironowicz_mixed_exact.py --theta 0 --alpha-points 61 --p-points 61

  python mironowicz_mixed_exact.py --theta-set fig2 --alpha-points 41 --p-points 41

  python mironowicz_mixed_exact.py --theta 0 --compute-sbs-delta \
      --alpha-points 21 --p-points 21 --sbs-maxiter 60

  python mironowicz_mixed_exact.py --theta 0 --progress-interval 5

Main outputs:
  - theta_*_scan.npz: arrays on the p x alpha2 grid.
  - PNG heatmaps for mutual information, best Holevo information, Z-basis
    Holevo information, Holevo gain over Z, best-basis angle, and
    best_chi / I(S:E1).
  - Optional SBS outputs:
      sbs_delta_hadamard_minus_z, sbs_distance_z, sbs_distance_hadamard,
      sbs_distance_full.

Interpretation:
  The Holevo scan maximizes chi_n(S:E1) over projective measurements on S.
  A rotated optimum is interesting only if best_chi is appreciable, not just if
  the maximizing basis is far from Z.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm, svdvals
from scipy.optimize import differential_evolution


I2 = np.eye(2, dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _as_float_expr(text: str) -> float:
    allowed = {"pi": math.pi, "tau": math.tau, "e": math.e}
    return float(eval(text, {"__builtins__": {}}, allowed))


def bits_to_int(bits: list[int]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def int_to_bits(value: int, nbits: int) -> list[int]:
    return [(value >> shift) & 1 for shift in range(nbits - 1, -1, -1)]


def kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=np.complex128)
    for op in ops:
        out = np.kron(out, op)
    return out


def embed_operator(op: np.ndarray, targets: list[int], nqubits: int) -> np.ndarray:
    targets = list(targets)
    k = len(targets)
    dim = 2**nqubits
    full = np.zeros((dim, dim), dtype=np.complex128)

    for in_idx in range(dim):
        in_bits = int_to_bits(in_idx, nqubits)
        target_in = bits_to_int([in_bits[target] for target in targets])
        for target_out in range(2**k):
            out_bits = in_bits.copy()
            for target, bit in zip(targets, int_to_bits(target_out, k)):
                out_bits[target] = bit
            out_idx = bits_to_int(out_bits)
            full[out_idx, in_idx] += op[target_out, target_in]
    return full


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    keep = sorted(keep)
    trace_out = [idx for idx in range(len(dims)) if idx not in keep]
    tensor = rho.reshape(dims + dims)
    live_dims = list(dims)
    for axis in sorted(trace_out, reverse=True):
        tensor = np.trace(tensor, axis1=axis, axis2=axis + len(live_dims))
        live_dims.pop(axis)
    out_dim = int(np.prod([dims[idx] for idx in keep]))
    return tensor.reshape(out_dim, out_dim)


def hermitize(rho: np.ndarray) -> np.ndarray:
    return 0.5 * (rho + rho.conj().T)


def entropy_vn(rho: np.ndarray, eps: float = 1e-13) -> float:
    vals = np.linalg.eigvalsh(hermitize(rho)).real
    vals = np.clip(vals, 0.0, None)
    vals = vals[vals > eps]
    if vals.size == 0:
        return 0.0
    return float(-np.sum(vals * np.log2(vals)))


def trace_norm(mat: np.ndarray) -> float:
    return float(np.sum(svdvals(mat)))


def qubit_basis(beta: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    c = math.cos(0.5 * beta)
    s = math.sin(0.5 * beta)
    phase = np.exp(1j * phi)
    v0 = np.array([c, phase * s], dtype=np.complex128)
    v1 = np.array([-np.conj(phase) * s, c], dtype=np.complex128)
    return v0, v1


def projector(vec: np.ndarray) -> np.ndarray:
    return np.outer(vec, vec.conj())


def c_inot_unitary(theta: float) -> np.ndarray:
    st = math.sin(theta)
    ct = math.cos(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, st, ct],
            [0.0, 0.0, ct, -st],
        ],
        dtype=np.complex128,
    )


def h_c_inot(theta: float) -> np.ndarray:
    st = math.sin(theta)
    ct = math.cos(theta)
    h = np.zeros((4, 4), dtype=np.complex128)
    h[2, 2] = 0.5 * math.pi * (1.0 - st)
    h[2, 3] = -0.5 * math.pi * ct
    h[3, 2] = -0.5 * math.pi * ct
    h[3, 3] = 0.5 * math.pi * (1.0 + st)
    return h


def total_hamiltonian(theta: float, alpha2: float) -> np.ndarray:
    hc = h_c_inot(theta)
    h = embed_operator(hc, [0, 1], 3)
    h += embed_operator(hc, [0, 2], 3)
    h += alpha2 * (embed_operator(Z, [1], 3) + embed_operator(Z, [2], 3))
    return hermitize(h)


def initial_state(p: float) -> np.ndarray:
    ket_plus = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    rho_s = projector(ket_plus)
    rho_e = np.diag([p, 1.0 - p]).astype(np.complex128)
    return kron_all([rho_s, rho_e, rho_e])


def evolve_rho_se1_from_u(u: np.ndarray, p: float) -> np.ndarray:
    rho0 = initial_state(p)
    rho3 = u @ rho0 @ u.conj().T
    rho_se1 = partial_trace(rho3, keep=[0, 1], dims=[2, 2, 2])
    rho_se1 = hermitize(rho_se1)
    return rho_se1 / np.trace(rho_se1).real


def conditional_env_state(rho_se: np.ndarray, system_vec: np.ndarray) -> np.ndarray:
    tensor = rho_se.reshape(2, 2, 2, 2)
    return np.einsum("a,aebf,b->ef", system_vec.conj(), tensor, system_vec)


def mutual_information(rho_se: np.ndarray) -> float:
    rho_s = partial_trace(rho_se, keep=[0], dims=[2, 2])
    rho_e = partial_trace(rho_se, keep=[1], dims=[2, 2])
    return entropy_vn(rho_s) + entropy_vn(rho_e) - entropy_vn(rho_se)


def holevo_for_basis(rho_se: np.ndarray, beta: float, phi: float) -> float:
    rho_e = partial_trace(rho_se, keep=[1], dims=[2, 2])
    chi = entropy_vn(rho_e)
    for vec in qubit_basis(beta, phi):
        rho_e_cond = conditional_env_state(rho_se, vec)
        prob = float(np.trace(rho_e_cond).real)
        if prob > 1e-13:
            chi -= prob * entropy_vn(rho_e_cond / prob)
    return max(0.0, float(chi))


def scan_holevo_basis(
    rho_se: np.ndarray,
    beta_values: np.ndarray,
    phi_values: np.ndarray,
) -> dict[str, float]:
    info = mutual_information(rho_se)
    chi_z = holevo_for_basis(rho_se, 0.0, 0.0)
    chi_h = holevo_for_basis(rho_se, 0.5 * math.pi, 0.0)

    best_chi = -np.inf
    best_beta = 0.0
    best_phi = 0.0
    for beta in beta_values:
        for phi in phi_values:
            chi = holevo_for_basis(rho_se, float(beta), float(phi))
            if chi > best_chi:
                best_chi = chi
                best_beta = float(beta)
                best_phi = float(phi)

    angle_from_z = min(best_beta, math.pi - best_beta)
    z_score = max(math.cos(0.5 * best_beta), math.sin(0.5 * best_beta))
    record_fraction = best_chi / info if info > 1e-12 else np.nan
    return {
        "mutual_info": info,
        "chi_best": best_chi,
        "chi_z": chi_z,
        "chi_hadamard": chi_h,
        "discord_z": info - chi_z,
        "discord_best": info - best_chi,
        "chi_gain_over_z": best_chi - chi_z,
        "best_beta": best_beta,
        "best_phi": best_phi,
        "best_angle_from_z": angle_from_z,
        "best_z_score": z_score,
        "best_record_fraction": record_fraction,
    }


def sbs_state(
    q: float,
    beta_s: float,
    phi_s: float,
    beta_e: float,
    phi_e: float,
) -> np.ndarray:
    s0, s1 = qubit_basis(beta_s, phi_s)
    e0, e1 = qubit_basis(beta_e, phi_e)
    return q * np.kron(projector(s0), projector(e0)) + (1.0 - q) * np.kron(projector(s1), projector(e1))


def minimize_sbs_distance(
    rho_se: np.ndarray,
    *,
    fixed_system_basis: str | None,
    seed: int,
    maxiter: int,
    popsize: int,
) -> float:
    if fixed_system_basis == "z":
        beta_s = 0.0
        phi_s = 0.0
    elif fixed_system_basis == "hadamard":
        beta_s = 0.5 * math.pi
        phi_s = 0.0
    elif fixed_system_basis is None:
        beta_s = None
        phi_s = None
    else:
        raise ValueError(f"Unknown fixed system basis: {fixed_system_basis}")

    if fixed_system_basis is None:
        bounds = [(0.5, 1.0), (0.0, math.pi), (0.0, 2.0 * math.pi), (0.0, math.pi), (0.0, 2.0 * math.pi)]

        def objective(x: np.ndarray) -> float:
            sigma = sbs_state(x[0], x[1], x[2], x[3], x[4])
            return trace_norm(rho_se - sigma)

    else:
        bounds = [(0.0, 1.0), (0.0, math.pi), (0.0, 2.0 * math.pi)]

        def objective(x: np.ndarray) -> float:
            sigma = sbs_state(x[0], beta_s, phi_s, x[1], x[2])
            return trace_norm(rho_se - sigma)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        polish=True,
        updating="immediate",
        workers=1,
    )
    return float(result.fun)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)


def finite_minmax(data: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return None, None
    return float(np.min(finite)), float(np.max(finite))


def plot_heatmap(
    path: Path,
    data: np.ndarray,
    alphas: np.ndarray,
    ps: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    extent = [float(alphas[0]), float(alphas[-1]), float(ps[0]), float(ps[-1])]
    kwargs = {}
    if symmetric:
        vmax = np.nanmax(np.abs(data))
        if np.isfinite(vmax) and vmax > 0:
            kwargs["norm"] = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        **kwargs,
    )
    ax.set_xlabel("alpha_2")
    ax.set_ylabel("environment mixedness p")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def theta_label(theta: float) -> str:
    known = {
        0.0: "0",
        math.pi / 8.0: "pi8",
        math.pi / 4.0: "pi4",
        0.9 * math.pi / 2.0: "0p9pi2",
    }
    for value, label in known.items():
        if abs(theta - value) < 1e-12:
            return label
    return f"{theta:.6g}".replace("-", "m").replace(".", "p")


def run_theta_scan(args, theta: float, out_dir: Path) -> dict:
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    ps = np.linspace(args.p_min, args.p_max, args.p_points)
    beta_values = np.linspace(0.0, math.pi, args.beta_count)
    phi_values = np.linspace(0.0, 2.0 * math.pi, args.phi_count, endpoint=False)

    metric_names = [
        "mutual_info",
        "chi_best",
        "chi_z",
        "chi_hadamard",
        "discord_z",
        "discord_best",
        "chi_gain_over_z",
        "best_beta",
        "best_phi",
        "best_angle_from_z",
        "best_z_score",
        "best_record_fraction",
    ]
    arrays = {name: np.full((len(ps), len(alphas)), np.nan, dtype=np.float64) for name in metric_names}

    if args.compute_sbs_delta:
        arrays["sbs_distance_z"] = np.full((len(ps), len(alphas)), np.nan, dtype=np.float64)
        arrays["sbs_distance_hadamard"] = np.full((len(ps), len(alphas)), np.nan, dtype=np.float64)
        arrays["sbs_delta_hadamard_minus_z"] = np.full((len(ps), len(alphas)), np.nan, dtype=np.float64)
    if args.compute_sbs_full:
        arrays["sbs_distance_full"] = np.full((len(ps), len(alphas)), np.nan, dtype=np.float64)

    started = time.time()
    total_points = len(alphas) * len(ps)
    completed_points = 0
    next_progress = started
    label = theta_label(theta)
    for ia, alpha2 in enumerate(alphas):
        h = total_hamiltonian(theta, float(alpha2))
        u = expm(-1j * args.time * h)
        for ip, p in enumerate(ps):
            rho_se = evolve_rho_se1_from_u(u, float(p))
            metrics = scan_holevo_basis(rho_se, beta_values, phi_values)
            for name, value in metrics.items():
                arrays[name][ip, ia] = value

            seed = args.sbs_seed + 100000 * ia + ip
            if args.compute_sbs_delta:
                d_z = minimize_sbs_distance(
                    rho_se,
                    fixed_system_basis="z",
                    seed=seed,
                    maxiter=args.sbs_maxiter,
                    popsize=args.sbs_popsize,
                )
                d_h = minimize_sbs_distance(
                    rho_se,
                    fixed_system_basis="hadamard",
                    seed=seed + 17,
                    maxiter=args.sbs_maxiter,
                    popsize=args.sbs_popsize,
                )
                arrays["sbs_distance_z"][ip, ia] = d_z
                arrays["sbs_distance_hadamard"][ip, ia] = d_h
                arrays["sbs_delta_hadamard_minus_z"][ip, ia] = d_h - d_z

            if args.compute_sbs_full:
                arrays["sbs_distance_full"][ip, ia] = minimize_sbs_distance(
                    rho_se,
                    fixed_system_basis=None,
                    seed=seed + 31,
                    maxiter=args.sbs_maxiter,
                    popsize=args.sbs_popsize,
                )

            completed_points += 1
            now = time.time()
            if not args.quiet and (now >= next_progress or completed_points == total_points):
                elapsed = now - started
                rate = completed_points / elapsed if elapsed > 0 else float("nan")
                remaining = (total_points - completed_points) / rate if rate > 0 else float("nan")
                extra = ""
                if args.compute_sbs_delta or args.compute_sbs_full:
                    extra = " sbs=on"
                print(
                    "progress "
                    f"theta={theta:.6g} "
                    f"points={completed_points}/{total_points} "
                    f"alpha={ia + 1}/{len(alphas)} p={ip + 1}/{len(ps)} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s "
                    f"rate={rate:.2f} pts/s{extra}",
                    flush=True,
                )
                next_progress = now + args.progress_interval

    npz_path = out_dir / f"theta_{label}_scan.npz"
    np.savez_compressed(
        npz_path,
        theta=np.array(theta),
        alpha2_values=alphas,
        p_values=ps,
        beta_values=beta_values,
        phi_values=phi_values,
        **arrays,
    )

    if not args.no_plots:
        plot_specs = [
            ("mutual_info", "I(S:E1)", "bits", "viridis", False),
            ("chi_best", "best Holevo chi_n(S:E1)", "bits", "viridis", False),
            ("chi_z", "Z-basis Holevo chi_Z(S:E1)", "bits", "viridis", False),
            ("discord_z", "Z-basis discord I - chi_Z", "bits", "magma", False),
            ("discord_best", "min grid discord I - max chi_n", "bits", "magma", False),
            ("chi_gain_over_z", "best chi_n - chi_Z", "bits", "viridis", False),
            ("best_record_fraction", "best record fraction chi_n / I", "fraction", "viridis", False),
            ("best_angle_from_z", "best basis angle from Z", "radians", "viridis", False),
            ("best_z_score", "Fig. 1(c)-style max(cos beta/2, sin beta/2)", "score", "viridis", False),
        ]
        if args.compute_sbs_delta:
            plot_specs.extend(
                [
                    ("sbs_distance_z", "restricted SBS trace norm distance: Z basis", "trace norm", "viridis", False),
                    (
                        "sbs_distance_hadamard",
                        "restricted SBS trace norm distance: Hadamard basis",
                        "trace norm",
                        "viridis",
                        False,
                    ),
                    (
                        "sbs_delta_hadamard_minus_z",
                        "SBS Delta = D_Hadamard - D_Z",
                        "trace norm difference",
                        "coolwarm",
                        True,
                    ),
                ]
            )
        if args.compute_sbs_full:
            plot_specs.append(("sbs_distance_full", "full SBS trace norm distance", "trace norm", "viridis", False))

        for metric, title, cbar_label, cmap, symmetric in plot_specs:
            plot_heatmap(
                out_dir / f"theta_{label}_{metric}.png",
                arrays[metric],
                alphas,
                ps,
                title=f"{title}, theta={theta:.6g}, t={args.time:g}",
                cbar_label=cbar_label,
                cmap=cmap,
                symmetric=symmetric,
            )

    stats = {name: dict(zip(["min", "max"], finite_minmax(value))) for name, value in arrays.items()}
    return {
        "theta": theta,
        "theta_label": label,
        "npz": str(npz_path),
        "metric_stats": stats,
        "elapsed_seconds": time.time() - started,
    }


def self_test() -> None:
    for theta in [0.0, math.pi / 8.0, math.pi / 4.0, 0.9 * math.pi / 2.0]:
        got = expm(-1j * h_c_inot(theta))
        expected = c_inot_unitary(theta)
        err = np.max(np.abs(got - expected))
        if err > 1e-12:
            raise AssertionError(f"C-INOT exp check failed for theta={theta}: {err}")

    h = total_hamiltonian(theta=0.0, alpha2=0.7)
    u = expm(-1j * h)
    rho_se = evolve_rho_se1_from_u(u, p=0.35)
    if rho_se.shape != (4, 4):
        raise AssertionError("rho_SE1 has wrong shape")
    if abs(np.trace(rho_se).real - 1.0) > 1e-12:
        raise AssertionError("rho_SE1 is not normalized")
    vals = np.linalg.eigvalsh(hermitize(rho_se))
    if np.min(vals.real) < -1e-11:
        raise AssertionError("rho_SE1 is not positive semidefinite")

    metrics = scan_holevo_basis(
        rho_se,
        beta_values=np.linspace(0.0, math.pi, 5),
        phi_values=np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False),
    )
    if not (metrics["chi_best"] + 1e-10 >= metrics["chi_z"] >= -1e-10):
        raise AssertionError("Holevo scan did not dominate Z basis")
    if metrics["chi_best"] - metrics["mutual_info"] > 1e-9:
        raise AssertionError("Holevo information exceeds mutual information")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone exact Mironowicz two-env-qubit mixed-state scanner.")
    parser.add_argument("--self-test", action="store_true", help="Run internal checks and exit.")
    parser.add_argument("--out-dir", help="Output directory. Default: data/mironowicz_mixed_exact/<timestamp>.")
    parser.add_argument("--theta", action="append", type=_as_float_expr, help="Theta value. Repeatable; accepts pi/8.")
    parser.add_argument(
        "--theta-set",
        choices=["fig1", "fig2"],
        default="fig1",
        help="fig1: theta=0. fig2: 0, pi/8, pi/4, 0.9*pi/2.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-points", type=int, default=41)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=0.5)
    parser.add_argument("--p-points", type=int, default=41)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--beta-count", type=int, default=31)
    parser.add_argument("--phi-count", type=int, default=64)
    parser.add_argument("--compute-sbs-delta", action="store_true", help="Compute Fig. 5-style restricted SBS Delta.")
    parser.add_argument("--compute-sbs-full", action="store_true", help="Compute full Eq. 15 SBS distance optimization.")
    parser.add_argument("--sbs-maxiter", type=int, default=80)
    parser.add_argument("--sbs-popsize", type=int, default=8)
    parser.add_argument("--sbs-seed", type=int, default=1234)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between progress/ETA prints.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    if args.theta:
        theta_values = args.theta
    elif args.theta_set == "fig2":
        theta_values = [0.0, math.pi / 8.0, math.pi / 4.0, 0.9 * math.pi / 2.0]
    else:
        theta_values = [0.0]

    if args.alpha_points < 2 or args.p_points < 2:
        raise ValueError("--alpha-points and --p-points must be at least 2")
    if args.beta_count < 2 or args.phi_count < 1:
        raise ValueError("--beta-count must be >= 2 and --phi-count must be >= 1")

    out_dir = Path(args.out_dir) if args.out_dir else Path("data") / "mironowicz_mixed_exact" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    theta_results = []
    for theta in theta_values:
        theta_results.append(run_theta_scan(args, float(theta), out_dir))

    manifest = {
        "script": Path(__file__).name,
        "paper": "Non-Perfect Propagation of Information to a Noisy Environment with Self-Evolution",
        "model": "three-qubit exact density matrix, S,E1,E2; trace out E2",
        "hamiltonian": "Eq. (6) with alpha1=alpha3=0: H_C-INOT^1 + H_C-INOT^2 + alpha2*(Z_E1 + Z_E2)",
        "initial_state": "|+><+|_S x rho_E1 x rho_E2, rho=p|0><0|+(1-p)|1><1|",
        "args": vars(args),
        "theta_results": theta_results,
        "elapsed_seconds": time.time() - started,
    }
    write_json(out_dir / "manifest.json", manifest)

    print("Mironowicz exact mixed-state scan complete")
    print(f"out_dir = {out_dir.resolve()}")
    for result in theta_results:
        print(f"theta={result['theta']:.6g}: {result['npz']}")


if __name__ == "__main__":
    main()
