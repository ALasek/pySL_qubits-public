"""Bounded time scan of the best system measurement basis for one witness."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import minimize

try:
    from mixed_state_comparison import I, PAIR, PAULI, X, Z, axis, branch_data, entropy, full_evolution, measurement_holevo
    from paper_plot_style import save_figure
except ModuleNotFoundError:
    from .mixed_state_comparison import I, PAIR, PAULI, X, Z, axis, branch_data, entropy, full_evolution, measurement_holevo
    from .paper_plot_style import save_figure


P_MIX = 0.1
THETAS = (np.pi / 4, .45 * np.pi)
FIELDS = (0., 1.5, 3.)
ENVIRONMENT_SIZES = (2, 3, 8)
TIME_STEP = .1
GAIN_TOLERANCE = 1e-6


def h2_from_radius(radius):
    q = np.clip((1 + np.asarray(radius)) / 2, 0., 1.)
    out = np.zeros_like(q, dtype=float)
    mask = (q > 1e-15) & (q < 1 - 1e-15)
    out[mask] = -q[mask] * np.log2(q[mask]) - (1 - q[mask]) * np.log2(1 - q[mask])
    return out


def bloch_data(rho):
    r = np.array([np.trace(rho @ np.kron(sigma, I)).real for sigma in PAULI])
    s = np.array([np.trace(rho @ np.kron(I, sigma)).real for sigma in PAULI])
    tensor = np.array([[np.trace(rho @ PAIR[i, j]).real for j in range(3)] for i in range(3)])
    return r, s, tensor, float(h2_from_radius(np.linalg.norm(s)))


def holevo_bloch(data, beta, azimuth):
    r, s, tensor, env_entropy = data
    beta, azimuth = np.broadcast_arrays(np.asarray(beta), np.asarray(azimuth))
    n = np.stack((np.sin(beta) * np.cos(azimuth),
                  np.sin(beta) * np.sin(azimuth), np.cos(beta)), axis=-1)
    rn = np.einsum('...i,i->...', n, r)
    tn = np.einsum('...i,ij->...j', n, tensor)
    plus_den, minus_den = 1 + rn, 1 - rn
    plus = (s + tn) / np.maximum(plus_den[..., None], 1e-14)
    minus = (s - tn) / np.maximum(minus_den[..., None], 1e-14)
    conditional = .5 * (plus_den * h2_from_radius(np.linalg.norm(plus, axis=-1))
                         + minus_den * h2_from_radius(np.linalg.norm(minus, axis=-1)))
    return env_entropy - conditional


def mutual_information(rho):
    shaped = rho.reshape(2, 2, 2, 2)
    system = np.trace(shaped, axis1=1, axis2=3)
    witness = np.trace(shaped, axis1=0, axis2=2)
    return entropy(system) + entropy(witness) - entropy(rho)


def separated_starts(beta_grid, phi_grid, values, limit=3):
    starts = [(0., 0.), (np.pi / 2, 0.), (np.pi / 2, np.pi / 2),
              (np.pi / 2, np.pi), (np.pi / 2, 3 * np.pi / 2)]
    vectors = [axis(beta, phi) for beta, phi in starts]
    for flat in np.argsort(values.ravel())[::-1]:
        beta, phi = float(beta_grid.ravel()[flat]), float(phi_grid.ravel()[flat])
        vector = axis(beta, phi)
        if all(abs(np.dot(vector, old)) < np.cos(np.pi / 12) for old in vectors):
            starts.append((beta, phi))
            vectors.append(vector)
        if len(starts) >= limit + 5:
            break
    return starts


def optimize_measurement(rho, polar_points=9, azimuth_points=32):
    data = bloch_data(rho)
    polar = np.linspace(0., np.pi / 2, polar_points)
    azimuth = np.linspace(0., 2 * np.pi, azimuth_points, endpoint=False)
    beta_grid, phi_grid = np.meshgrid(polar, azimuth, indexing='ij')
    grid_values = holevo_bloch(data, beta_grid, phi_grid)
    chi_z = float(holevo_bloch(data, 0., 0.))
    candidates = [(chi_z, 0., 0., 'z')]
    grid_index = int(np.argmax(grid_values))
    candidates.append((float(grid_values.ravel()[grid_index]),
                       float(beta_grid.ravel()[grid_index]), float(phi_grid.ravel()[grid_index]), 'grid'))
    for beta, phi in separated_starts(beta_grid, phi_grid, grid_values):
        result = minimize(lambda x: -float(holevo_bloch(data, x[0], x[1])),
                          (beta, phi), method='L-BFGS-B',
                          bounds=((0., np.pi / 2), (0., 2 * np.pi)),
                          options={'ftol': 1e-14, 'gtol': 1e-11, 'maxiter': 300})
        candidates.append((-float(result.fun), float(result.x[0]), float(result.x[1]), 'local'))
    chi_best, beta, phi, source = max(candidates, key=lambda item: item[0])
    phi %= 2 * np.pi
    return dict(chi_z=chi_z, chi_best=chi_best, gain_raw=chi_best-chi_z,
                best_polar=beta, best_azimuth=phi,
                grid_best=float(np.max(grid_values)), optimum_source=source,
                starts=len(candidates), max_grid_polar=polar_points,
                max_grid_azimuth=azimuth_points)


def direct_z_initial(theta, h, p, t, environment_qubits, system_one):
    gate = np.pi / 2 * (I - np.cos(theta) * X - np.sin(theta) * Z)
    dim = 2 ** environment_qubits
    gate_sum = np.zeros((dim, dim), complex)
    field_sum = np.zeros((dim, dim), complex)
    for j in range(environment_qubits):
        left, right = np.eye(2 ** j), np.eye(2 ** (environment_qubits - j - 1))
        gate_sum += np.kron(left, np.kron(gate, right))
        field_sum += np.kron(left, np.kron(h * Z, right))
    hamiltonian = np.kron((I - Z) / 2, gate_sum) + np.kron(I, field_sum)
    env = np.diag([p, 1 - p]).astype(complex)
    state = (I - Z) / 2 if system_one else (I + Z) / 2
    for _ in range(environment_qubits):
        state = np.kron(state, env)
    evolved = expm(-1j * t * hamiltonian) @ state
    evolved = evolved @ expm(1j * t * hamiltonian)
    rest = 2 ** (environment_qubits - 1)
    return np.trace(evolved.reshape(2, 2, rest, 2, 2, rest), axis1=2, axis2=5).reshape(4, 4)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rounded(value):
    return round(float(value), 12)


def enrich(theta, h, n_env, t, initial_coherence=0.):
    rho, branch = branch_data(theta, h, P_MIX, t, n_env, initial_coherence)
    result = optimize_measurement(rho)
    info = mutual_information(rho)
    dz = info - result['chi_z']
    result.update(theta=theta, theta_over_pi=theta / np.pi, h=h,
                  environment_qubits=n_env, total_qubits=n_env + 1, t=t,
                  p_mix=P_MIX, initial_purity=rounded(P_MIX**2 + (1 - P_MIX)**2
                                                      + 2 * initial_coherence**2 * P_MIX * (1 - P_MIX)),
                  initial_coherence=initial_coherence, mutual_information=info,
                  D_Z=dz, complement_coherence=branch['coherence'],
                  gain_leq_DZ_margin=dz-result['gain_raw'],
                  gain_plot=max(0., result['gain_raw']),
                  angle_defined=int(result['gain_raw'] > GAIN_TOLERANCE))
    return result


def enrich_task(task):
    return enrich(*task)


def recurrence_checks(rows):
    by_key = {(rounded(r['theta']), r['environment_qubits'], rounded(r['t'])): r
              for r in rows if r['h'] == 0.}
    checks = []
    for theta in THETAS:
        for n_env in ENVIRONMENT_SIZES:
            odd = [by_key[(rounded(theta), n_env, float(t))]['gain_raw'] for t in range(1, 20, 2)]
            even = [by_key[(rounded(theta), n_env, float(t))]['gain_raw'] for t in range(0, 21, 2)]
            first = by_key[(rounded(theta), n_env, 1.)]['gain_raw']
            checks.append(dict(theta=theta, theta_over_pi=theta / np.pi,
                               environment_qubits=n_env, gain_t1=first,
                               max_odd_recurrence_deviation=max(abs(x-first) for x in odd),
                               max_even_gain=max(abs(x) for x in even)))
    return checks


def make_plot(rows, output):
    fig, axes = plt.subplots(len(ENVIRONMENT_SIZES), len(FIELDS), figsize=(8.2, 6.1), sharex=True, sharey=True)
    gain_upper = max(.23, 1.1 * max(row['gain_plot'] for row in rows))
    for row_index, n_env in enumerate(ENVIRONMENT_SIZES):
        for col_index, h in enumerate(FIELDS):
            ax = axes[row_index, col_index]
            for theta, color, label in zip(THETAS, ('C0', 'C3'), (r'$\theta=\pi/4$', r'$\theta=.45\pi$')):
                selected = [r for r in rows if r['environment_qubits'] == n_env and r['h'] == h and r['theta'] == theta]
                ax.plot([r['t'] for r in selected], [r['gain_plot'] for r in selected], color=color, label=label)
                tiny = [r for r in selected if not r['angle_defined']]
                if tiny:
                    ax.scatter([r['t'] for r in tiny], [0.] * len(tiny), color=color, marker='|', s=16)
            if row_index == 0:
                ax.set_title(rf'$h={h:g}$')
            if col_index == 0:
                ax.set_ylabel(rf'$N_{{\mathcal{{E}}}}={n_env}$' + '\n' + r'$\chi_{\rm best}-\chi_Z$')
            if row_index == len(ENVIRONMENT_SIZES) - 1:
                ax.set_xlabel(r'$t$')
            ax.set_xlim(0, 20)
            ax.set_ylim(-1e-5, gain_upper)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(r'One-witness basis gain; $p=.1$ (initial purity $.82$)')
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        save_figure(fig, output / f'basis_gain_time{suffix}', dpi=220 if suffix == '.png' else None,
                    bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('analysis/mixed_basis_time'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    times = np.round(np.arange(0., 20. + TIME_STEP / 2, TIME_STEP), 12)
    tasks = [(theta, h, n_env, t) for theta in THETAS for h in FIELDS
             for n_env in ENVIRONMENT_SIZES for t in times]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(enrich_task, tasks, chunksize=8))
    write_csv(args.output / 'time_scan.csv', rows)
    pure_times = (0., 1., 10., 20.)
    pure_tasks = [(theta, h, n_env, t, 1.) for theta in THETAS for h in FIELDS
                  for n_env in ENVIRONMENT_SIZES for t in pure_times]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pure_rows = list(pool.map(enrich_task, pure_tasks, chunksize=8))
    write_csv(args.output / 'pure_controls.csv', pure_rows)
    measurement_errors = []
    samples = (rows[0], rows[len(rows)//3], rows[2*len(rows)//3], rows[-1],
               max(rows, key=lambda item: item['gain_raw']))
    for row in samples:
        rho, _ = branch_data(row['theta'], row['h'], P_MIX, row['t'], row['environment_qubits'])
        data = bloch_data(rho)
        for beta, phi in ((0., 0.), (row['best_polar'], row['best_azimuth']), (np.pi/2, np.pi/7)):
            measurement_errors.append(abs(float(holevo_bloch(data, beta, phi))
                                          - measurement_holevo(rho, beta, phi)))

    reduced_errors = []
    for theta, h, t, n_env, coherence in ((THETAS[0], 1.5, .7, 2, 0.), (THETAS[1], 3., 4.3, 2, 1.),
                                           (THETAS[0], 1.5, .7, 3, 0.), (THETAS[1], 3., 4.3, 3, 1.)):
        rho, _ = branch_data(theta, h, P_MIX, t, n_env, coherence)
        reduced_errors.append(float(np.max(np.abs(rho - full_evolution(theta, h, P_MIX, t, n_env, coherence)))))
    z_errors = []
    for theta, h, t, n_env in ((THETAS[0], 1.5, .7, 2), (THETAS[1], 3., 4.3, 2),
                                (THETAS[0], 1.5, .7, 3), (THETAS[1], 3., 4.3, 3)):
        for system_one in (False, True):
            direct = direct_z_initial(theta, h, P_MIX, t, n_env, system_one)
            branch, _ = branch_data(theta, h, P_MIX, t, n_env)
            expected = np.zeros((4, 4), complex)
            if system_one:
                expected[2:, 2:] = 2 * branch[2:, 2:]
            else:
                expected[:2, :2] = 2 * branch[:2, :2]
            z_errors.append(float(np.max(np.abs(direct - expected))))
    recurrence = recurrence_checks(rows)
    write_csv(args.output / 'h0_recurrence.csv', recurrence)
    max_gain_excess = max(row['gain_raw'] - row['D_Z'] for row in rows)
    max_pure_gain = max(abs(row['gain_raw']) for row in pure_rows)
    assert max(measurement_errors) < 2e-12
    assert max_gain_excess < 2e-10
    assert max_pure_gain < 1e-8
    assert max(check['max_odd_recurrence_deviation'] for check in recurrence) < 2e-10
    assert max(check['max_even_gain'] for check in recurrence) < 2e-10
    max_row = max(rows, key=lambda row: row['gain_raw'])
    positive = [row for row in rows if row['gain_raw'] > GAIN_TOLERANCE]
    transition = min(positive, key=lambda row: row['gain_raw']) if positive else max_row
    refinement_rows = []
    for label, row in (('maximum', max_row), ('near_transition', transition)):
        rho, _ = branch_data(row['theta'], row['h'], P_MIX, row['t'], row['environment_qubits'])
        refined = optimize_measurement(rho, polar_points=17, azimuth_points=96)
        refined.update(label=label, theta=row['theta'], h=row['h'], environment_qubits=row['environment_qubits'],
                       t=row['t'], coarse_chi_best=row['chi_best'], coarse_gain=row['gain_raw'],
                       refined_gain_raw=refined['gain_raw'])
        refinement_rows.append(refined)
    write_csv(args.output / 'grid_refinement.csv', refinement_rows)
    make_plot(rows, args.output)
    manifest = dict(
        purpose='Exploratory finite-time one-witness measurement-basis scan; numerical optima are not certified global.',
        model='S plus N_E independent environment qubits; exactly one observed witness',
        p_mix=P_MIX, initial_mixed_purity=.82, thetas=list(map(float, THETAS)), fields=list(FIELDS),
        environment_qubits=list(ENVIRONMENT_SIZES), observed_environment_qubits=1,
        time_range=[0., 20.], time_step=TIME_STEP, measurements='Axes modulo opposite labels: polar beta 0..pi/2 and azimuth 0..2pi.',
        coarse_grid=[9, 32], refinement_grid=[17, 96], local_optimization='Separated top-grid candidates plus Z/equator starts; L-BFGS-B. No global certificate.',
        workers=args.workers,
        gain_tolerance=GAIN_TOLERANCE, complement_coherence='Unobserved-witness factor |gamma|^(N_E-1).',
        cases=len(rows), pure_control_cases=len(pure_rows), pure_control_max_abs_gain=max_pure_gain,
        max_gain=max_row['gain_raw'], max_gain_case={key: max_row[key] for key in ('theta', 'h', 'environment_qubits', 't')},
        max_gain_minus_DZ=max_gain_excess, measurement_holevo_vectorized_checks=len(measurement_errors),
        max_measurement_holevo_vectorized_error=max(measurement_errors),
        full_evolution_reduced_checks=len(reduced_errors), max_full_evolution_reduced_error=max(reduced_errors),
        direct_z_initial_checks=len(z_errors), max_direct_z_initial_error=max(z_errors),
        recurrence_checks=recurrence, refinement=refinement_rows,
        elapsed_seconds=time.perf_counter()-started, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('mixed_state_comparison.py', 'paper_plot_style.py')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    readme = """# Finite-time mixed-basis pilot\n\nThis bounded exploratory scan retains one environment witness. It evaluates diagonal mixed initial environment states with `p=.1` (purity `.82`), interaction angles `pi/4` and `.45 pi`, fields `h=0,1.5,3`, and `N_E=2,3,8`, at `t=0..20` in steps of `.1`.\n\n`time_scan.csv` reports Z Holevo information, the best sampled-and-locally-refined measurement Holevo information, their raw gain, optimum axis, `D_Z=I(S:E)-chi_Z`, and residual complement coherence. The best axis is numerical only: the scan covers a `9 x 32` hemisphere grid, then refines separated high candidates with explicit Z/equator starts. Axes are equivalent under opposite outcome labels. `angle_defined=0` masks axes whose gain is at most `1e-6`.\n\n`pure_controls.csv` is the matched-population coherent control (`initial_coherence=1`) at selected times. `h0_recurrence.csv` checks the exact `h=0` period-two recurrence: odd integer times are compared with `t=1`, and even integer gains with zero. `grid_refinement.csv` repeats a maximum-gain and near-transition point on a `17 x 96` grid. `manifest.json` contains validation errors and runtime.\n\nThe pilot concerns finite time and one witness only. It does not test asymptotic behavior, redundancy, fragment scaling, SBS distances, or a certified global measurement optimum.\n\nReproduce from the repository root with `python scripts/validation/scan_mixed_basis_time.py`.\n"""
    (args.output / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
