import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import differential_evolution, minimize


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
PAULI = np.array([X, Y, Z])
LEFT = np.array([np.kron(a, I) for a in PAULI])
RIGHT = np.array([np.kron(I, a) for a in PAULI])
PAIR = np.array([[np.kron(a, b) for b in PAULI] for a in PAULI])
THETAS = [0., np.pi / 8, np.pi / 4, .45 * np.pi]
MIXEDNESS = [0., .1, .25, .4, .5]
BOUNDS = [(-1., 1.), (0., np.pi), (0., 2 * np.pi),
          (0., np.pi), (0., 2 * np.pi)]


def entropy(rho):
    vals = np.linalg.eigvalsh(rho)
    if vals.min() < -1e-10:
        raise ValueError(f"Nonpositive density matrix: {vals.min()}")
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log2(vals)))


def binary_entropy(p):
    vals = np.clip([p, 1 - p], 0, 1)
    vals = vals[vals > 1e-15]
    return float(-np.sum(vals * np.log2(vals)))


def axis(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi), np.cos(theta)])


def branch_data(theta, h, p, t=1., environment_qubits=2, initial_coherence=0.):
    if environment_qubits < 1:
        raise ValueError('At least one observed environment qubit is required')
    rho = np.diag([p, 1 - p]).astype(complex)
    rho[0, 1] = rho[1, 0] = initial_coherence * np.sqrt(p*(1-p))
    u0 = expm(-1j * t * h * Z)
    u1 = expm(-1j * t * (np.pi / 2 * (I - np.cos(theta) * X
                                    - np.sin(theta) * Z) + h * Z))
    a, b = u0 @ rho @ u0.conj().T, u1 @ rho @ u1.conj().T
    cross = u0 @ rho @ u1.conj().T
    gamma = np.trace(cross)
    remainder = gamma ** (environment_qubits - 1)
    reduced = .5 * np.block([[a, remainder * cross],
                             [remainder.conjugate() * cross.conj().T, b]])
    k, az = np.pi / 2 * np.cos(theta), h - np.pi / 2 * np.sin(theta)
    omega = np.hypot(k, az)
    q = float((k * t * np.sinc(omega * t / np.pi)) ** 2)
    eta = 1 - 2 * p
    fidelity2 = float((np.trace(a @ b)
                       + 2 * np.sqrt(max(0., np.linalg.det(a).real
                                        * np.linalg.det(b).real))).real)
    chi = entropy((a + b) / 2) - entropy(rho)
    r = np.array([2*initial_coherence*np.sqrt(p*(1-p)), 0., 2*p-1])
    relative = u0.conj().T @ u1
    relative /= np.sqrt(np.linalg.det(relative))
    rotation = np.array([(0.5j*np.trace(sigma @ relative)).real for sigma in PAULI])
    separation = float(np.linalg.norm(np.cross(rotation, r))**2)
    radius = float(np.linalg.norm(r))
    analytic_chi = (binary_entropy((1 + np.sqrt(max(0., radius**2-separation))) / 2)
                    - binary_entropy((1 + radius) / 2))
    values = dict(theta=theta, h=h, p_mix=p, t=t, eta=eta, Q=q,
                  fidelity2=fidelity2, separation=1 - fidelity2,
                  chi_z=chi, coherence=float(abs(remainder)),
                  local_decoherence=float(abs(gamma)),
                  system_coherence=float(abs(gamma) ** environment_qubits),
                  fidelity_error=abs(fidelity2 - (1 - separation)),
                  holevo_error=abs(chi - analytic_chi))
    return reduced, values


def full_evolution(theta, h, p, t, environment_qubits=2, initial_coherence=0.):
    p1 = (I - Z) / 2
    gate = np.pi / 2 * (I - np.cos(theta) * X - np.sin(theta) * Z)
    dim = 2 ** environment_qubits
    gate_sum, field_sum = np.zeros((dim, dim), complex), np.zeros((dim, dim), complex)
    for j in range(environment_qubits):
        left, right = np.eye(2**j), np.eye(2**(environment_qubits-j-1))
        gate_sum += np.kron(left, np.kron(gate, right))
        field_sum += np.kron(left, np.kron(h*Z, right))
    hamiltonian = np.kron(p1, gate_sum) + np.kron(I, field_sum)
    env = np.diag([p, 1 - p])
    env[0, 1] = env[1, 0] = initial_coherence*np.sqrt(p*(1-p))
    initial = (I + X) / 2
    for _ in range(environment_qubits):
        initial = np.kron(initial, env)
    unitary = expm(-1j * t * hamiltonian)
    evolved = unitary @ initial @ unitary.conj().T
    rest = 2 ** (environment_qubits - 1)
    return np.trace(evolved.reshape(4, rest, 4, rest), axis1=1, axis2=3)


def sbs_state(params):
    w, ts, ps, te, pe = params
    n, m = axis(ts, ps), axis(te, pe)
    return .25 * (np.eye(4) + w * (np.einsum('i,ijk->jk', n, LEFT)
                                  + np.einsum('i,ijk->jk', m, RIGHT))
                  + np.einsum('i,j,ijkl->kl', n, m, PAIR))


def distance(rho, params):
    return float(np.abs(np.linalg.eigvalsh(rho - sbs_state(params))).sum())


def measurement_holevo(rho, theta, phi):
    n = axis(theta, phi)
    projectors = [(I + np.einsum('i,ijk->jk', sign * n, PAULI)) / 2
                  for sign in (1, -1)]
    shaped = rho.reshape(2, 2, 2, 2)
    env = np.trace(shaped, axis1=0, axis2=2)
    conditional_entropy = 0.
    for projector in projectors:
        block = np.einsum('ba,aibj->ij', projector, shaped)
        prob = np.trace(block).real
        if prob > 1e-14:
            conditional_entropy += prob * entropy(block / prob)
    return entropy(env) - conditional_entropy


def optimize_sbs(rho, seed, starts, fixed_z=False, initial=None, global_check=False):
    rng = np.random.default_rng(seed)
    bounds = [BOUNDS[i] for i in (0, 3, 4)] if fixed_z else BOUNDS

    def expand(x):
        return np.array([x[0], 0., 0., x[1], x[2]]) if fixed_z else np.array(x)

    def objective(x):
        return distance(rho, expand(x))

    guesses = []
    for w, ts, ps, te, pe in [(0, 0, 0, np.pi, 0),
                              (1, np.pi / 2, 0, np.pi, 0),
                              (0, np.pi / 2, np.pi / 2, np.pi / 2, 0)]:
        guesses.append(np.array([w, te, pe] if fixed_z else [w, ts, ps, te, pe]))
    if initial is not None:
        guesses.append(np.array(initial)[[0, 3, 4]] if fixed_z else np.array(initial))
    while len(guesses) < starts:
        guesses.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
    candidates, successes = [], 0
    for guess in guesses:
        candidates.append((objective(guess), guess))
        fit = minimize(objective, guess, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 350, 'ftol': 1e-12, 'gtol': 1e-7})
        candidates.append((float(fit.fun), fit.x))
        successes += int(fit.success)
    if global_check:
        fit = differential_evolution(objective, bounds, seed=seed + 100000,
                                     popsize=14, maxiter=250, tol=1e-9,
                                     atol=1e-10, polish=False)
        candidates.append((float(fit.fun), fit.x))
    best, parameters = min(candidates, key=lambda item: item[0])
    fit = minimize(objective, parameters, method='Powell', bounds=bounds,
                   options={'maxiter': 500, 'xtol': 1e-8, 'ftol': 1e-11})
    if fit.fun < best:
        best, parameters = float(fit.fun), fit.x
    return best, expand(parameters), successes


def solve_case(task):
    index, theta, h, p, starts, seed, validation, environment_qubits = task
    rho, values = branch_data(theta, h, p, environment_qubits=environment_qubits)
    return fit_sbs(rho, values, index, starts, seed, validation)


def fit_sbs(rho, values, index, starts, seed, validation=True):
    dz, xz, sz = optimize_sbs(rho, seed + index * 2, starts, fixed_z=True)
    df, xf, sf = optimize_sbs(rho, seed + index * 2 + 1, starts, initial=xz)
    gap_z = gap_free = 0.
    if validation:
        dz2, xz2, _ = optimize_sbs(rho, seed + 50000 + index * 2, starts * 2,
                                  fixed_z=True, initial=xz, global_check=True)
        df2, xf2, _ = optimize_sbs(rho, seed + 50001 + index * 2, starts * 2,
                                  initial=xf, global_check=True)
        gap_z, gap_free = max(0., dz - dz2), max(0., df - df2)
        if dz2 < dz:
            dz, xz = dz2, xz2
        if df2 < df:
            df, xf = df2, xf2
    if dz < df:
        df, xf = dz, xz.copy()
    values.update(distance_z=dz, distance_free=df, basis_gain=dz-df,
                  axis_abs_z=abs(np.cos(xf[1])), weight=(xf[0]+1)/2,
                  chi_sbs_basis=measurement_holevo(rho, xf[1], xf[2]),
                  validation=int(validation), validation_gap_z=gap_z,
                  validation_gap_free=gap_free, successful_starts_z=sz,
                  successful_starts_free=sf)
    for label, params in [('z', xz), ('free', xf)]:
        for name, value in zip(['w', 'theta_s', 'phi_s', 'theta_e', 'phi_e'], params):
            values[f'{label}_{name}'] = float(value)
    return values


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--environment-qubits', type=int, choices=range(1, 9), default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--starts', type=int, default=10)
    parser.add_argument('--field-points', type=int, default=21)
    args = parser.parse_args()
    if args.output is None:
        suffix = '' if args.environment_qubits == 2 else f'_{args.environment_qubits+1}qubits'
        args.output = Path(f'analysis/mixed_state_comparison{suffix}')
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dense = [branch_data(theta, h, p, environment_qubits=args.environment_qubits)[1]
             for theta in THETAS for p in MIXEDNESS
             for h in np.linspace(0, 3, 301)]
    write_csv(args.output / 'local_records.csv', dense)
    checks = []
    rng = np.random.default_rng(20260914)
    for _ in range(64):
        theta, h, p, t = rng.uniform(0, np.pi/2), rng.uniform(0, 3), rng.uniform(0, .5), rng.uniform(0, 4)
        rho, metrics = branch_data(theta, h, p, t, args.environment_qubits)
        checks.append(float(np.max(np.abs(rho-full_evolution(theta, h, p, t, args.environment_qubits)))))
        assert abs(np.trace(rho)-1) < 1e-12
        assert np.linalg.eigvalsh(rho).min() > -1e-12
        assert abs(measurement_holevo(rho, 0, 0)-metrics['chi_z']) < 1e-12
    assert max(checks) < 1e-12
    assert max(r['fidelity_error'] for r in dense) < 1e-12
    assert max(r['holevo_error'] for r in dense) < 1e-12
    seed = 20260914
    fields = np.linspace(0, 3, args.field_points)
    tasks = []
    for theta in THETAS:
        for p in MIXEDNESS:
            for j, h in enumerate(fields):
                validation = True
                tasks.append((len(tasks), theta, h, p, args.starts, seed, validation, args.environment_qubits))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(solve_case, tasks, chunksize=1):
            rows.append(row)
            if len(rows) % 20 == 0:
                write_csv(args.output / 'sbs_distances.csv', rows)
                print(f'{len(rows)}/{len(tasks)} SBS cases, {time.perf_counter()-started:.1f} s', flush=True)
    write_csv(args.output / 'sbs_distances.csv', rows)
    manifest = dict(model=f'S + {args.environment_qubits} independent environment qubits; one observed',
                    environment_qubits=args.environment_qubits, total_qubits=args.environment_qubits+1,
                    observed_environment_qubits=1, traced_environment_qubits=args.environment_qubits-1,
                    coherence_convention='Twice the off-diagonal block trace norm = |gamma|^(N_E-1)',
                    source_doi='10.3390/e24040467', source_equations=[5, 6, 7, 8, 9, 15, 17],
                    g=1., alpha1=0., alpha3=0., time=1., field_range=[0., 3.],
                    theta_values=THETAS, p_mix_values=MIXEDNESS,
                    distance_convention='Trace norm, without factor 1/2, matching source Eq. 17',
                    sbs_parameterization='Two orthogonal pure local states on each qubit; mixture weight [0,1]',
                    optimized_distances='Numerical candidate minima (upper bounds); no global certificate',
                    starts=args.starts, workers=args.workers, seed=seed,
                    validation='Independent doubled starts plus differential evolution at every SBS grid point',
                    total_cases=len(rows), validation_cases=sum(r['validation'] for r in rows),
                    max_validation_improvement_z=max(r['validation_gap_z'] for r in rows),
                    max_validation_improvement_free=max(r['validation_gap_free'] for r in rows),
                    full_matrix_checks=64, max_reduced_state_error=max(checks),
                    max_fidelity_error=max(r['fidelity_error'] for r in dense),
                    max_holevo_error=max(r['holevo_error'] for r in dense),
                    elapsed_seconds=time.perf_counter()-started,
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
