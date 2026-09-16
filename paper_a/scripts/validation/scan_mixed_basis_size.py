import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from mixed_state_comparison import binary_entropy, branch_data, write_csv
from scan_mixed_basis_time import bloch_data, enrich, holevo_bloch
from paper_plot_style import apply_style


SIZES = (2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
THETA = .45 * np.pi
FIELDS = (0., 1.5, 3.)
P = .1


def angular_grid(theta, h, n_env, t, polar_points=9, azimuth_points=64):
    rho, local = branch_data(theta, h, P, t, n_env)
    data = bloch_data(rho)
    beta, phi = np.meshgrid(np.linspace(0., np.pi/2, polar_points),
                           np.linspace(0., 2*np.pi, azimuth_points, endpoint=False), indexing='ij')
    values = holevo_bloch(data, beta, phi)
    index = np.argmax(values)
    return dict(t=float(t), gain=max(0., float(values.flat[index])-local['chi_z']),
                beta=float(beta.flat[index]), phi=float(phi.flat[index]))


def time_envelope(task):
    h, n_env, refinement = task
    step = .05 if refinement else .1
    # Resolve peaks approaching revivals as size grows; a fixed time grid can mimic exponential decay.
    offsets = np.geomspace(1e-5, min(1., 4/np.sqrt(n_env)), 70 if refinement else 40)
    revival_centers = np.arange(0., 20.1, 2.) if h == 0 else (0.,)
    extra = [center + sign*offset for center in revival_centers for sign in (-1, 1) for offset in offsets
             if 0 <= center + sign*offset <= 20]
    times = np.unique(np.r_[np.arange(0., 20.+step/2, step), extra])
    grid = [angular_grid(THETA, h, n_env, t, 17 if refinement else 9,
                         96 if refinement else 64) for t in times]
    peaks = [i for i, row in enumerate(grid) if row['gain'] > 1e-13
             and (i == 0 or row['gain'] >= grid[i-1]['gain'])
             and (i == len(grid)-1 or row['gain'] >= grid[i+1]['gain'])]
    peaks = sorted(peaks, key=lambda i: grid[i]['gain'], reverse=True)
    best = max(grid, key=lambda r: r['gain']).copy()
    evaluated = 0
    for i in peaks[:32]:
        row = grid[i]
        lo, hi = times[max(0, i-1)], times[min(len(times)-1, i+1)]

        def objective(x):
            rho, local = branch_data(THETA, h, P, x[0], n_env)
            return -(float(holevo_bloch(bloch_data(rho), x[1], x[2]))-local['chi_z'])

        fit = minimize(objective, (row['t'], row['beta'], row['phi']),
                       method='L-BFGS-B', bounds=((lo, hi), (0., np.pi/2), (None, None)),
                       options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10})
        evaluated += fit.nfev
        if -fit.fun > best['gain']:
            best = dict(t=float(fit.x[0]), gain=float(-fit.fun),
                        beta=float(fit.x[1]), phi=float(fit.x[2] % (2*np.pi)))
    final = enrich(THETA, h, n_env, best['t'])
    assert final['gain_raw'] >= best['gain'] - 2e-10
    assert final['gain_raw'] <= final['D_Z'] + 2e-10
    return dict(h=h, environment_qubits=n_env, theta=THETA, p_mix=P,
                peak_time=best['t'], peak_gain=final['gain_raw'], chi_z=final['chi_z'],
                D_Z=final['D_Z'], complement_coherence=final['complement_coherence'],
                best_polar=final['best_polar'], best_azimuth=final['best_azimuth'],
                time_samples=len(times), angular_grid_polar=17 if refinement else 9,
                angular_grid_azimuth=96 if refinement else 64, refined_peaks=min(len(peaks), 32),
                optimizer_evaluations=evaluated, dense_check=int(refinement))


def fixed_case(task):
    theta, h, n_env, t = task
    row = enrich(theta, h, n_env, t)
    _, local = branch_data(theta, h, P, t, n_env)
    row['local_decoherence'] = local['local_decoherence']
    assert row['gain_raw'] <= row['D_Z'] + 2e-10
    return row


def check_bounds(fixed, peaks):
    errors, exponential_excess, uniform_excess = [], [], []
    for row in fixed + peaks:
        n = row['environment_qubits']
        t = row.get('t', row.get('peak_time'))
        _, local = branch_data(row['theta'], row['h'], P, t, n)
        z = local['local_decoherence']
        exact_dz = binary_entropy((1+z**n)/2) - binary_entropy((1+z**(n-1))/2)
        gain = row.get('gain_raw', row.get('peak_gain'))
        errors.append(abs(row['D_Z']-exact_dz))
        exponential_excess.append(gain-z**(2*(n-1)))
        uniform_excess.append(gain-1/(2*(n-1)*np.log(2)))
    assert max(errors) < 2e-10
    assert max(exponential_excess) < 2e-10
    assert max(uniform_excess) < 2e-10
    return dict(cases=len(errors), max_entropy_identity_error=max(errors),
                max_exponential_bound_excess=max(exponential_excess),
                max_uniform_bound_excess=max(uniform_excess))


def plot(fixed, peaks, output):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    for h, color in zip(FIELDS, ('#0072B2', '#D55E00', '#009E73')):
        selected = [r for r in fixed if r['h'] == h and r['theta'] == THETA]
        sizes = [r['environment_qubits'] for r in selected]
        gains = [max(0., r['gain_raw']) for r in selected]
        axes[0].semilogx(sizes, gains, 'o-', color=color, label=rf'$h={h:g}$')
        axes[1].semilogy(sizes, [r['complement_coherence'] for r in selected], 'o-', color=color)
        selected = [r for r in peaks if r['h'] == h]
        axes[2].loglog([r['environment_qubits'] for r in selected],
                       [r['peak_gain'] for r in selected], 'o-', color=color)
    n = np.array(SIZES)
    axes[2].loglog(n, .27/n, ':', color='black', label=r'$0.27/N_{\mathcal E}$ guide')
    axes[0].set_title(r'(a) Advantage at fixed $t=1$')
    axes[1].set_title(r'(b) Coherence factor at $t=1$')
    axes[2].set_title(r'(c) Maximum over time, $0\leq t\leq20$')
    axes[0].set_ylabel(r'$\chi_{\rm best}-\chi_Z$ (bits)')
    axes[1].set_ylabel(r'$|\gamma(1)|^{N_{\mathcal E}-1}$')
    axes[2].set_ylabel(r'Maximum advantage (bits)')
    axes[0].set_ylim(-.005, .22)
    axes[1].set_ylim(1e-30, 1)
    for ax in axes:
        ax.set_xlabel(r'$N_{\mathcal E}$')
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.suptitle(r'One observed witness; diagonal mixture $\mathcal P=0.82$, $\theta=0.45\pi$')
    fig.tight_layout()
    fig.savefig(output/'size_dependence.pdf', bbox_inches='tight')
    fig.savefig(output/'size_dependence.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('analysis/mixed_basis_size'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tasks = [(theta, h, n, 1.) for theta in (np.pi/4, THETA) for h in FIELDS for n in SIZES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fixed = list(pool.map(fixed_case, tasks))
    write_csv(args.output/'fixed_time.csv', fixed)
    print(f'Fixed-time cases complete: {len(fixed)}', flush=True)
    peaks = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(time_envelope, (h, n, False)) for h in FIELDS for n in SIZES]
        for future in as_completed(futures):
            peaks.append(future.result())
            if len(peaks) % 8 == 0:
                print(f'Time maxima complete: {len(peaks)}/{len(futures)}', flush=True)
    peaks.sort(key=lambda r: (r['h'], r['environment_qubits']))
    write_csv(args.output/'time_maxima.csv', peaks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        dense = list(pool.map(time_envelope, [(h, n, True) for h in FIELDS for n in (8, 64, 256)]))
    for row in dense:
        coarse = next(r for r in peaks if r['h']==row['h'] and r['environment_qubits']==row['environment_qubits'])
        row['coarse_peak_gain'] = coarse['peak_gain']
        row['gain_difference'] = row['peak_gain']-coarse['peak_gain']
    write_csv(args.output/'refinement.csv', dense)
    fits = []
    for h in FIELDS:
        selected = [r for r in peaks if r['h']==h and r['environment_qubits']>=32]
        n = np.array([r['environment_qubits'] for r in selected])
        log_gain = np.log([r['peak_gain'] for r in selected])
        slope, intercept = np.polyfit(np.log(n), log_gain, 1)
        exp_slope, exp_intercept = np.polyfit(n, log_gain, 1)
        fits.append(dict(h=h, fit_min_size=32, power_exponent=float(slope),
                         power_log_rmse=float(np.sqrt(np.mean((log_gain-intercept-slope*np.log(n))**2))),
                         exponential_rate=float(-exp_slope),
                         exponential_log_rmse=float(np.sqrt(np.mean((log_gain-exp_intercept-exp_slope*n)**2)))))
    plot(fixed, peaks, args.output)
    manifest = dict(p_mix=P, initial_purity=.82, observed_witnesses=1, coupling=1,
                    sizes=list(SIZES), fields=list(FIELDS), fixed_times=[1.],
                    fixed_time_thetas=[float(np.pi/4), float(THETA)], envelope_theta=float(THETA),
                    time_window=[0,20], coarse_time_step=.1,
                    adaptive_grid='Geometric offsets toward zero; also toward exact even-integer revivals when h=0.',
                    optimizer='Hemisphere grid and joint local time/axis refinement; no global certificate.',
                    fixed_cases=len(fixed), envelope_cases=len(peaks), refinement_cases=len(dense),
                    dense_max_gain_difference=max(abs(r['gain_difference']) for r in dense), fits=fits,
                    bound_checks=check_bounds(fixed, peaks),
                    elapsed_seconds=time.perf_counter()-start,
                    sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in (Path(__file__).name, 'scan_mixed_basis_time.py',
                                          'mixed_state_comparison.py', 'paper_plot_style.py')})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
