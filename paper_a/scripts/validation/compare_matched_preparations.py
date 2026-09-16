import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mixed_state_comparison import THETAS, branch_data, fit_sbs, full_evolution, write_csv
from paper_plot_style import apply_style


POPULATIONS = [.1, .25, .5]
FIELDS = ['theta', 'h', 'p_mix', 'chi_z', 'separation', 'coherence',
          'distance_free', 'distance_z', 'basis_gain']


def solve(task):
    index, theta, h, p = task
    rho, metrics = branch_data(theta, h, p, environment_qubits=3, initial_coherence=1.)
    return fit_sbs(rho, metrics, index, 10, 20260914)


def comparison_figure(rows, output, metric=None):
    apply_style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10})
    columns = THETAS if metric else [THETAS[i] for i in [0, 2, 3]]
    titles = {THETAS[0]: r'$\theta=0$', THETAS[1]: r'$\theta=\pi/8$',
              THETAS[2]: r'$\theta=\pi/4$', THETAS[3]: r'$\theta=0.9\pi/2$'}
    labels = {'chi_z': r'$Z$-Holevo information (bits)',
              'separation': r'Branch separation $1-B^2$',
              'distance_free': r'Free-basis SBS distance'}
    panels = [(p, metric) for p in POPULATIONS] if metric else [(.25, k) for k in labels]
    fig, axes = plt.subplots(3, len(columns), figsize=(11 if metric else 9, 7),
                             sharex=True, sharey='row')
    for row_index, (p, field) in enumerate(panels):
        for col, theta in enumerate(columns):
            ax = axes[row_index, col]
            for preparation, color, style, label in [
                    ('pure', '#0072B2', '-', 'Pure, coherent'),
                    ('mixed', '#D55E00', '--', 'Mixed, diagonal')]:
                selected = sorted([r for r in rows if r['preparation']==preparation
                                   and abs(r['p_mix']-p)<1e-10 and abs(r['theta']-theta)<1e-10
                                   and field in r], key=lambda r:r['h'])
                ax.plot([r['h'] for r in selected], [r[field] for r in selected],
                        color=color, linestyle=style, lw=1.8, label=label)
            ax.grid(alpha=.2)
            ax.set_xlim(0, 3)
            if field in ['chi_z', 'separation']:
                ax.set_ylim(-.025, 1.025)
            if row_index == 0:
                ax.set_title(titles[theta])
            if row_index == 2:
                ax.set_xlabel(r'Uniform field $h$ ($g=1$)')
        axes[row_index, 0].set_ylabel((f'$p={p:g}$\n' if metric else '')+labels[field])
    handles, legend_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5,.955),
               ncol=2, frameon=False)
    title = r'Matched populations: four total qubits, one observed, $t=1$'
    if metric is None:
        title = r'Matched $p=0.25$: four total qubits, one observed, $t=1$'
    fig.suptitle(title, y=.995)
    fig.tight_layout(rect=(0,0,1,.90))
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=Path('analysis/matched_pure_mixed'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    baseline_path = Path('analysis/mixed_state_comparison_4qubits/sbs_distances.csv')
    with baseline_path.open(encoding='utf-8') as stream:
        baseline = [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    dense = []
    for theta in THETAS:
        for p in POPULATIONS:
            for h in np.linspace(0, 3, 301):
                for preparation, coherence in [('pure', 1.), ('mixed', 0.)]:
                    _, metrics = branch_data(theta, h, p, environment_qubits=3, initial_coherence=coherence)
                    metrics['preparation'] = preparation
                    dense.append(metrics)
    write_csv(args.output/'local_records.csv', dense)
    rng = np.random.default_rng(20260914)
    errors = []
    for _ in range(64):
        theta,h,p,t = rng.uniform(0,np.pi/2),rng.uniform(0,3),rng.uniform(0,.5),rng.uniform(0,4)
        rho,_ = branch_data(theta,h,p,t,3,1.)
        errors.append(float(np.max(np.abs(rho-full_evolution(theta,h,p,t,3,1.)))))
    assert max(errors) < 1e-12
    assert max(r['fidelity_error'] for r in dense) < 1e-12
    assert max(r['holevo_error'] for r in dense) < 1e-12
    tasks = [(theta,h,p) for theta in THETAS for p in POPULATIONS for h in np.linspace(0,3,21)]
    pure = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(solve, [(i,*task) for i,task in enumerate(tasks)]):
            pure.append(r)
            if len(pure)%20 == 0:
                print(f'{len(pure)}/{len(tasks)} pure SBS cases; {time.perf_counter()-start:.1f}s',flush=True)
    write_csv(args.output/'pure_sbs.csv', pure)
    combined = []
    for preparation, records in [('pure',pure),('mixed',baseline)]:
        for r in records:
            if r['p_mix'] in POPULATIONS:
                combined.append(dict(preparation=preparation, **{k:r[k] for k in FIELDS}))
    write_csv(args.output/'sbs_comparison.csv', combined)
    comparison_figure(dense, args.output/'matched_information', 'chi_z')
    comparison_figure(combined, args.output/'matched_sbs', 'distance_free')
    overview = [{k:v for k,v in r.items() if k not in ['chi_z','separation','coherence']}
                for r in combined]+dense
    comparison_figure(overview, args.output/'matched_overview')
    manifest = dict(environment_qubits=3, observed_environment_qubits=1, time=1.,g=1.,
                    populations=POPULATIONS, theta_values=THETAS, field_range=[0.,3.],
                    preparation='Same populations; off-diagonal entry sqrt(p(1-p)) for pure and zero for mixed',
                    newly_computed_pure_sbs_cases=len(pure), reused_mixed_sbs_cases=len(combined)-len(pure),
                    starts=10, independent_validation_starts=20, differential_evolution_each_point=True,
                    numerical_minima='Best feasible candidates, no global certificate',
                    max_optimizer_refinement_free=max(r['validation_gap_free'] for r in pure),
                    max_optimizer_refinement_z=max(r['validation_gap_z'] for r in pure),
                    max_full_evolution_error=max(errors),
                    max_fidelity_error=max(r['fidelity_error'] for r in dense),
                    max_holevo_error=max(r['holevo_error'] for r in dense),
                    elapsed_seconds=time.perf_counter()-start, baseline=str(baseline_path),
                    baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    solver_sha256=hashlib.sha256(Path('scripts/validation/mixed_state_comparison.py').read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
