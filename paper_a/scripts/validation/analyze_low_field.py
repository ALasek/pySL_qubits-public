from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np

from recalculate_exact import recalculate, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--original-exact', type=Path, required=True)
    parser.add_argument('--simulator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.simulator / 'analysis_scripts'))
    import paper_a_field_profile_control as field

    args.output.mkdir(parents=True, exist_ok=True)
    field.EXPECTED_RATIOS['aligned'] = (0., .05, .1, .15, .2, .25, .5, 1., 2., 5., 10., 20., 30.)
    reports, points, health = [], [], []
    rng = np.random.default_rng(20260907)
    sources = sorted((args.source / 'runs').glob('*/params.json'))
    if len(sources) != 8:
        raise ValueError(f'Expected eight new cases, found {len(sources)}')
    for params in sources:
        source = params.parent
        target = args.output / 'exact' / 'runs' / source.name
        signature = {name: file_hash(source / name) for name in ('params.json', 'results.npz')}
        signature['exact_script_sha256'] = file_hash(Path(__file__).with_name('recalculate_exact.py'))
        report_path = target / 'recalculation.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        if report is None or report['signature'] != signature:
            report = recalculate(source, target, signature)
        reports.append(report)
        print(source.name, report['Holevo_Z_S_Ef_max_mean_difference_bits'], flush=True)
    roots = [args.original_exact / 'runs', args.output / 'exact' / 'runs']
    for root in roots:
        for params in sorted(root.glob('*/params.json')):
            p = json.loads(params.read_text())
            if p['field_geometry'] != 'aligned':
                continue
            point, check = field._load_point(params.parent, (30., 60.), rng, 2000, 1)
            with np.load(params.parent / 'results.npz') as data:
                mean = data['Holevo_Z_S_Ef_fractionsT'] / np.log(2)
                rmean = field._redundancy(mean, np.ones(61), 16)
                point.metrics['redundancy_mean'] = field._bootstrap(field._late_values(rmean, np.arange(61) >= 30), rng, 2000)
                q = data['Holevo_Z_S_Ef_fractionsT_quantiles'][np.argmin(abs(data['fragment_quantile_levels']-.1))] / np.log(2)
                point.metrics['persistent_fraction'] = field._bootstrap(np.min(q[8, 30:], axis=0) >= .9, rng, 2000)
            points.append(point)
            health.append(check)
    cases = field._case_map(points)
    if len(cases) != 25 or len(points) != 25:
        raise ValueError('Expected 25 distinct aligned cases including the shared zero-field point')
    seeds = points[0].params['run_seeds']
    if any(p.params['run_seeds'] != seeds for p in points):
        raise ValueError('Realization seeds do not match')
    field._write_csv(args.output / 'metrics.csv', field._metric_rows(points))
    field._write_csv(args.output / 'paired.csv', field._paired_rows(points, rng, 2000))

    # Retain the original panel definitions; a separate mean-threshold column is in the CSV.
    def strength_axis(axis, ratios):
        axis.set_xscale('symlog', linthresh=.25, linscale=1.3, base=10)
        ticks = [0, .1, .25, .5, 1, 2, 5, 10, 30]
        axis.set_xticks(ticks)
        axis.set_xticklabels([f'{x:g}' for x in ticks], fontsize=8)
        axis.set_xlabel(r'$h_{\rm rms}/g$', labelpad=2)
        axis.grid(alpha=.2)
    field._set_strength_axis = strength_axis
    field._plot_aligned(points, args.output)
    manifest = {
        'source': str(args.source), 'original_exact': str(args.original_exact),
        'analysis_script_sha256': file_hash(Path(__file__)),
        'field_analysis_sha256': file_hash(args.simulator / 'analysis_scripts/paper_a_field_profile_control.py'),
        'new_cases': reports, 'combined_aligned_cases': len(points),
        'window': [30, 60], 'bootstrap_draws': 2000, 'bootstrap_seed': 20260907,
        'interval_scope': 'Realizations only, conditional on sampled fragment identities; not nested fragment resampling.',
        'central_estimator': 'Mean of realization-level time medians; persistence is mean of realization-level time minima of fragment q10.',
        'redundancy': 'Time median conditional on crossing at evaluated m<=8; NaN when no crossing.',
        'health': health,
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for row in field._metric_rows(points):
        if row['field_strength_ratio'] <= .5:
            print({k: row[k] for k in ('field_profile', 'field_strength_ratio', 'holevo_fhalf', 'holevo_fhalf_min', 'redundancy_mean', 'redundancy_q10', 'redundancy_q10_n', 'persistent_fraction')})


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        main()
