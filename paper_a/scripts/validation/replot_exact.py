from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Run existing CPU plotters against the exact derived-data mirror.')
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--notes-root', type=Path, required=True)
    parser.add_argument('--simulation-root', type=Path, required=True)
    args = parser.parse_args()
    data, plots = args.work_root/'data', args.work_root/'plots'
    plots.mkdir(exist_ok=True)
    notes, sim = args.notes_root/'scripts', args.simulation_root/'analysis_scripts'
    batch = lambda name: data/('Paper_A_final_'+name)
    # The plateau plotter reports a solver-health diagnostic in addition to exact information.
    for run in (batch('HolevoPlateauFullFragments')/'runs').iterdir():
        record_path = run/'recalculation.json'
        record = json.loads(record_path.read_text())
        with np.load(run/'results.npz') as archive:
            if 'modified_norms' in archive:
                continue
            arrays = dict(archive)
        with np.load(Path(record['source'])/'results.npz') as original:
            arrays['modified_norms'] = original['modified_norms']
        np.savez_compressed(run/'results.npz', **arrays)
        record['plot_health_keys_copied_from_original'] = ['modified_norms']
        record['derived_results_sha256'] = hashlib.sha256((run/'results.npz').read_bytes()).hexdigest()
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        comparison_path = args.work_root/'exact_comparison.json'
        comparison = json.loads(comparison_path.read_text())
        comparison['runs'] = [record if row['run_id']==run.name else row for row in comparison['runs']]
        comparison_path.write_text(json.dumps(comparison, indent=2)+'\n')
    jobs = [
        ('field', sim/'paper_a_field_profile_control.py', ['--batch-dir', batch('FieldProfileControl'), '--output-dir', plots/'field', '--bootstrap', 2000]),
        ('submission', sim/'paper_a_submission_panels.py', ['--data-root', data, '--output-dir', plots/'submission', '--families', 'lambda', 'stability', '--strict-missing']),
        ('matched_sweep', notes/'plot_matched_lambda_fragment_sweep.py', [batch('SubmissionMatchedLambda'), plots/'matched_sweep']),
        ('matched_late', notes/'plot_matched_lambda_late_window.py', [plots/'submission/paper_a_late_window_points.csv', plots/'matched_late', '--batch-manifest', batch('SubmissionMatchedLambda')/'batch_manifest.json']),
        ('plateau', notes/'plot_holevo_plateau_overview.py', ['--batch-dir', batch('HolevoPlateauFullFragments'), '--output-dir', plots/'plateau']),
        ('endpoints', notes/'plot_stability_endpoint_supplement.py', ['--data-root', data, '--analyzer', sim/'paper_a_submission_panels.py', '--output-dir', plots/'endpoints']),
        ('higher_n', sim/'paper_a_higher_n_validation.py', ['--high-n-batch-dir', batch('HigherNValidation'), '--matched-lambda-dir', batch('SubmissionMatchedLambda'), '--field-control-dir', batch('FieldProfileControl'), '--output-dir', plots/'higher_n']),
        ('lambda_fixed', notes/'plot_lambda_collapse.py', ['--data-dir', data, '--output-dir', plots/'lambda_fixed', '--time', 36, '--no-time-maps']),
        ('iid_thresholds', notes/'plot_fragment_threshold_scaling.py', ['--groups', args.notes_root/'generated/lambda_collapse/lambda_collapse_groups.csv', '--output-dir', plots/'iid_thresholds']),
    ]
    records = []
    for name, script, arguments in jobs:
        command = [sys.executable, str(script), *map(str, arguments)]
        if name == 'matched_sweep':
            command.insert(1, str(Path(__file__).with_name('plot_exact_fragment_sweep.py')))
        print(f'Plotting {name}', flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        (plots/f'{name}.log').write_text(result.stdout+'\n'+result.stderr)
        records.append({'name': name, 'command': command, 'returncode': result.returncode,
                        'script_sha256_lf': hashlib.sha256(script.read_text().encode()).hexdigest()})
        (plots/'plotting_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {plots/name}.log: {result.stderr[-1500:]}')
    print(f'Completed {len(records)} plot jobs', flush=True)


if __name__ == '__main__':
    main()
