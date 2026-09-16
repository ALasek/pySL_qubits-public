from pathlib import Path
import argparse
import csv
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import apply_style
from recalculate_exact import entropy, file_hash, fragment_members, squared_overlaps


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'analysis/theta_sweeps'
PREPARATIONS = (0., .25, .5)
STRENGTHS = (0., .5, 2., 10.)
COLORS = ('#222222', '#0072B2', '#D55E00', '#009E73')


def alignment(p, theta):
    return 1 - (2*np.sqrt(p*(1-p))*np.cos(theta) + (2*p-1)*np.sin(theta))**2


def memberships(config):
    n, nr, cap = config['nqubits_E'], len(config['run_seeds']), config['fragment_sample_count']
    counts = np.zeros((n+1, 1, nr), dtype=int)
    for m in (2, 8):
        counts[m] = min(cap, math.comb(n, m))
    members = fragment_members(config, counts)
    return {m: members[m, :, :counts[m, 0, 0], :m]-1 for m in (2, 8)}


def evaluate(config, members, theta, p, profile, strength, times):
    params = dict(config, Mironowicz_theta=theta, psi_bias=p,
                  Mironowicz_h0=strength*config['H_SE_J'] if profile == 'uniform' else 0.,
                  Mironowicz_alpha2=strength*config['H_SE_J'] if profile == 'gaussian' else 0.)
    result = np.empty((len(config['run_seeds']), 3))
    for k, seed in enumerate(config['run_seeds']):
        b2 = squared_overlaps(params, seed, times)
        for column, m in enumerate((2, 8)):
            chi = entropy(np.sqrt(np.prod(b2[members[m][k]], axis=1)))/np.log(2)
            result[k, column] = np.median(np.mean(chi, axis=0))
            if m == 8:
                result[k, 2] = np.min(np.quantile(chi, .1, axis=0))
    return result


def summarize(values, draws):
    resampled = values[draws].mean(axis=1)
    low, high = np.quantile(resampled, [.025, .975], axis=0)
    return values.mean(axis=0), low, high


def calculate(config):
    members = memberships(config)
    theta = np.linspace(0, np.pi/2, 121)
    times = np.arange(20., 41.)
    cases, values = [], []
    for p in PREPARATIONS:
        for profile in ('uniform', 'gaussian'):
            for strength in STRENGTHS:
                if profile == 'gaussian' and strength == 0:
                    continue
                for angle in theta:
                    cases.append((p, profile, strength, angle))
                    values.append(evaluate(config, members, angle, p, profile, strength, times))
                print(f'p={p:g}, {profile}, rms/g={strength:g}', flush=True)
    values = np.array(values)
    draws = np.random.default_rng(20260915).integers(0, len(config['run_seeds']), (2000, len(config['run_seeds'])))
    rows = []
    for case, data in zip(cases, values):
        mean, low, high = summarize(data, draws)
        row = dict(zip(('p', 'profile', 'rms_over_g', 'theta'), case))
        for col, metric in enumerate(('chi_m2', 'chi_m8', 'min_q10_m8')):
            row.update({metric: mean[col], metric+'_low': low[col], metric+'_high': high[col]})
        rows.append(row)
    with (OUTPUT/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(OUTPUT/'realizations.npz', values=values, times=times,
                        seeds=config['run_seeds'], fragments_m2=members[2], fragments_m8=members[8])
    # A finer time grid checks the reported medians and the sampled-time minimum separately.
    checks = []
    for p, profile, strength, angle in ((.25, 'uniform', 0., np.pi/3),
                                      (0., 'uniform', 2., np.pi/4),
                                      (.25, 'gaussian', 2., np.pi/3),
                                      (.5, 'uniform', 10., np.pi/4)):
        coarse = evaluate(config, members, angle, p, profile, strength, times)
        fine = evaluate(config, members, angle, p, profile, strength, np.arange(20., 40.01, .5))
        checks.append(dict(p=p, profile=profile, rms_over_g=strength, theta=angle,
                           mean_difference=(fine-coarse).mean(axis=0).tolist(),
                           max_realization_difference=np.max(abs(fine-coarse), axis=0).tolist()))
    manifest = dict(config=config, theta_count=len(theta), time_window=[20, 40], time_step=1,
                    preparations=list(PREPARATIONS), field_rms_over_g=list(STRENGTHS),
                    cases=len(cases), metrics=['chi_m2', 'chi_m8', 'min_q10_m8'],
                    reduction='Fragment mean, time median within realization, mean over realizations. Minimum metric: fragment q10, time minimum, realization mean.',
                    intervals='Pointwise 95% bootstrap intervals, 2000 whole-realization resamples, seed 20260915.',
                    finer_time_grid_checks=checks,
                    hashes={str(path.relative_to(ROOT)): file_hash(path) for path in
                            (Path(__file__), Path(__file__).with_name('recalculate_exact.py'), OUTPUT/'summary.csv', OUTPUT/'realizations.npz')})
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return rows


def select(rows, p, profile, strength):
    if strength == 0:
        profile = 'uniform'
    return sorted((r for r in rows if r['p'] == p and r['profile'] == profile
                   and r['rms_over_g'] == strength), key=lambda r: r['theta'])


def curve(ax, rows, color, label, style='-'):
    x = np.array([r['theta']/np.pi for r in rows])
    y, lo, hi = (np.array([r[key] for r in rows]) for key in ('chi_m2', 'chi_m2_low', 'chi_m2_high'))
    ax.plot(x, y, color=color, lw=1.5, ls=style, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)


def finish(fig, axes, path):
    for ax in axes.flat:
        ax.set_xlim(0, .5)
        ax.set_ylim(-.025, 1.025)
        ax.set_xticks([0, .125, .25, .375, .5], ['0', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
        ax.set_xlabel(r'Interaction angle $\theta$')
        ax.grid(alpha=.16)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=200)
    plt.close(fig)


def plot_no_field(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    angles = np.linspace(0, np.pi/2, 481)
    for p, color, label, style in zip(PREPARATIONS, COLORS, (r'$p=0$', r'$p=1/4$', r'$p=1/2$'), ('-', '--', '-.')):
        axes[0].plot(angles/np.pi, alignment(p, angles), color=color, ls=style, lw=1.5, label=label)
        curve(axes[1], select(rows, p, 'uniform', 0.), color, label, style)
    axes[0].set_title('(a) Recording capacity', loc='left')
    axes[0].set_ylabel(r'$\Lambda(p,\theta)$')
    axes[0].legend(frameon=False, loc='center right')
    axes[1].set_title('(b) Information in two-qubit fragments', loc='left')
    axes[1].set_ylabel(r'$\chi_z/H_Z(\mathcal{S})$')
    finish(fig, axes, ROOT/'figures/paper_a/theta_record_information')


def plot_fields(rows, output=None):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True, sharey=True)
    for i, profile in enumerate(('uniform', 'gaussian')):
        ax = axes[i]
        for strength, color, style in zip(STRENGTHS, COLORS, ('-', '--', '-.', ':')):
            curve(ax, select(rows, .5, profile, strength), color, rf'${strength:g}$', style)
        name = 'Uniform' if profile == 'uniform' else 'Gaussian'
        ax.set_title(f'({chr(97+i)}) {name} fields', loc='left')
        ax.set_xlabel(r'Interaction angle $\theta$')
    axes[0].set_ylabel(r'$\chi_z/H_Z(\mathcal{S})$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=r'$h_{\rm rms}/g$', loc='upper center',
               bbox_to_anchor=(.53, 1.15), ncol=4, frameon=False)
    for ax in axes.flat:
        ax.set_xlim(0, .5)
        ax.set_ylim(-.025, 1.025)
        ax.set_xticks([0, .25, .5], ['0', r'$\pi/4$', r'$\pi/2$'])
        ax.grid(alpha=.16)
    fig.tight_layout()
    target = (output or ROOT/'figures/paper_a/supplement')/'theta_field_information'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(target.with_suffix('.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot', action='store_true')
    parser.add_argument('--field-only', action='store_true', help='Render only the field-comparison figure.')
    args = parser.parse_args()
    config = json.loads((OUTPUT/'config.json').read_text())
    if args.replot:
        with (OUTPUT/'summary.csv').open() as stream:
            rows = [{k: v if k == 'profile' else float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    else:
        rows = calculate(config)
    apply_style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 9})
    if not args.field_only:
        plot_no_field(rows)
    plot_fields(rows)
    manifest_path = OUTPUT/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['figure_hashes'] = {
        str(path.relative_to(ROOT)): file_hash(path)
        for stem in ('theta_record_information', 'supplement/theta_field_information')
        for suffix in ('.pdf', '.png')
        for path in [ROOT/'figures/paper_a'/f'{stem}{suffix}']}
    manifest['plot_style_sha256'] = file_hash(Path(__file__).with_name('paper_plot_style.py'))
    manifest['plot_script_sha256'] = file_hash(Path(__file__))
    manifest['field_figure_preparations'] = [.5]
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
