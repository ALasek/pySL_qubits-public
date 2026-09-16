from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as effects
import numpy as np

from paper_plot_style import apply_style
from plot_theta_sweeps import ROOT, OUTPUT, alignment, memberships
from recalculate_exact import entropy, file_hash, seeded


def calculate():
    config = json.loads((OUTPUT/'config.json').read_text())
    theta = np.linspace(0, np.pi/2, 121)
    populations = np.linspace(0, 1, 101)
    capacity = alignment(populations[:, None], theta[None, :])
    capacity = np.clip(capacity, 0., 1.)
    times = np.arange(20., 41.)
    sites = memberships(config)[2]
    values = np.empty((len(config['run_seeds']), capacity.size))
    for k, seed in enumerate(config['run_seeds']):
        rng = seeded(seed, 'mironowicz.J_SE')
        g = np.array([rng.gauss(0, config['H_SE_J']) for _ in range(config['nqubits_E'])])
        oscillation = np.sin(np.pi*g[sites[k], None]*times/2)**2
        for start in range(0, capacity.size, 128):
            lam = capacity.ravel()[start:start+128, None, None, None]
            bf = np.sqrt(np.prod(1-lam*oscillation[None, ...], axis=2))
            information = entropy(bf)/np.log(2)
            values[k, start:start+128] = np.median(information.mean(axis=1), axis=1)
        print(f'Realization {k+1}/{len(values)}', flush=True)
    values = values.reshape(len(values), len(populations), len(theta))
    mean = values.mean(axis=0)
    errors = []
    with (OUTPUT/'summary.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['profile'] != 'uniform' or float(row['rms_over_g']) != 0:
                continue
            ip = int(np.argmin(abs(populations-float(row['p']))))
            it = int(np.argmin(abs(theta-float(row['theta']))))
            errors.append(abs(mean[ip, it]-float(row['chi_m2'])))
    assert max(errors) < 1e-12, max(errors)
    np.savez_compressed(OUTPUT/'preparation_map.npz', theta=theta, p=populations,
                        capacity=capacity, chi_mean=mean, chi_realizations=values,
                        times=times, seeds=config['run_seeds'], fragments_m2=sites)
    manifest = dict(p_points=len(populations), theta_points=len(theta), p_range=[0, 1],
                    theta_range=[0, np.pi/2], parameter_cases=capacity.size,
                    source_config='analysis/theta_sweeps/config.json',
                    realizations=len(values), n_environment=config['nqubits_E'],
                    fragment_size=2, fragments_per_realization=sites.shape[1],
                    time_window=[20, 40], time_step=1, g_std=config['H_SE_J'],
                    calculation='Direct no-field overlap at each grid point; no interpolation of information values.',
                    reduction='Fragment mean, time median within each realization, realization mean; information in bits, H_Z(S)=1.',
                    overlay_maximum='p=(1-cos(theta))/2; Lambda=1',
                    overlay_minimum='p=(1+sin(theta))/2; Lambda=0; also the isolated endpoint p=0, theta=pi/2',
                    previous_slice_points_checked=len(errors), max_previous_slice_difference=max(errors),
                    hashes={str(path.relative_to(ROOT)): file_hash(path) for path in
                            (Path(__file__), OUTPUT/'config.json', OUTPUT/'preparation_map.npz',
                             Path(__file__).with_name('recalculate_exact.py'),
                             Path(__file__).with_name('plot_theta_sweeps.py'))})
    (OUTPUT/'preparation_map_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def plot(output=None, update_manifest=True):
    with np.load(OUTPUT/'preparation_map.npz') as data:
        theta, p, capacity, information = (data[key] for key in ('theta', 'p', 'capacity', 'chi_mean'))
    apply_style()
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'legend.fontsize': 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), sharex=True,
                             gridspec_kw={'width_ratios': [1, 1, 1.05]})
    angles = np.linspace(0, np.pi/2, 501)
    for ax, values, title, label in zip(axes[:2], (capacity, information),
                                      (r'(a) Recording capacity $\Lambda$', '(b) Fragment information'),
                                      (r'$\Lambda(p,\theta)$', r'$\chi_z/H_Z(\mathcal{S})$')):
        mesh = ax.pcolormesh(theta/np.pi, p, values, cmap='cividis', vmin=0, vmax=1,
                             shading='nearest', rasterized=True)
        for y, style, text in (((1-np.cos(angles))/2, '-', r'Maximum ($\Lambda=1$)'),
                               ((1+np.sin(angles))/2, '--', r'Blind alignment ($\Lambda=0$)')):
            line, = ax.plot(angles/np.pi, y, color='black', ls=style, lw=1.15, label=text)
            line.set_path_effects([effects.Stroke(linewidth=2.1, foreground='white'), effects.Normal()])
        ax.set_title(title, loc='left')
        ax.set_xlabel(r'Interaction angle $\theta$')
        ax.set_xlim(0, .5)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, .25, .5], ['0', r'$\pi/4$', r'$\pi/2$'])
        ax.set_yticks([0, .25, .5, .75, 1])
        fig.colorbar(mesh, ax=ax, pad=.025, fraction=.05, label=label)
    axes[0].set_ylabel(r'Preparation population $p$')
    axes[1].tick_params(labelleft=False)
    cuts = []
    for population, color, style, label_x, label_y in (
            (.15, '#D55E00', '-', .18, .87),
            (.25, '#CC79A7', '--', .40, .85),
            (.75, '#009E73', '-.', .35, .10)):
        index = int(np.argmin(abs(p-population)))
        assert np.isclose(p[index], population, rtol=0, atol=1e-12)
        axes[1].axhline(population, color=color, ls=style, lw=1.3)
        axes[2].plot(theta/np.pi, information[index], color=color, ls=style, lw=1.6)
        axes[2].text(label_x, label_y, fr'$p={population:.2f}$', color=color,
                     ha='center')
        cuts.append({'p': float(p[index]), 'grid_index': index})
    axes[2].set_title('(c) Holevo information', loc='left')
    axes[2].set_xlabel(r'Interaction angle $\theta$')
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([0, .25, .5, .75, 1])
    axes[2].grid(axis='y', alpha=.2, lw=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.04), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93))
    stem = (output or ROOT/'figures/paper_a')/'theta_preparation_map'
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)
    if not update_manifest:
        return
    path = OUTPUT/'preparation_map_manifest.json'
    manifest = json.loads(path.read_text())
    script = str(Path(__file__).relative_to(ROOT))
    manifest['calculation_script_sha256'] = manifest['hashes'].pop(script, manifest.get('calculation_script_sha256'))
    manifest['plot_script_sha256'] = file_hash(Path(__file__))
    manifest['panel_c_cuts'] = cuts
    manifest['panel_c_source'] = 'Exact rows of chi_mean in preparation_map.npz; same reduction as panel b, no smoothing or new simulation.'
    manifest['figure_hashes'] = {str(stem.with_suffix(s).relative_to(ROOT)): file_hash(stem.with_suffix(s)) for s in ('.pdf', '.png')}
    path.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replot', action='store_true')
    args = parser.parse_args()
    if not args.replot:
        calculate()
    plot()
