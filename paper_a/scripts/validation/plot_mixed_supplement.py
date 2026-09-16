import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import apply_style


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [{k: (v if k == 'preparation' else float(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def select(rows, theta, p, preparation=None):
    return sorted([r for r in rows if abs(r['theta']-theta)<1e-10
                   and abs(r['p_mix']-p)<1e-10
                   and (preparation is None or r['preparation']==preparation)], key=lambda r:r['h'])


def main(data_root=Path('.'), output=None):
    paths = {
        'mixed_local': Path('analysis/mixed_state_comparison_4qubits/local_records.csv'),
        'matched_local': Path('analysis/matched_pure_mixed/local_records.csv'),
        'matched_sbs': Path('analysis/matched_pure_mixed/sbs_comparison.csv'),
        'size_scan': Path('analysis/mixed_basis_size/time_maxima.csv'),
    }
    paths = {key: data_root/path for key, path in paths.items()}
    data = {key: read(path) for key, path in paths.items()}
    apply_style()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.5))
    ax = axes[0,0]
    for p, color, style in [(0, '#111111', '-'), (.1, '#0072B2', '--'), (.25, '#D55E00', '-.')]:
        rows = select(data['mixed_local'], np.pi/4, p)
        ax.plot([r['h'] for r in rows], [r['separation'] for r in rows],
                color=color, ls=style, label=rf'$\mathcal P={p*p+(1-p)**2:g}$')
    ax.set_title('(a) Fixed initial direction', loc='left')
    ax.set_ylabel(r'Branch separation $1-B^2$')
    ax.legend(frameon=False)
    for ax, source, metric, title, ylabel in [
            (axes[0,1], 'matched_local', 'chi_z', '(b) Matched populations', r'$Z$-Holevo information (bits)'),
            (axes[1,0], 'matched_sbs', 'distance_free', '(c) Optimized SBS distance', r'$D_{\rm SBS}$')]:
        for preparation, color, style, label in [('pure','#0072B2','-',r'Pure superposition ($\mathcal P=1$)'),
                                                 ('mixed','#D55E00','--',r'Diagonal mixture ($\mathcal P=0.82$)')]:
            rows = select(data[source], np.pi/4, .1, preparation)
            ax.plot([r['h'] for r in rows], [r[metric] for r in rows], color=color, ls=style, label=label)
        ax.set_title(title, loc='left')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    ax = axes[1,1]
    for h, color, style in [(0., '#0072B2', '-'), (1.5, '#D55E00', '--'), (3., '#009E73', '-.')]:
        rows = sorted([r for r in data['size_scan'] if r['h'] == h], key=lambda r: r['environment_qubits'])
        ax.loglog([r['environment_qubits'] for r in rows], [r['peak_gain'] for r in rows],
                  color=color, ls=style, marker='o', markersize=3, label=rf'$h={h:g}$')
    guide_sizes = np.array([8, 256])
    ax.loglog(guide_sizes, .27/guide_sizes, ':', color='black', lw=1,
              label=r'$1/N_{\mathcal E}$ guide')
    ax.set_title(r'(d) Maximum gain over $Z$', loc='left')
    ax.set_ylabel('Holevo information gain (bits)')
    ax.set_xlabel(r'Environment size $N_{\mathcal E}$')
    ax.text(.97, .98, 'Diagonal mixture, $\\mathcal P=0.82$\nOne witness observed',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.legend(frameon=False, loc='lower left', fontsize=9)
    ax.set_xlim(1.7, 320)
    ax.set_ylim(.0007, .4)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.set_xlim(0, 3)
        ax.set_xlabel(r'Uniform field $h$ ($g=1$)')
    fig.tight_layout()
    output = (output or data_root/'figures/paper_a/supplement')/'mixed_state_comparison'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'))
    fig.savefig(output.with_suffix('.png'), dpi=200)
    plt.close(fig)
    manifest = dict(inputs={key:{'path':path.relative_to(data_root).as_posix(), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                            for key,path in paths.items()},
                    panels={'a': 'theta=pi/4; p_mix=0,0.1,0.25; N_E=3; one observed',
                            'b_c': 'theta=pi/4; p=p_mix=0.1; N_E=3; one observed',
                            'd': 'theta=0.9*pi/2; p_mix=0.1; N_E=2..256; one observed; h=0,1.5,3; numerical maximum over t=0..20 and system measurement axes'},
                    g=1, time_a_b_c=1, time_window_d=[0,20],
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
