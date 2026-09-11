from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recalculate_exact import entropy, file_hash, fragment_members, squared_overlaps, _validate


BATCHES = ('Paper_A_final_LambdaCollapse', 'Paper_A_final_FragmentThresholdIntermediateN',
           'Paper_A_final_FragmentThresholdFiniteN')
LAMBDAS = (.75, .933012702, 1.)


def redundancy(q):
    crosses = q >= .9
    first = np.argmax(crosses, axis=0) + 1
    return np.where(np.any(crosses, axis=0), 2*q.shape[0]/first, np.nan)


def summarize(values, rng):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {'mean': None, 'low': None, 'high': None, 'defined': 0, 'total': len(values)}
    draws = np.mean(finite[rng.integers(0, len(finite), (2000, len(finite)))], axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return {'mean': float(np.mean(finite)), 'low': float(low), 'high': float(high),
            'defined': len(finite), 'total': len(values)}


def load_case(path):
    p = json.loads((path/'params.json').read_text())
    _validate(p)
    if p['Mironowicz_h0'] != 0 or p['Mironowicz_alpha2'] != 0:
        raise ValueError('Expected no self-fields')
    n = p['nqubits_E']
    time_index = round(36/p['printT'])
    if time_index*p['printT'] != 36:
        raise ValueError('Comparison time is not saved')
    with np.load(path/'results.npz') as data:
        levels = data['fragment_quantile_levels']
        iq = int(np.argmin(abs(levels-.1)))
        if abs(levels[iq]-.1)>1e-12:
            raise ValueError('Missing q10')
        counts = data['Holevo_Z_S_Ef_fractionsT_Nsamples'].astype(int)
        members = fragment_members(p, counts)
        retained = 'fragment_sample_members' in data
        if retained and not np.array_equal(members, data['fragment_sample_members']):
            raise ValueError('Fragment identity mismatch')
        numerical = data['Holevo_Z_S_Ef_fractionsT_quantiles'][iq, 1:n//2+1, time_index]/np.log(2)
        if not np.all(np.isfinite(numerical)):
            raise ValueError('Missing evaluated quantile')
        exact = np.empty_like(numerical, dtype=float)
        for k, seed in enumerate(p['run_seeds']):
            b2 = squared_overlaps(p, seed, np.array([36.]))[:, 0]
            for m in range(1, n//2+1):
                count = counts[m, time_index, k]
                sites = members[m, k, :count, :m]-1
                chi = entropy(np.sqrt(np.prod(b2[sites], axis=1)))/np.log(2)
                exact[m-1, k] = np.quantile(chi, .1)
    if not np.array_equal(redundancy(numerical), redundancy(exact), equal_nan=True):
        raise ValueError(f'Exact and state-vector thresholds differ: {path}')
    return {'run_id': path.name, 'source': str(path), 'params': p,
            'numerical': redundancy(numerical), 'exact': redundancy(exact),
            'max_q10_difference_bits': float(np.max(abs(exact-numerical))),
            'membership': 'verified_retained' if retained else 'seed_reconstruction',
            'params_sha256': file_hash(path/'params.json'),
            'results_sha256': file_hash(path/'results.npz')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--panel-a-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {}
    for batch in BATCHES:
        for f in sorted((args.data_root/batch/'runs').glob('*/params.json')):
            p = json.loads(f.read_text())
            theta, bias = p['Mironowicz_theta'], p['psi_bias']
            lam = 1-(np.cos(theta)*2*np.sqrt(bias*(1-bias))+np.sin(theta)*(2*bias-1))**2
            selected = next((x for x in LAMBDAS if abs(x-lam)<2e-8), None)
            if selected is None:
                continue
            groups.setdefault((p['nqubits_E'], selected), []).append(load_case(f.parent))
    expected = {(n,lam) for n in (8,10,12,14,16) for lam in LAMBDAS}
    expected |= {(n,lam) for n in (20,24) for lam in (.75,1.)}
    if set(groups) != expected:
        raise ValueError(f'Unexpected groups: {set(groups)^expected}')
    rng = np.random.default_rng(20260907)
    rows, evidence = [], []
    for (n,lam), cases in sorted(groups.items()):
        seeds = cases[0]['params']['run_seeds']
        if any(c['params']['run_seeds'] != seeds for c in cases):
            raise ValueError('Geometries are not paired')
        row = {'n_environment': n, 'lambda': lam, 'geometry_count': len(cases),
               'fragment_cap': cases[0]['params']['fragment_sample_count']}
        for kind in ('numerical', 'exact'):
            values = np.stack([c[kind] for c in cases])
            if not np.all(np.isfinite(values) == np.isfinite(values[0])):
                raise ValueError('Matched geometries disagree on which realizations cross')
            per_real = np.mean(values, axis=0)
            row[kind] = summarize(per_real, rng)
            row[kind+'_per_realization'] = [float(v) if np.isfinite(v) else None for v in per_real]
        rows.append(row)
        for c in cases:
            evidence.append({k:v for k,v in c.items() if k not in ('params','numerical','exact')})

    with args.panel_a_csv.open(newline='', encoding='utf-8') as handle:
        arows = list(csv.DictReader(handle))
    avals = {}
    for r in arows:
        avals[float(r['lambda'])] = r
    x = sorted(avals)
    def column(key):
        return [float(avals[v][key]) if avals[v][key] not in ('','None') else np.nan for v in x]
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'legend.fontsize':6.5,
                         'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig, (left,right) = plt.subplots(1,2,figsize=(7.15,3.15),constrained_layout=True)
    left.plot(x,column('m_delta_q10'),'o-',color='#c44e52',ms=3.5,label='10th-percentile threshold')
    left.plot(x,column('m_delta_mean'),'s--',color='#4c72b0',ms=3,label='Mean-information threshold')
    left.plot(x,column('naive_m_delta'),':',color='.3',label='Typical-overlap estimate')
    left.set(xlim=(-.03,1.03),ylim=(1,20),xlabel=r'Alignment $\Lambda$',ylabel=r'Threshold size $m_{0.1}$')
    left.legend(loc='upper right',frameon=False)
    colors = ('#22834b','#c44e52','#65529a')
    markers = ('s','D','^')
    for lam,color,marker,offset in zip(LAMBDAS,colors,markers,(-.18,0,.18)):
        series = sorted([r for r in rows if r['lambda']==lam],key=lambda r:r['n_environment'])
        ns = [r['n_environment']+offset for r in series]
        ys = [r['exact']['mean'] if r['exact']['mean'] is not None else np.nan for r in series]
        right.plot(ns,ys,'-',color=color,lw=.9,label=rf'$\Lambda={lam:.3g}$')
        for xx,r in zip(ns,series):
            e,s = r['exact'],r['numerical']
            if s['mean'] is None:
                continue
            right.errorbar(xx,e['mean'],yerr=[[max(0,e['mean']-e['low'])],[max(0,e['high']-e['mean'])]],
                           fmt='none',ecolor=color,capsize=2,lw=.8)
            right.plot(xx,s['mean'],marker=marker,ms=4.5,
                       mfc=color if s['defined']==s['total'] else 'white',mec=color,linestyle='none')
    right.set(xlabel=r'Environment size $N_E$',ylabel=r'Mean redundancy $\langle R_{0.1}^{(q_{0.1})}\rangle$',
              xlim=(6.8,25.2),ylim=(1.8,6.7),xticks=(8,12,16,20,24))
    right.legend(loc='upper left',frameon=False)
    right.text(.98,.04,'Lines: exact; symbols: state vector',ha='right',transform=right.transAxes,fontsize=6.4)
    for tag,ax in zip('ab',(left,right)):
        ax.grid(alpha=.2,lw=.5)
        ax.text(.03 if tag=='a' else .97,.95,f'({tag})',transform=ax.transAxes,
                ha='left' if tag=='a' else 'right',va='top',fontweight='bold')
    for suffix in ('pdf','png'):
        fig.savefig(args.output/f'fragment_threshold_scaling.{suffix}',dpi=300)
    plt.close(fig)
    manifest = {'time':36,'threshold':.9,'quantile':.1,'bootstrap_draws':2000,'seed':20260907,
                'order':'Within each realization: fragment q10, first threshold m, R=N/m; average paired geometries within realization, then average defined realizations.',
                'noncrossings':'Undefined for no crossing among m<=N/2; reported conditional means exclude these realizations and display counts.',
                'intervals':'Realization-only percentile bootstrap of exact R; paired geometries stay together; sampled fragments held fixed.',
                'panel_a_source':str(args.panel_a_csv),'panel_a_sha256':file_hash(args.panel_a_csv),
                'script_sha256':file_hash(Path(__file__)),'exact_helper_sha256':file_hash(Path(__file__).with_name('recalculate_exact.py')),
                'groups':rows,'sources':evidence,
                'figure_sha256':file_hash(args.output/'fragment_threshold_scaling.pdf')}
    (args.output/'unified_scaling_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    flat=[]
    for r in rows:
        flat.append({**{k:r[k] for k in ('n_environment','lambda','geometry_count','fragment_cap')},
                     **{f'{kind}_{k}':v for kind in ('numerical','exact') for k,v in r[kind].items()}})
    with (args.output/'unified_scaling.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for r in flat:
        print(r,flush=True)


if __name__ == '__main__':
    main()
