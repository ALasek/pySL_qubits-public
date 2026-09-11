from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recalculate_exact import entropy, file_hash, fragment_members, squared_overlaps, _validate
from plot_threshold_sizes import threshold_size, summarize

CRITERIA = ('mean', 'q25', 'q10')


def statistics(values):
    return np.array([np.mean(values), np.quantile(values, .25), np.quantile(values, .1)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(args.reference.read_text())
    grouped, evidence = {}, []
    for source in reference['sources']:
        old = Path(source['source'])
        path = args.data_root / old.parent.parent.name / 'runs' / old.name
        p = json.loads((path/'params.json').read_text())
        _validate(p)
        assert file_hash(path/'params.json') == source['params_sha256']
        assert file_hash(path/'results.npz') == source['results_sha256']
        n, seeds = p['nqubits_E'], p['run_seeds']
        ti = round(36/p['printT'])
        assert ti*p['printT'] == 36 and p['Mironowicz_h0'] == p['Mironowicz_alpha2'] == 0
        theta, bias = p['Mironowicz_theta'], p['psi_bias']
        lam = 1-(np.cos(theta)*2*np.sqrt(bias*(1-bias))+np.sin(theta)*(2*bias-1))**2
        lam = min((.75,.933012702,1.),key=lambda x:abs(x-lam))
        with np.load(path/'results.npz') as d:
            counts = d['Holevo_Z_S_Ef_fractionsT_Nsamples'].astype(int)
            members = fragment_members(p,counts)
            retained = 'fragment_sample_members' in d
            if retained:
                np.testing.assert_array_equal(members,d['fragment_sample_members'])
            curves = np.empty((3,n//2,len(seeds)))
            raw_curves = np.full_like(curves,np.nan)
            raw_key = 'Holevo_Z_S_Ef_fractionsT_samples'
            has_raw = raw_key in d
            for k,seed in enumerate(seeds):
                b2 = squared_overlaps(p,seed,np.array([36.]))[:,0]
                for m in range(1,n//2+1):
                    count = counts[m,ti,k]
                    sites = members[m,k,:count,:m]-1
                    chi = entropy(np.sqrt(np.prod(b2[sites],axis=1)))/np.log(2)
                    curves[:,m-1,k] = statistics(chi)
                    if has_raw:
                        raw = d[raw_key][m,ti,k,:count]/np.log(2)
                        assert np.all(np.isfinite(raw))
                        raw_curves[:,m-1,k] = statistics(raw)
            thresholds = np.stack([threshold_size(c) for c in curves])
            iq = np.argmin(abs(d['fragment_quantile_levels']-.1))
            stored = d['Holevo_Z_S_Ef_fractionsT_quantiles'][iq,1:n//2+1,ti]/np.log(2)
            np.testing.assert_allclose(thresholds[2],threshold_size(stored),equal_nan=True)
            assert np.all(curves[1]>=curves[2]-1e-14)
            case = {'seeds':seeds,'thresholds':thresholds}
            grouped.setdefault((n,lam),[]).append(case)
            raw_disagreements = None
            if has_raw:
                rt = np.stack([threshold_size(c) for c in raw_curves])
                same = (rt==thresholds)|(np.isnan(rt)&np.isnan(thresholds))
                raw_disagreements = dict(zip(CRITERIA,(~same).sum(axis=1).tolist()))
            evidence.append({**source,'retained_members':retained,'raw_samples_available':has_raw,
                             'raw_threshold_disagreements':raw_disagreements})
    rng = np.random.default_rng(20260908)
    rows, individual = [], []
    for (n,lam),cases in sorted(grouped.items()):
        assert all(c['seeds']==cases[0]['seeds'] for c in cases)
        stack = np.stack([c['thresholds'] for c in cases])
        assert np.all(np.isfinite(stack)==np.isfinite(stack[0]))
        sizes = np.mean(stack,axis=0)
        common = np.all(np.isfinite(sizes),axis=0)
        for i,criterion in enumerate(CRITERIA):
            stats = summarize(sizes[i],rng)
            rows.append({'n_environment':n,'lambda':lam,'criterion':criterion,**stats,
                         'common_defined':int(common.sum()),
                         'common_mean':float(np.mean(sizes[i,common])) if common.any() else None})
            individual.append({'n_environment':n,'lambda':lam,'criterion':criterion,
                               'sizes':[float(x) if np.isfinite(x) else None for x in sizes[i]],
                               'run_seeds':cases[0]['seeds']})
    with (args.output/'threshold_comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,constrained_layout=True)
    for ax,lam in zip(axes,(.75,.933012702,1.)):
        for criterion,color,marker,offset in zip(CRITERIA,('#2878b5','#d58a20','#af3951'),('s','D','o'),(-.2,0,.2)):
            rs=[r for r in rows if r['lambda']==lam and r['criterion']==criterion]
            ax.plot([],[],color=color,marker=marker,ls='none',label={'mean':'Mean information','q25':'25th percentile','q10':'10th percentile'}[criterion])
            for r in rs:
                y=r['mean'];x=r['n_environment']+offset
                ax.errorbar(x,y,yerr=[[y-r['low']],[r['high']-y]],fmt=marker,color=color,
                            mfc=color if r['defined']==r['total'] else 'white',ms=5,capsize=2)
        ax.set(title=rf'$\Lambda={lam:.3g}$',xlabel='Environment qubits',xticks=[8,12,16,20,24],ylim=(1.8,7.3))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean threshold fragment size')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('90% information threshold at t=36; open symbols exclude noncrossing realizations',fontsize=11)
    fig.savefig(args.output/'threshold_comparison.png',dpi=180)
    manifest={'reference_sha256':file_hash(args.reference),'script_sha256':file_hash(Path(__file__)),
              'method':'Exact overlaps on the original saved or seed-reconstructed fragment selections; per-realization thresholds, paired-preparation averaging, then mean of defined sizes.',
              'time':36,'information_threshold':.9,'bootstrap_draws':2000,'bootstrap_seed':20260908,
              'common_cohort':'common_mean restricts all three criteria to realizations crossing under every criterion.',
              'sources':evidence,'individual':individual}
    (args.output/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Source cases:',len(evidence),'raw-sample cases:',sum(e['raw_samples_available'] for e in evidence))
    print('Raw threshold disagreements:',{c:sum((e['raw_threshold_disagreements'] or {}).get(c,0) for e in evidence) for c in CRITERIA})
    for r in rows:
        if r['n_environment'] in (16,24): print(r)


if __name__ == '__main__':
    main()
