from pathlib import Path
import json
import csv
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy

import argparse

parser = argparse.ArgumentParser(description='Reproduce the scoped September 2026 manuscript audit.')
parser.add_argument('--notes-root', type=Path, default=Path(__file__).resolve().parents[3] / 'QD-summary-Imperfect-CNOT')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.notes_root
if not root.is_dir():
    raise FileNotFoundError(root)
args.output.parent.mkdir(parents=True, exist_ok=True)

def entropy(p):
    return -(xlogy(p,p)+xlogy(1-p,1-p))/np.log(2)

fields = []
for path in (root/'data/Paper_A_final_FieldProfileControl/runs').glob('*/params.json'):
    p = json.loads(path.read_text())
    if p['field_geometry'] != 'aligned':
        continue
    with np.load(path.with_name('results.npz')) as d:
        levels = d['fragment_quantile_levels']
        q = d['Holevo_Z_S_Ef_fractionsT_quantiles'][np.argmin(abs(levels-.1)),8,30:61,:] / np.log(2)
        minimum = np.min(q,axis=0)
        fields.append({'profile':p['field_profile'],'ratio':p['field_strength_ratio'],
                       'mean_min':float(np.mean(minimum)),
                       'persistent_count':int(np.count_nonzero(minimum>=.9)),
                       'realizations':len(minimum),'worst_min':float(np.min(minimum))})
fields.sort(key=lambda r:(r['profile'],r['ratio']))
maps=[]
for path in (root/'data/Paper_A_final_StabilityMap/runs').glob('*/params.json'):
    p=json.loads(path.read_text())
    if abs(p['Mironowicz_theta']-np.pi/4)>1e-8:
        continue
    with np.load(path.with_name('results.npz')) as d:
        levels=d['fragment_quantile_levels']
        q=d['Holevo_Z_S_Ef_fractionsT_quantiles'][np.argmin(abs(levels-.1)),8,20:41,:] / np.log(2)
        minimum=np.min(q,axis=0)
        maps.append({'p':p['psi_bias'],'W':p['Mironowicz_alpha2'],
                     'mean_min':float(np.mean(minimum)),
                     'persistent_count':int(np.count_nonzero(minimum>=.9)),
                     'realizations':len(minimum),'worst_min':float(min(minimum))})
if len(fields) != 17 or len(maps) != 121:
    raise ValueError(f'Expected 17 field cases and 121 map points, found {len(fields)} and {len(maps)}')
mapsummary={'points':len(maps),'mean_pass':sum(r['mean_min']>=.9 for r in maps),
            'all_realizations_pass':sum(r['persistent_count']==r['realizations'] for r in maps),
            'mean_pass_but_not_all': [r for r in maps if r['mean_min']>=.9 and r['persistent_count']<r['realizations']]}
b=brentq(lambda b: entropy((1+b)/2)-.9,0,1)
perr=(1-np.sqrt(1-b*b))/2
mc=[]
rng=np.random.default_rng(9052026)
for lam in [.25,.5,.75,.933012702,1.]:
    g=rng.normal(0,.1,(200000,24))
    overlaps=np.cumprod(np.sqrt(1-lam*np.sin(np.pi*g*36/2)**2),axis=1)
    chi=entropy((1+overlaps)/2)
    mean=chi.mean(axis=0); q=np.quantile(chi,.1,axis=0)
    mc.append({'lambda':lam,'mean_threshold':int(np.flatnonzero(mean>=.9)[0]+1),
               'q10_threshold':int(np.flatnonzero(q>=.9)[0]+1)})
res={'holevo_90pct':{'b':b,'helstrom_error':perr,'accessible_information':float(1-entropy(perr))},
     'field_persistence':fields,'stability_map_summary':mapsummary,'independent_product_MC_200000':mc}
args.output.write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
