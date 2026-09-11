from pathlib import Path
import hashlib, random, json
import numpy as np
from scipy.special import xlogy

import argparse

parser = argparse.ArgumentParser(description='Reproduce the scoped September 2026 manuscript audit.')
parser.add_argument('--notes-root', type=Path, default=Path(__file__).resolve().parents[3] / 'QD-summary-Imperfect-CNOT')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.notes_root / "data"
if not root.is_dir():
    raise FileNotFoundError(root)
args.output.parent.mkdir(parents=True, exist_ok=True)
rows=[]
for path in (root/'Paper_A_final_FieldProfileControl/runs').glob('*/params.json'):
    p=json.loads(path.read_text())
    if p['field_geometry']!='aligned' or p['field_strength_ratio'] not in [.25,.5,30.]:
        continue
    with np.load(path.with_name('results.npz')) as d:
        fid=d['SBS_fid_runs_1']
        samples=d['Holevo_Z_S_Ef_fractionsT_samples']
        members=d['fragment_sample_members']
        max_fid=max_chi=max_d=max_late_chi=0.
        exact_min=[]; stored_min=[]; censored_d=0; identity_error=0.
        d_samples=d['Discord_Z_S_Ef_fractionsT_samples']
        i_samples=d['I_S_Ef_fractionsT_samples']
        for k,seed in enumerate(p['run_seeds']):
            rngs=[random.Random(int.from_bytes(hashlib.blake2b(f'{seed}:{label}'.encode(),digest_size=16).digest(),'big')) for label in ['mironowicz.J_SE','mironowicz.J_E']]
            g=np.array([rngs[0].gauss(0,p['H_SE_J']) for _ in range(16)])[:,None]
            h=np.array([p['Mironowicz_h0']+rngs[1].gauss(0,p['Mironowicz_alpha2']) for _ in range(16)])[:,None]
            t=np.arange(61)[None,:]*p['printT']
            theta=p['Mironowicz_theta']; bias=p['psi_bias']
            ax=-np.pi*g*np.cos(theta)/2; az=h-np.pi*g*np.sin(theta)/2
            omega=np.sqrt(ax*ax+az*az)
            sinc=np.sinc(omega*t/np.pi)*t
            qx=np.cos(h*t)*sinc*ax
            qy=-np.sin(h*t)*sinc*ax
            qz=np.cos(h*t)*sinc*az-np.sin(h*t)*np.cos(omega*t)
            q0=np.cos(h*t)*np.cos(omega*t)+np.sin(h*t)*sinc*az
            b2=np.clip(q0*q0+(qx*2*np.sqrt(bias*(1-bias))+qz*(2*bias-1))**2,0,1)
            max_fid=max(max_fid,float(np.max(abs(b2-fid[:,:,k]))))
            for m in [1,4,8]:
                sites=members[m,k,:,:m]
                valid=np.all(sites>=0,axis=1)
                bf=np.sqrt(np.prod(b2[sites[valid]-1],axis=1))
                prob=(1+bf)/2
                chi=-(xlogy(prob,prob)+xlogy(1-prob,1-prob))
                stored=samples[m,:,k,:][:,valid].T
                max_chi=max(max_chi,float(np.nanmax(abs(chi-stored)/np.log(2))))
                max_late_chi=max(max_late_chi,float(np.nanmax(abs(chi[:,30:]-stored[:,30:])/np.log(2))))
                be=np.sqrt(np.prod(b2,axis=0))
                br=np.array([np.sqrt(np.prod(b2[np.setdiff1d(np.arange(16),s-1)],axis=0)) for s in sites[valid]])
                def hb(b):
                    p=(1+b)/2
                    return -(xlogy(p,p)+xlogy(1-p,1-p))
                exact_d=hb(be)[None,:]-hb(br)
                stored_d=d_samples[m,:,k,:][:,valid].T
                stored_i=i_samples[m,:,k,:][:,valid].T
                max_d=max(max_d,float(np.nanmax(abs(exact_d-stored_d)/np.log(2))))
                censored_d+=int(np.count_nonzero((stored_d==0)&(exact_d>1e-5)))
                identity_error=max(identity_error,float(np.nanmax(abs(stored_d-(stored_i-stored))/np.log(2))))
                if m==8:
                    exact_min.append(float(np.min(np.quantile(chi[:,30:]/np.log(2),.1,axis=0))))
                    stored_min.append(float(np.min(np.quantile(stored[:,30:]/np.log(2),.1,axis=0))))
        rows.append({'profile':p['field_profile'],'ratio':p['field_strength_ratio'],
                     'max_local_squared_fidelity_error':max_fid,
                     'max_fragment_holevo_error_bits':max_chi,
                     'max_late_fragment_holevo_error_bits':max_late_chi,
                     'max_fragment_discord_error_bits':max_d,
                     'stored_discord_identity_max_error_bits':identity_error,
                     'stored_zero_discord_with_exact_above_1e_minus5_nats':censored_d,
                     'mean_exact_min':float(np.mean(exact_min)),
                     'mean_stored_min':float(np.mean(stored_min)),
                     'exact_persistent_count':int(np.count_nonzero(np.array(exact_min)>=.9)),
                     'stored_persistent_count':int(np.count_nonzero(np.array(stored_min)>=.9)),
                     'classification_changes':int(np.count_nonzero((np.array(exact_min)>=.9) != (np.array(stored_min)>=.9))),
                     'realizations':24,'sizes':[1,4,8],'time_samples':61})
if len(rows) != 6:
    raise ValueError(f'Expected six field cases, found {len(rows)}')
args.output.write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
