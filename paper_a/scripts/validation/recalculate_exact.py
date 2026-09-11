from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import shutil
import time
import warnings

import numpy as np
from scipy.special import xlogy

BATCHES = {
    'Paper_A_final_SubmissionMatchedLambda': 38,
    'Paper_A_final_FieldProfileControl': 26,
    'Paper_A_final_StabilityMap': 363,
    'Paper_A_final_LambdaCollapse': 75,
    'Paper_A_final_HolevoPlateauFullFragments': 1,
    'Paper_A_final_HigherNValidation': 5,
    'Paper_A_final_N24FixedTimeValidation': 1,
    'Paper_A_final_TimestepConvergence': 3,
}
PREFIXES = ('I_S_Ef', 'Holevo_Z_S_Ef', 'Discord_Z_S_Ef')


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def seeded(seed, label):
    value = int.from_bytes(hashlib.blake2b(f'{seed}:{label}'.encode(), digest_size=16).digest(), 'big')
    return random.Random(value)


def fragment_members(p, counts):
    n, seeds, cap = p['nqubits_E'], p['run_seeds'], p['fragment_sample_count']
    if not p['fragment_reuse_samples']:
        raise ValueError('Cannot replay non-reused fragment sampling from the named seed stream')
    members = np.full((n+1, len(seeds), cap, n), -1, dtype=np.int16)
    for m in range(1, n+1):
        for k, seed in enumerate(seeds):
            expected = int(counts[m, 0, k])
            if not np.all(counts[m, :, k] == expected):
                raise ValueError('Fragment sample count changes over time')
            if not expected:
                continue
            if expected != min(cap, math.comb(n, m)):
                raise ValueError(f'Unexpected count for m={m}: {expected}')
            rng, seen, subsets = seeded(seed, f'fragments:{m}'), set(), []
            while len(subsets) < expected:
                subset = tuple(sorted(rng.sample(range(1, n+1), m)))
                if subset not in seen:
                    seen.add(subset)
                    subsets.append(subset)
            members[m, k, :expected, :m] = subsets
    return members


def squared_overlaps(p, seed, times):
    n = p['nqubits_E']
    rg, rh = seeded(seed, 'mironowicz.J_SE'), seeded(seed, 'mironowicz.J_E')
    g = np.array([rg.gauss(0, p['H_SE_J']) for _ in range(n)])[:, None]
    h = np.array([p.get('Mironowicz_h0', 0)+rh.gauss(0, p['Mironowicz_alpha2']) for _ in range(n)])[:, None]
    t = times[None, :]
    theta, bias = p['Mironowicz_theta'], p['psi_bias']
    ax, az = -np.pi*g*np.cos(theta)/2, h-np.pi*g*np.sin(theta)/2
    omega = np.sqrt(ax*ax+az*az)
    st = np.sinc(omega*t/np.pi)*t
    qx = np.cos(h*t)*st*ax
    qz = np.cos(h*t)*st*az-np.sin(h*t)*np.cos(omega*t)
    q0 = np.cos(h*t)*np.cos(omega*t)+np.sin(h*t)*st*az
    return np.clip(q0*q0+(qx*2*np.sqrt(bias*(1-bias))+qz*(2*bias-1))**2, 0, 1)


def entropy(overlap):
    p = (1+np.clip(overlap, 0, 1))/2
    return -(xlogy(p, p)+xlogy(1-p, 1-p))


def _validate(p):
    required = {'H_SE_Special': 'Mironowicz_rand', 'nqubits_S': 1, 'psi_S_spec': 'x+', 'psi_E_spec': 'bias'}
    for key, value in required.items():
        if p.get(key) != value:
            raise ValueError(f'Unsupported {key}={p.get(key)}')
    for key in ('H_EE_J', 'H_E_J', 'Mironowicz_epsilon', 'Mironowicz_epsilon2'):
        if p.get(key, 0) != 0:
            raise ValueError(f'Unsupported nonzero {key}')
    if len(p['run_seeds']) != int(p['AverageOverRunsN']):
        raise ValueError('Realization seed count mismatch')


def recalculate(source, target, signature):
    p = json.loads((source/'params.json').read_text())
    _validate(p)
    n, kcount, cap = p['nqubits_E'], len(p['run_seeds']), p['fragment_sample_count']
    with np.load(source/'results.npz', allow_pickle=False) as old:
        counts = old['Holevo_Z_S_Ef_fractionsT_Nsamples'].astype(int)
        tcount = counts.shape[1]
        if counts.shape != (n+1, tcount, kcount):
            raise ValueError('Array dimension mismatch')
        times = np.arange(tcount)*p['printT']
        members = fragment_members(p, counts)
        retained = 'fragment_sample_members' in old
        if retained and not np.array_equal(members, old['fragment_sample_members']):
            raise ValueError('Replayed fragment identities disagree with retained identities')
        levels = old['fragment_quantile_levels']
        out = {'fragment_sample_members': members, 'fragment_quantile_levels': levels,
               'rhoS_T': old['rhoS_T'], 'norms': old['norms']}
        samples = {key: np.full((n+1, tcount, kcount, cap), np.nan) for key in PREFIXES}
        fid = np.empty((n, tcount, kcount))
        for k, seed in enumerate(p['run_seeds']):
            b2 = squared_overlaps(p, seed, times)
            fid[:, :, k] = b2
            system_entropy = entropy(np.sqrt(np.prod(b2, axis=0)))
            for m in range(1, n+1):
                count = counts[m, 0, k]
                if not count:
                    continue
                sites = members[m, k, :count, :m]-1
                bf = np.sqrt(np.prod(b2[sites], axis=1))
                complements = np.array([np.setdiff1d(np.arange(n), s) for s in sites])
                br = np.sqrt(np.prod(b2[complements], axis=1))
                chi = entropy(bf)
                discord = system_entropy[None, :]-entropy(br)
                if np.min(discord) < -1e-12:
                    raise ValueError('Negative exact discord beyond rounding')
                for key, value in zip(PREFIXES, (chi+discord, chi, discord)):
                    samples[key][m, :, k, :count] = value.T
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for key, values in samples.items():
                base = key+'_fractionsT'
                mean = np.nanmean(values, axis=-1)
                std = np.nanstd(values, axis=-1, ddof=1)
                std[counts == 1] = 0
                mean[0], std[0] = 0, 0
                out[base] = mean
                out[base+'_STD'] = std
                out[base+'_samples'] = values
                out[base+'_Nsamples'] = counts
                out[base+'_quantiles'] = np.nanquantile(values, levels, axis=-1)
                out[base+'_quantiles'][:, 0] = 0
                out[base+'_quantiles_runAv'] = np.nanmean(out[base+'_quantiles'], axis=-1)
                out[base+'_runAv'] = np.nanmean(mean, axis=-1)
                out[base+'_runSTD'] = np.nanstd(mean, axis=-1, ddof=1)
                out[base+'_runSEM'] = out[base+'_runSTD']/np.sqrt(kcount)
        out['SBS_fid_runs_1'], out['SBS_fid_1'] = fid, np.mean(fid, axis=-1)
        iq = int(np.argmin(abs(levels-.1)))
        late = times >= p['T']/2
        qold = old['Holevo_Z_S_Ef_fractionsT_quantiles'][iq]/np.log(2)
        qnew = out['Holevo_Z_S_Ef_fractionsT_quantiles'][iq]/np.log(2)
        old_min, new_min = np.min(qold[n//2, late], axis=0), np.min(qnew[n//2, late], axis=0)
        report = {'batch': source.parent.parent.name, 'run_id': source.name, 'signature': signature,
                  'source': str(source), 'n_environment': n, 'realizations': kcount,
                  'membership': 'replay_verified_against_retained' if retained else 'deterministic_replay',
                  'mean_min_old': float(np.mean(old_min)), 'mean_min_exact': float(np.mean(new_min)),
                  'persistent_old': int(np.sum(old_min >= .9)), 'persistent_exact': int(np.sum(new_min >= .9)),
                  'classification_changes': int(np.sum((old_min >= .9) != (new_min >= .9))),
                  'max_q10_difference_bits': float(np.nanmax(abs(qold[:n//2+1]-qnew[:n//2+1]))),
                  'max_local_squared_overlap_difference': float(np.max(abs(fid-old['SBS_fid_runs_1'])))}
        for key in PREFIXES:
            base = key+'_fractionsT'
            report[key+'_max_mean_difference_bits'] = float(np.nanmax(abs(out[base][:n//2+1]-old[base][:n//2+1]))/np.log(2))
        if 36 in times:
            ti = int(np.flatnonzero(times == 36)[0])
            for tag, q in [('old', qold), ('exact', qnew)]:
                passing = np.flatnonzero(np.mean(q[1:n//2+1, ti], axis=-1) >= .9)
                report['threshold_t36_'+tag] = int(passing[0]+1) if len(passing) else None
        target.mkdir(parents=True, exist_ok=True)
        for name in ('params.json', 'metadata.json'):
            if (source/name).exists():
                shutil.copy2(source/name, target/name)
        np.savez_compressed(target/'results.npz', **out)
        report['derived_results_sha256'] = file_hash(target/'results.npz')
        (target/'recalculation.json').write_text(json.dumps(report, indent=2)+'\n')
        return report


def main():
    parser = argparse.ArgumentParser(description='Exact uncensored Paper A reanalysis; never modifies production output.')
    parser.add_argument('--source-roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--batches', nargs='+', default=list(BATCHES), choices=list(BATCHES))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if any(args.output_root.resolve().is_relative_to(r.resolve()) for r in args.source_roots):
        raise ValueError('Output must be outside production data roots')
    reports, missing = [], []
    script_hash = hashlib.sha256(Path(__file__).read_text().encode()).hexdigest()
    start = time.perf_counter()
    for batch in args.batches:
        sources = {}
        for root in reversed(args.source_roots):
            sources.update({p.parent.name: p.parent for p in (root/batch/'runs').glob('*/params.json')})
        if len(sources) != BATCHES[batch]:
            missing.append({'batch': batch, 'expected': BATCHES[batch], 'found': len(sources)})
        target_batch = args.output_root/batch
        target_batch.mkdir(exist_ok=True)
        for root in args.source_roots:
            if (root/batch/'batch_manifest.json').exists():
                shutil.copy2(root/batch/'batch_manifest.json', target_batch/'batch_manifest.json')
                break
        for source in sorted(sources.values()):
            if not (source/'results.npz').exists():
                missing.append({'batch': batch, 'run_id': source.name, 'missing': 'results.npz'})
                continue
            signature = {'script_sha256_lf': script_hash, 'params_sha256': file_hash(source/'params.json'),
                         'results_sha256': file_hash(source/'results.npz')}
            target = target_batch/'runs'/source.name
            previous = target/'recalculation.json'
            if previous.exists() and json.loads(previous.read_text()).get('signature') == signature:
                reports.append(json.loads(previous.read_text()))
                continue
            reports.append(recalculate(source, target, signature))
            if len(reports) % 20 == 0:
                print(f'{len(reports)} cases, {time.perf_counter()-start:.1f}s', flush=True)
    summary = {'seconds': time.perf_counter()-start, 'cases': len(reports), 'missing': missing,
               'membership_verified_cases': sum(r['membership']=='replay_verified_against_retained' for r in reports),
               'classification_changes': sum(r['classification_changes'] for r in reports), 'runs': reports}
    (args.output_root.parent/'exact_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='runs'}), flush=True)


if __name__ == '__main__':
    main()
