import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recalculate_exact import entropy, file_hash, fragment_members, squared_overlaps


def evaluate(p, times, members):
    n, seeds = p["nqubits_E"], p["run_seeds"]
    mean = np.zeros((n//2, len(times), len(seeds)))
    q10, dz = np.zeros_like(mean), np.zeros_like(mean)
    for k, seed in enumerate(seeds):
        b2 = squared_overlaps(p, seed, times)
        hs = entropy(np.sqrt(np.prod(b2, axis=0))) / np.log(2)
        for m in range(1, n//2+1):
            count = min(p["fragment_sample_count"], math.comb(n, m))
            sites = members[m, k, :count, :m] - 1
            rest = np.array([np.setdiff1d(np.arange(n), s) for s in sites])
            chi = entropy(np.sqrt(np.prod(b2[sites], axis=1))) / np.log(2)
            d = hs - entropy(np.sqrt(np.prod(b2[rest], axis=1))) / np.log(2)
            mean[m-1, :, k] = chi.mean(axis=0)
            q10[m-1, :, k] = np.quantile(chi, .1, axis=0)
            dz[m-1, :, k] = d.mean(axis=0)
    return mean, q10, dz


def main():
    parser = argparse.ArgumentParser(description="Exact CPU exploration of independent pure witnesses; no state-vector evolution.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    references = []
    for batch in ("Paper_A_final_FieldProfileControl", "Paper_A_final_LowFieldRefinement", "Paper_A_final_FieldMeanWidth"):
        for path in sorted((args.data_root/batch/"runs").glob("*/params.json")):
            p = json.loads(path.read_text())
            if p["psi_bias"] == .5 and p["Mironowicz_theta"] == 0 and (path.parent/"results.npz").exists():
                references.append((path, p))
    if not references:
        raise ValueError("A completed aligned field reference is required for paired seeds and sampling")
    source, base = references[0]
    assert base["nqubits_E"] == 16 and len(base["run_seeds"]) == 24
    assert base["H_SE_J"] == .1 and base["H_EE_J"] == 0 and base["H_E_J"] == 0
    times = np.arange(61, dtype=float)
    late = times >= 30
    counts = np.zeros((17, 61, 24), dtype=int)
    for m in range(1, 9):
        counts[m] = min(64, math.comb(16, m))
    base = {**base, "fragment_sample_count": 64}
    members = fragment_members(base, counts)
    ratios = (0, .05, .10, .15, .20, .25, .5, 1, 2, 5, 10, 20, 30)
    etas = (0, .25, .5, .75, 1)
    rows, all_min, all_statistics, cache = [], [], [], {}
    for ratio in ratios:
        for eta in ((0,) if ratio == 0 else etas):
            h0, width = .1*ratio*math.sqrt(1-eta**2), .1*ratio*eta
            p = {**base, "Mironowicz_h0": h0, "Mironowicz_alpha2": width}
            mean, q10, dz = evaluate(p, times, members)
            cache[(round(h0, 12), round(width, 12))] = (mean, q10, dz)
            minima = q10[-1, late].min(axis=0)
            passing = q10[:, late] >= .9
            threshold = np.where(passing.any(axis=0), passing.argmax(axis=0)+1, np.nan)
            rmed = np.array([np.median(16/t[np.isfinite(t)]) if np.isfinite(t).any() else np.nan for t in threshold.T])
            medians = rmed[np.isfinite(rmed)]
            all_statistics.append(np.stack((np.median(mean[-1, late], axis=0),
                np.median(dz[-1, late], axis=0), rmed, minima)))
            rows.append(dict(ratio=ratio, eta=eta, h0=h0, width=width,
                mean_holevo=np.median(mean[-1, late], axis=0).mean(),
                mean_remainder=np.median(dz[-1, late], axis=0).mean(),
                mean_minimum=minima.mean(), persistent=int((minima >= .9).sum()),
                conditional_redundancy=np.mean(medians) if len(medians) else float("nan"),
                defined_realizations=len(medians)))
            all_min.append(minima)
    validations = []
    for path, p in references:
        key = (round(p["Mironowicz_h0"], 12), round(p["Mironowicz_alpha2"], 12))
        if key not in cache or p["run_seeds"] != base["run_seeds"]:
            continue
        with np.load(path.parent/"results.npz", allow_pickle=False) as old:
            if "fragment_sample_members" in old:
                np.testing.assert_array_equal(old["fragment_sample_members"][:9], members[:9])
            qindex = np.flatnonzero(np.isclose(old["fragment_quantile_levels"], .1))[0]
            exact = cache[key][1]
            numerical = old["Holevo_Z_S_Ef_fractionsT_quantiles"][qindex, 1:9] / np.log(2)
            assert numerical.shape == exact.shape
            validations.append(dict(source=str(path.parent), params_sha256=file_hash(path),
                results_sha256=file_hash(path.parent/"results.npz"),
                max_q10_error_bits=float(np.max(abs(exact-numerical))),
                changed_persistence=int(np.sum((exact[-1, late].min(axis=0)>=.9)!=(numerical[-1, late].min(axis=0)>=.9)))))
    with (args.output/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output/"realization_minima.npz", minima=np.array(all_min), statistics=np.array(all_statistics),
                        ratios=[r["ratio"] for r in rows], etas=[r["eta"] for r in rows], seeds=base["run_seeds"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for eta in etas:
        selected = [r for r in rows if r["eta"] == eta and r["ratio"] > 0]
        for ax, metric, label in zip(axes.flat, ("mean_holevo", "mean_remainder", "mean_minimum", "persistent"),
                                     ("Mean Holevo information (bits)", "Mean pointer-basis remainder (bits)", "Mean minimum q10 Holevo (bits)", "Persistent realizations / 24")):
            ax.plot([r["ratio"] for r in selected], [r[metric] for r in selected], ".-", label=f"eta = {eta:g}")
            ax.set(xscale="log", xlabel="Field RMS / g", ylabel=label)
            ax.grid(alpha=.2)
    axes[0,0].legend(ncol=2)
    axes[1,0].axhline(.9, color="gray", ls=":")
    fig.suptitle("Exact CPU exploration: N = 16, 24 paired realizations, t = 30–60")
    fig.savefig(args.output/"comparison.png", dpi=180)
    plt.close(fig)
    manifest = dict(cases=len(rows), seconds=time.perf_counter()-started, baseline=str(source),
        script_sha256=file_hash(Path(__file__)), overlap_script_sha256=file_hash(Path(__file__).with_name("recalculate_exact.py")),
        base_params=base, times=times.tolist(), validations=validations,
        scope="Exact independent-qubit evolution; finite paired realization/fragment sampling, no confidence intervals; exploratory.")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(cases=len(rows), seconds=manifest["seconds"], validated=len(validations),
        max_error=max((v["max_q10_error_bits"] for v in validations), default=None),
        changed_persistence=sum(v["changed_persistence"] for v in validations))))


if __name__ == "__main__":
    main()
