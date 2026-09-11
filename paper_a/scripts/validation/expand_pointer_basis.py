import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_plot_style import apply_style, save_figure

apply_style()
import numpy as np
from scipy.special import xlogy

from check_pointer_exact import (
    DEFAULT_RUN_DIR, _blocks_from_gammas, _brute_force_check, _chi_grid,
    _local_gammas, _sha256, _validate_params,
)
from recalculate_exact import fragment_members


def binary_entropy(p):
    p = np.clip(p, 0, 1)
    return -(xlogy(p, p) + xlogy(1-p, 1-p)) / np.log(2)


def chi_grid(gammas, sites, beta, phi):
    indices = np.asarray(sites)-1
    f = np.prod(gammas[indices])
    r = np.prod(np.delete(gammas, indices))
    cs = np.sin(beta[:, None])/2
    probability = .5 + cs*np.real(np.exp(1j*phi[None, :])*np.conj(f*r))
    determinant = cs**2 * max(0, 1-abs(f)**2) * max(0, 1-abs(r)**2) / 4
    entropy = binary_entropy((1+abs(f))/2)
    # The two conditional matrices have the same determinant, but different traces.
    for weight in (probability, 1-probability):
        ratio = np.divide(4*determinant, weight**2, out=np.zeros_like(weight), where=weight>1e-15)
        entropy = entropy - weight*binary_entropy((1+np.sqrt(np.clip(1-ratio, 0, 1)))/2)
    return entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = args.run_dir/"params.json"
    p = json.loads(source.read_text())
    _validate_params(p)
    original_seeds = p["run_seeds"]
    seeds = [int(s.generate_state(1, dtype=np.uint32)[0])
             for s in np.random.SeedSequence(p["seed_base"]).spawn(24)]
    assert seeds[:len(original_seeds)] == original_seeds
    p.update(run_seeds=seeds, AverageOverRunsN=24, fragment_sample_count=64, fragment_reuse_samples=True)
    counts = np.zeros((18, 1, 24), dtype=int)
    for m in range(1, 9):
        counts[m] = min(64, math.comb(17, m))
    members = fragment_members(p, counts)
    beta = np.unique(np.r_[np.linspace(0, np.pi/2, 181), np.geomspace(1e-4, np.pi/360, 24, endpoint=False)])
    phi = np.linspace(0, 2*np.pi, 80, endpoint=False)
    grids = np.zeros((8, 24, len(beta), len(phi)))
    rows = []
    direct_error = _brute_force_check()
    accelerated_error = 0.
    for k, seed in enumerate(seeds):
        gammas = _local_gammas(p, seed, 60.)
        for m in range(1, 9):
            size = counts[m, 0, k]
            for fragment_id, sites in enumerate(members[m, k, :size, :m]):
                values = chi_grid(gammas, sites, beta, phi)
                if k < 2 and fragment_id == 0:
                    test_beta = np.array([0., 1e-4, .01, .5, np.pi/2, np.pi])
                    test_phi = phi[::10]
                    ref = _chi_grid(*_blocks_from_gammas(gammas, sites.tolist()), test_beta, test_phi)
                    fast = chi_grid(gammas, sites, test_beta, test_phi)
                    accelerated_error = max(accelerated_error, float(np.max(abs(ref-fast))))
                best = np.unravel_index(np.argmax(values), values.shape)
                gain = float(values[best]-values[0, 0])
                rows.append(dict(size=m, realization=k, fragment=fragment_id,
                                 gain_bits=gain, best_beta=float(beta[best[0]]),
                                 z_bits=float(values[0, 0]), sites=" ".join(map(str, sites))))
                grids[m-1, k] += values/size
        print(f"Completed realization {k+1}/24", flush=True)
    assert accelerated_error < 1e-11, accelerated_error
    with (args.output/"individual_optima.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output/"basis_grids.npz", grids=grids, beta=beta, phi=phi,
                        seeds=seeds, fragment_members=members)
    rng = np.random.default_rng(20260909)
    weights = rng.multinomial(24, np.full(24, 1/24), size=2000)/24
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    curve_rows = []
    for m, style in zip((1, 2, 4, 6, 8), ("-", "--", "-.", ":", "-")):
        mean = grids[m-1].mean(axis=0)
        gain = mean.max(axis=1)-mean[0, 0]
        boot = []
        for start in range(0, len(weights), 100):
            average = (weights[start:start+100] @ grids[m-1].reshape(24, -1)).reshape(-1, len(beta), len(phi))
            boot.append(average.max(axis=2)-average[:, 0, 0, None])
        low, high = np.quantile(np.concatenate(boot), [.025, .975], axis=0)
        line, = ax.plot(beta/np.pi, gain, linestyle=style, linewidth=1.6, label=f"$m={m}$")
        ax.fill_between(beta/np.pi, low, high, color=line.get_color(), alpha=.13, linewidth=0)
        for angle, value, lo, hi in zip(beta, gain, low, high):
            curve_rows.append(dict(size=m, beta=angle, gain=value, low=lo, high=hi))
    ax.axhline(0, color="gray", ls=":", lw=.8)
    ax.set(xlim=(0, .5), ylim=(-1.03, .035), xlabel=r"System measurement angle $\beta/\pi$",
           ylabel=r"$\max_\phi\langle\chi_{\beta,\phi}\rangle-\langle\chi_z\rangle$ (bits)")
    ax.set_xticks([0, .125, .25, .375, .5])
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(alpha=.18)
    save_figure(fig, args.output/"pointer_basis_expanded.pdf")
    save_figure(fig, args.output/"pointer_basis_expanded.png", dpi=180)
    plt.close(fig)
    with (args.output/"curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_rows[0]); writer.writeheader(); writer.writerows(curve_rows)
    report = dict(params=p, source_params_sha256=_sha256(source), script_sha256=_sha256(Path(__file__)),
                  reference_script_sha256=_sha256(Path(__file__).with_name("check_pointer_exact.py")),
                  beta_count=len(beta), phi_count=len(phi), smallest_nonzero_beta=float(beta[1]),
                  time=60, individual_scans=len(rows), direct_reduction_error=direct_error,
                  accelerated_matrix_error=accelerated_error, max_individual_gain=max(r["gain_bits"] for r in rows),
                  z_argmax_count=sum(r["best_beta"]==0 for r in rows),
                  advantage_above_1e_minus12_count=sum(r["gain_bits"]>1e-12 for r in rows),
                  bootstrap_draws=2000, bootstrap_seed=20260909,
                  averaging="Mean fragments within realization; mean realizations; maximize over azimuth. Bootstrap whole realizations with fragments fixed and repeat maximization.",
                  seconds=time.perf_counter()-started)
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("params", "averaging")}))


if __name__ == "__main__":
    main()
