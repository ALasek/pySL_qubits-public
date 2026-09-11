"""
CLI README

Usage:
  python analyze_pointer_basis_scan.py --subdir SUBDIR [options]

Inputs:
  The analyzer reads pySL v2 run-store snapshots, not standalone snapshot globs.
  Each saved file under data/<subdir>/runs/<run_id>/psi_snapshots/ is treated as
  one realization. This matches AverageOverRunsN: scalar metrics are averaged,
  while wavefunctions are never averaged.

Common options:
  --filter KEY=VALUE          Extra run-parameter filter. Repeatable.
  --run-id RUN_ID             Analyze a specific v2 run ID. Repeatable.
  --out-dir DIR               Output directory. Default:
                              data/figs/<subdir>/pointer_basis_scan.
  --analysis-name NAME        Named output bundle under data/figs/<subdir>/.
  --fragment-sizes 4 6 8      Fragment sizes to sample from environment qubits.
  --num-fragments N           Number of fragments per size.
  --fragment-mode MODE        random, prefix-permutation, or explicit.
  --explicit-fragments FILE   JSON fragments for fragment-mode=explicit.
  --beta-count N              Number of beta grid points in [0, pi].
  --phi-count N               Number of phi grid points in [0, 2pi).
  --device cuda|cpu           CuPy GPU path or NumPy CPU path.
  --dtype complex64|complex128
  --basis-batch-size N|auto   Candidate basis eigensolve batch size.
  --time-indices IDX ...      Analyze only selected saved snapshot indices.
  --times T ...               Analyze nearest saved snapshots to physical times.
  --max-realizations N        Limit snapshot files for quick checks.
  --system-qubit Q            Default: 0.
  --environment-qubits Q ...  Default: all qubits except system qubit.
  --endianness big|little     Default: big, matching pySL qubit labels.
  --save-local-samples        Write per-sample local best-basis rows.
  --bootstrap N               Bootstrap error bars over realization IDs.
  --plot                      Save diagnostic PNG plots.
  --no-normalize              Trust stored snapshot norm.
  --no-checkpoint             Disable fragment-sample checkpointing.
  --force-recompute-checkpoints
                              Ignore existing checkpoint entries.
"""

import argparse
import json
import os

from src.analysis.pointer_basis_scan import (
    analyze_pointer_basis_scan,
    default_output_dir,
    parse_filter,
)


def _parse_filter_arg(value):
    try:
        return parse_filter(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main():
    parser = argparse.ArgumentParser(description="Scan candidate system pointer bases from saved pySL psi snapshots.")
    parser.add_argument("--subdir", required=True, help="Data subdirectory under data/.")
    parser.add_argument("--out-dir", help="Output directory.")
    parser.add_argument("--analysis-name", help="Named output bundle under data/figs/<subdir>/.")
    parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        type=_parse_filter_arg,
        metavar="KEY=VALUE",
        help="Extra run-parameter filter, for example --filter nqubits_E=17.",
    )
    parser.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        help="Analyze a specific v2 run ID. Repeatable.",
    )
    parser.add_argument("--fragment-sizes", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--num-fragments", type=int, default=50)
    parser.add_argument("--fragment-seed", type=int, default=1234)
    parser.add_argument(
        "--fragment-mode",
        choices=["random", "prefix-permutation", "explicit"],
        default="random",
    )
    parser.add_argument("--explicit-fragments", help="JSON fragment map for fragment-mode=explicit.")
    parser.add_argument("--beta-count", type=int, default=31)
    parser.add_argument("--phi-count", type=int, default=64)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--basis-batch-size", default="auto")
    parser.add_argument("--eig-eps", type=float, default=1e-12)
    parser.add_argument("--time-indices", nargs="+", type=int)
    parser.add_argument("--times", nargs="+", type=float, dest="times_requested")
    parser.add_argument("--max-realizations", type=int)
    parser.add_argument("--system-qubit", type=int, default=0)
    parser.add_argument("--environment-qubits", nargs="+", type=int)
    parser.add_argument("--endianness", choices=["big", "little"], default="big")
    parser.add_argument("--no-normalize", action="store_true", help="Do not renormalize snapshots before analysis.")
    parser.add_argument("--save-local-samples", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--force-recompute-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    filters = dict(args.filters or [])
    explicit_fragments = None
    if args.explicit_fragments:
        with open(args.explicit_fragments, encoding="utf-8") as f:
            explicit_fragments = json.load(f)

    out_dir = args.out_dir or default_output_dir(args.subdir, analysis_name=args.analysis_name)
    summary = analyze_pointer_basis_scan(
        subdir=args.subdir,
        out_dir=out_dir,
        filters=filters,
        run_ids=args.run_ids,
        fragment_sizes=args.fragment_sizes,
        num_fragments=args.num_fragments,
        fragment_seed=args.fragment_seed,
        fragment_mode=args.fragment_mode,
        explicit_fragments=explicit_fragments,
        beta_count=args.beta_count,
        phi_count=args.phi_count,
        device=args.device,
        dtype=args.dtype,
        eig_eps=args.eig_eps,
        time_indices=args.time_indices,
        times_requested=args.times_requested,
        max_realizations=args.max_realizations,
        normalize=not args.no_normalize,
        save_local_samples=args.save_local_samples,
        bootstrap=args.bootstrap,
        plot=args.plot,
        basis_batch_size=args.basis_batch_size,
        checkpoint=not args.no_checkpoint,
        force_recompute_checkpoints=args.force_recompute_checkpoints,
        system_qubit=args.system_qubit,
        environment_qubits=args.environment_qubits,
        endianness=args.endianness,
        verbose=args.verbose,
    )

    print("Pointer-basis scan complete")
    print(f"subdir = {args.subdir}")
    print(f"filters = {filters}")
    if args.run_ids:
        print(f"run_ids = {args.run_ids}")
    print(f"realizations = {summary['num_realizations']}")
    print(f"out_dir = {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
