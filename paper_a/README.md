# Paper A: processed data and reproduction

This package accompanies *Environment Alignment and Redundant Record
Formation in Imperfect-CNOT Quantum Darwinism*, by Aleksander Lasek and
Paweł Horodecki. Release `paper-a-v1.2.0` matches the final figure set.

## Redraw the figures

From the repository root, in a Python 3.11 virtual environment:

```sh
python -m pip install -r paper_a/requirements.txt
python paper_a/reproduce_figures.py
```

This uses bundled processed data and requires no GPU, CUDA, raw trajectories,
private repositories, or environment variables. LaTeX must be on PATH with
`fontenc`, `lmodern`, `amsmath`, `amssymb`, and Matplotlib's TeX support
packages (`type1cm`, `cm-super`). TeX Live and MiKTeX can supply these.

The output directory is `paper_a/output/`; change it with `--output PATH`.
It contains the 15 numerical plots (main Figs. 2–9 and Supplemental Figs.
S1–S7), their hashes, and derived summaries. The `FIGURES` mapping in
`reproduce_figures.py` gives the filenames and manuscript numbering.
Figure 1 is supplied as TikZ source:

```sh
cd paper_a
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output schematic.tex
```

That command also needs `standalone` and TikZ/PGF. The schematic has no
numerical inputs. Do not use an input-data directory as `--output`.

All 15 regenerated plot PDFs match the manuscript's rendered pixels at
72 dpi in the checked Python 3.11.9 environment with the pinned dependencies.
The release also passes 23 CPU analysis tests. `VALIDATION.json` records the
checks. PDF file hashes can vary with timestamps and TeX versions; they are
not a portable numerical equivalence test.

## Data and conventions

- `figures/paper_a/source_manifests/`: saved tables for the original main
  plots and numerical validation. The complete threshold-size manifest
  supplies current Fig. 7(b).
- `analysis/theta_sweeps/`: the full preparation/angle grid (Fig. 2) and
  local-field angle sweeps (Fig. S6), including seeds and fragment choices.
- `analysis/mixed_state_comparison_4qubits/` and `analysis/matched_pure_mixed/`:
  local records and SBS-distance comparisons for Fig. S7(a–c). Four qubits
  means the system plus three environment qubits; one witness is observed.
- `analysis/mixed_basis_size/`: fixed-time, time-optimized, and refined
  measurement-basis results for Fig. S7(d). Environment size excludes the
  system; one witness remains observed as unobserved witnesses are added.
- The other `analysis/` directories retain the expanded pure-state basis
  scan, paired preparations, percentile comparison, and field mean/width data.
- `scripts/`: plotters, exact calculations, validation routines, and tests.
- `SOURCE_MANIFEST.json`: source/data hashes and the manuscript revision used
  to assemble this package. Plotter adaptations separate input and output paths.
  Historical calculation manifests retain the original calculation-time
  hashes; `SOURCE_MANIFEST.json` records the current packaged files. Absolute
  seed-source paths have been reduced to their original batch-relative paths;
  the bundled seeds are sufficient for recalculation.

Information is in bits or normalized by the pointer entropy as indicated
in the manuscript. Raw NPZ trajectory arrays used by older validation
commands store information in nats. Each figure retains its stated order
of averaging. Fig. 7(b) finds the threshold size within each realization,
averages equivalent preparations with paired seeds, and then averages
realizations with a crossing. Noncrossings remain undefined and their
counts are reported. Persistence is assessed at the sampled times.

In Fig. S7(a), purity changes at fixed initial Bloch direction. Panels
(b,c) instead match populations, so purity and direction both change.
Panel (d) maximizes the information gain over measurement axes and the
finite time interval [0,20]. Optimized SBS distances and information gains
are numerical candidates, not certified global optima.

## Recompute rather than redraw

The redraw command does not regenerate trajectories or rerun optimizations.
The root `REPRODUCING_PAPER_A.md` documents GPU batches; full raw outputs are
available from the authors on reasonable request. The newer exact studies
run on CPU. From `paper_a/`, in a working copy where analysis outputs may be
regenerated:

```sh
python scripts/validation/plot_theta_sweeps.py
python scripts/validation/plot_theta_preparation_map.py
python scripts/validation/mixed_state_comparison.py --environment-qubits 3
python scripts/validation/compare_matched_preparations.py
python scripts/validation/scan_mixed_basis_size.py
```

The last three commands are separate from the quick redraw; SBS optimization
can be expensive. The matched-preparation calculation reads the preceding
mixed-state SBS results. All commands retain the paper's independent-witness
assumption. The mixed-state calculations explicitly evolve density matrices
or use equivalent reduced-state formulas.

For the original exact reconstruction, provide the eight baseline raw batches:

```sh
python scripts/validation/recalculate_exact.py --source-roots /path/to/raw/data --output-root /path/to/separate/exact/data
```

From the repository root, install `pytest` and run the CPU validation suite:

```sh
python -m pytest paper_a/scripts/validation/test_recalculate_exact.py paper_a/scripts/validation/test_bootstrap_persistence.py paper_a/scripts/validation/test_threshold_sizes.py paper_a/scripts/validation/test_unified_scaling.py paper_a/scripts/validation/test_theta_sweeps.py paper_a/scripts/validation/test_mixed_state_comparison.py -q
```

The package is distributed under the repository's BSD-3-Clause license.
No manuscript draft or private research notes are included.
