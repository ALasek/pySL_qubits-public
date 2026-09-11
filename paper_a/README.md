# Paper A: processed data and reproduction

This package accompanies *Environment Alignment and Redundant Record
Formation in Imperfect-CNOT Quantum Darwinism*, by Aleksander Lasek and
Paweł Horodecki. It is included in the `paper-a-v1.1.0` software snapshot.

## Redraw the published plots

From the repository root, in a Python 3.11 virtual environment:

```sh
python -m pip install -r paper_a/requirements.txt
python paper_a/reproduce_figures.py
```

This uses the bundled processed data and requires no GPU, CUDA, raw
trajectories, private repositories, or environment variables. LaTeX must
be on PATH with `fontenc`, `lmodern`, `amsmath`, `amssymb`, and the standard
Matplotlib TeX support packages (`type1cm`, `cm-super`). TeX Live and MiKTeX
can supply these. Matplotlib uses LaTeX only to render labels.

The output directory is `paper_a/output/`; change it with `--output PATH`.
It contains main Figs. 2–8 and Supplemental Figs. S1–S5, their hashes, and
small derived summary files. Figure 1 is supplied as TikZ source:

```sh
cd paper_a
pdflatex -interaction=nonstopmode -halt-on-error schematic.tex
```

That command additionally needs `standalone` and TikZ/PGF. The schematic
has no numerical inputs. Do not use an input-data directory as `--output`.

The release check ran on Python 3.11.9 with the pinned versions above.
All 12 regenerated plot PDFs matched the manuscript's rendered pixels
exactly at 72 dpi. The six main-figure tables also matched their retained
source hashes. PDF file hashes can vary with timestamps and TeX versions;
they are not a portable numerical equivalence test.

## Contents and interpretation

- `figures/paper_a/source_manifests/`: the tables used for main Figs. 2–7,
  S2, S4, and the fixed-time larger-environment check. The complete
  `threshold_sizes_complete_2026_09_08` manifest is the source for Fig. 6(b).
- `analysis/`: expanded pointer-basis curves (Fig. 8), matched-preparation
  statistics and paired differences (S1), the percentile comparison (S3),
  and field mean/width statistics (S5).
- `scripts/`: retained plotters, exact conditional-state calculations,
  validation routines, and tests. The simple command above is the portable
  entry point; older validation commands may require raw inputs and explicit
  path arguments.
- `SOURCE_MANIFEST.json`: hashes of bundled scripts/data and the manuscript
  revision from which they were assembled.

Information is in bits or normalized by the pointer entropy as indicated
in the manuscript. Original NPZ information arrays used by raw-data checks
are in nats. Figures average fragment statistics within each realization
before averaging realizations. Fig. 6(b) finds the threshold fragment size
within each realization, averages equivalent preparations with paired seeds,
and then averages over realizations with a crossing. Undefined crossings
are excluded and marked as conditional in the paper. Persistence means
passing at every sampled time; a disorder-averaged minimum above threshold
does not imply that every realization passes.

## Recompute rather than redraw

The processed-data workflow redraws the reported results; it does not
independently regenerate the underlying trajectories or random fragments.
The root `REPRODUCING_PAPER_A.md` documents the GPU batches and configurations.
Full raw outputs are available from the authors on reasonable request.
With the eight baseline batches under a chosen raw-data directory, exact
independent-qubit reconstruction can be run on a CPU:

```sh
python paper_a/scripts/validation/recalculate_exact.py --source-roots /path/to/raw/data --output-root /path/to/separate/exact/data
python -m pytest paper_a/scripts/validation/test_recalculate_exact.py paper_a/scripts/validation/test_bootstrap_persistence.py -q
```

Install `pytest` for the tests. The additional basis-scan, field mean/width,
and finite-size scripts accept their own raw-data arguments; use `--help`.
The exact method assumes pure product witness preparations and no
environment–environment interactions, as in this paper.

The software and this reproduction package are distributed under the
repository's BSD-3-Clause license. The retained plotting scripts were copied
from the author's research repositories; the source manifest records their
origins. No manuscript draft, private research notes, or raw trajectories
are needed by the plotting command.
