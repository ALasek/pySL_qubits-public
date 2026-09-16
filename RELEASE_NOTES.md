# Release notes

## paper-a-v1.2.0 — 2026-09-16

This release matches the final Paper A figure set: main Figs. 2–9 and
Supplemental Figs. S1–S7, with Fig. 1 supplied as standalone TikZ source.
The simulator and GPU batch configurations are unchanged from v1.1.0.

- Added the preparation/interaction-angle map and the three Holevo cuts at
  p=0.15, 0.25, and 0.75 (Fig. 2).
- Added the comparison of interaction-angle dependence for uniform and
  Gaussian-distributed local fields (Fig. S6).
- Added mixed-state and population-matched pure-state results, and the
  environment-size dependence of the time-optimized Holevo gain (Fig. S7),
  with the CPU calculation and validation scripts.
- Updated plot terminology and the model schematic to match the manuscript.
- Updated citation metadata and processed-data/script hashes.

Validation: all 15 regenerated numerical figures match the manuscript's
rendered pixels at 72 dpi; 23 CPU analysis tests pass; the standalone
schematic compiles. The processed-data redraw is also checked in an isolated
copy. Full GPU simulation batches were not rerun for this release.

The mixed-state SBS distances and optimized measurement gains are numerical
optimization results, not certified global optima. The size study concerns
one observed witness; increasing environment size adds unobserved witnesses.

No Zenodo deposit or DOI is assigned to this release.

## paper-a-v1.1.0 — 2026-09-09

This successor to the historical `paper-a-v1.0.0` release captures the latest
Paper-A simulator, batch configurations, analysis code, and the fixed-RMS
Gaussian field mean/width study.

The public `paper_a/` package contains saved figure tables and CPU plotting
scripts. It regenerates the release plots without raw GPU outputs or the
manuscript notes repository. The detailed from-scratch GPU analysis remains in
this repository; the large raw production batches are available from the
authors upon reasonable request.

For Fig. 6(b), each realization supplies a first threshold fragment size from
its q10 Holevo curve. Defined paired geometries are averaged within a
realization before defined realization values are averaged. Noncrossings remain
undefined and are reported through their crossing counts; the estimator does
not average or invert redundancies first.

No Zenodo release or DOI is assigned to this version.
