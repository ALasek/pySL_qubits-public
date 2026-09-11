# Release notes

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
