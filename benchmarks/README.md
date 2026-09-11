# Solver regression benchmarks

This suite freezes small deterministic GPU runs before numerical changes to the time evolution. The reference artifacts are ordinary v2 run directories containing `params.json`, `metadata.json`, and `results.npz`.

Commands:

```bash
python benchmark_suite.py list
python benchmark_suite.py verify --profile strict
python benchmark_suite.py verify --profile scientific --report data/benchmark_scientific_report.json
```

`strict` is the unchanged-solver regression profile. `scientific` is a deliberately broader first screen for expected integrator changes; a pass is not a substitute for inspecting field-level differences or timestep convergence.

Recording or replacing baselines is explicit:

```bash
python benchmark_suite.py record --force
```

Each recording updates `manifest.json` with the Git commit, a hash of the numerical source files, Python/CUDA/GPU provenance, artifact hashes, and compact summaries of representative observables.
