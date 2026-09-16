# pySL

This public repository starts from a fresh snapshot of the development
repository. The `paper-a-v1.2.0` release matches the final Paper A figures and
processed data. The earlier `paper-a-v1.1.0` tag remains available.
Earlier development history and branches are not included.
 
## Table of Contents
- [pySL](#pysl)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Installation](#installation)
    - [GPU environment](#gpu-environment)
    - [(Optional) ffmpeg](#optional-ffmpeg)
  - [Usage](#usage)
    - [Input parameters](#input-parameters)
    - [Parameter sweeps](#parameter-sweeps)
    - [Batch analysis](#batch-analysis)
    - [Spectral analysis](#spectral-analysis)
  - [Paper A reproducibility](#paper-a-reproducibility)
  - [Authors and provenance](#authors-and-provenance)
  - [Citation](#citation)
  - [License](#license)
  - [Publications](#publications)
  - [Acknowledgments](#acknowledgments)


## About
The Python Staggered Leapfrog (pySL) package solves the time-dependent
Schrödinger equation using CUDA acceleration. This fork adds the simulation,
batching, diagnostics, and analysis workflows used for the Quantum Darwinism
Paper-A study.

## Installation
pySL works on NVIDIA GPU systems running Windows or Linux. The Paper-A release
provides dependency sets for CUDA 12 and CUDA 13; select the one matching the
host installation.

### GPU environment
Confirm that the NVIDIA driver sees the target GPU before creating the Python
environment:

```bash
nvidia-smi
```

Install or update CUDA software only through NVIDIA's current [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) or [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) guide when the host needs it. Do not remove or replace an existing driver as part of this project's setup.

Create and activate a virtual environment, then install the existing dependency
set for exactly one CUDA major version:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# POSIX: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda12.txt  # CUDA 12
# or: python -m pip install -r requirements-cuda13.txt  # CUDA 13
```

Install test tooling only when needed:

```bash
python -m pip install -r requirements-dev.txt
```

The CUDA-specific requirements include the shared dependencies and one mutually
exclusive CuPy wheel. Verify the selected environment without running a
simulation:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

### (Optional) ffmpeg

`ffmpeg` is used to generate plot animations. This is not a required feature since plotting is not necessary and other software can be used for analysis purposes. Installation will vary depending on the host operating system.


## Usage

To run the solver, use the command:
```
python pySL.py [TARGET_GPU_ID]
```
The argument following the command indicates which GPU to use in a multi-GPU system. If no argument is given or if the argument is incorrect, the simulation will default to GPU0.

Standalone runs load one JSON parameter file from `run_params/`. By default this is `run_params/params.json`:
```bash
python pySL.py
```

To select a different standalone parameter file, pass the filename with `--params`:
```bash
python pySL.py --params params_MBLtrans_yesQDreg.json
```

With an explicit GPU id, put the GPU id before flags:
```bash
python pySL.py 1 --params params_MBLtrans_yesQDreg.json
```

`pySL.py --params` expects a JSON file under `run_params/`, not a Python file from `batch_configs/`. Use `pySLbatch.py --config ...` for batch config scripts.

An output directory will be created for each different simulation name. Within this directory several results are recorded in plain text files.
* `eigs.txt` lists the lowest 5 eigenstates found using the built-in eigensolver.
* `wave.txt` outputs the complex wave function of the initial state as determined by the user.
* `waves\wave_iteraton_XXX.txt` are the complex wave functions at every iteration as defined by `PrintStep` in the input parameters.

### Input parameters
Standalone simulation parameters are JSON files in `run_params/`. Use `run_params/params.example.json` as the complete reference and copy/edit it into a run-specific file such as `run_params/params_MBLtrans_yesQDreg.json`.

Plot display behavior can be controlled with `plot_mode`:
- `auto` leaves Matplotlib backend behavior unchanged. This is the default and preserves normal Spyder or notebook behavior.
- `external_blocking` uses the standard blocking `plt.show()` behavior for external plot windows.
- `external_nonblocking` opens external plot windows during the run without blocking, then waits once at the end.
- `plot_pane` collects figures into the Tk plot pane.
- `none` suppresses plot display and closes figures after optional figure saving; use this for batch runs.

For timestep performance diagnosis, set `"profile_evolution": true` in the selected params JSON. This samples CUDA-event timings for the physical-state update, correlator `phi_j` updates, and noise-Hamiltonian assembly over the first `"profile_evolution_max_steps"` timesteps, then reports the breakdown in the final perf summary. It synchronizes GPU work during the sampled steps and should be disabled for production sweeps.

GPU memory logging is enabled by default with `"log_gpu_memory": true`. The final summary reports sampled CUDA device usage and CuPy memory-pool high-water marks; this is usually more reliable than Windows Performance Monitor for short-lived CUDA workspaces.

Hamiltonian precompute defaults to `"H_precompute_mode": "combined_cpu"`, which builds one combined CSR matrix on CPU and transfers it to GPU so each timestep uses a single sparse matvec. `"grouped_cpu"` extracts diagonal terms into one vector and combines non-diagonal terms into CPU-built CSR groups of `"H_precompute_group_size"` terms; this is the memory-aware fallback before `"sum_terms"`. `"sum_terms"` avoids CPU combination but applies each Hamiltonian term separately per timestep and can be much slower for large N.

The `SL` evolver uses a synchronized second-order staggered leapfrog step. For work vectors `r` and `q`, each timestep applies an imaginary half-step, a real full-step, and a final imaginary half-step. This symmetric form is time-reversible, supports real and complex Hermitian Hamiltonians, and keeps snapshots, gates, correlators, and norm diagnostics at the same physical time. The explicit three-operation form is intentional; do not collapse it into two same-time full updates, which is only first-order symplectic Euler.

The `Mironowicz_rand_EE_transverse` Hamiltonian variant keeps the corrected random Mironowicz self-fields and replaces nearest-neighbor E-E coupling with the explicit transverse term `0.5 * Z kron (I + X + Y)`. `Mironowicz_rand_EE_ZX` is the simplified variant with only the `Z kron X` pair term. `Mironowicz_rand_EE_ZZ_to_ZX` uses `Z kron (cos(Mironowicz_ZZ_to_ZX_epsilon) Z + sin(Mironowicz_ZZ_to_ZX_epsilon) X)`.

Other input files such as an initial wave function or static potentials should be written as text files and placed in the `/input/` directory.

### Parameter sweeps
Batch runs are configured with Python files in `batch_configs/`. A config defines:
* `subdir`: output folder under `data/`.
* `params_default`: baseline simulation parameters.
* exactly one of `sweep` (a Cartesian product) or `cases` (an explicit list of correlated parameter sets).
* `analysis`: default plot axes and metric settings for later analysis.

Run the default theta sweep on a GPU machine with:
```bash
python pySLbatch.py --config batch_configs.mironowicz_theta_sweep
```

Run multiple independent sweep points concurrently with worker processes:
```bash
python pySLbatch.py --config batch_configs.mironowicz_theta_sweep --workers 3 --gpu-ids 0 --skip-existing
```

`--workers` controls concurrent simulations. `--gpu-ids` is a comma-separated list of physical GPU ids assigned round-robin to worker tasks; current one-GPU servers should use `--gpu-ids 0`. `--skip-existing` skips runs whose v2 `results.npz` already exists, which is useful when restarting a partial sweep. Use `--worker-stagger-seconds S` to stagger the first worker launches and reduce simultaneous allocation spikes.

Parallel batch runs should use `"plot_mode": "none"` and `"save_figures": "off"`/omitted in the batch config. Interactive plot panes and shared `latest_figs` output are intended for standalone runs, not concurrent workers.

Other included configs:
```bash
python pySLbatch.py --config batch_configs.mironowicz_mbl_sweep
python pySLbatch.py --config batch_configs.mironowicz_epsilon2_sweep
```

To resume partway through a sweep, skip completed combinations by zero-based index:
```bash
python pySLbatch.py --config batch_configs.mironowicz_theta_sweep --combo-start 10
```

Each batch writes `data/<subdir>/batch_manifest.json`. Schema-v2 manifests record the concrete sweep or explicit cases, defaults snapshot, parameter-set count, analysis defaults, and Git commit/dirty state. Existing `.rundata` pickle files are unchanged.

Parallel batches also append worker outcomes to `data/<subdir>/batch_status.jsonl`.

To write only the manifest without launching GPU simulations:
```bash
python pySLbatch.py --config batch_configs.mironowicz_theta_sweep --manifest-only
```

### Paper-A suite

`paper-a-v1.2.0` includes the final figure reproduction package, including
angle sweeps and mixed-state comparisons, alongside the simulator, batch
configurations, and fixed-RMS Gaussian field mean/width study.
The earlier `paper-a-v1.0.0` release is not included in this public repository.

The compact, public reproduction package is documented in
[paper_a/README.md](paper_a/README.md). Its saved tables and scripts regenerate
the release plots on CPU without raw simulation outputs or the manuscript
notes repository. [REPRODUCING_PAPER_A.md](REPRODUCING_PAPER_A.md) records the
release boundary, estimator, and the separate raw-data workflow.

The Paper-A suite accepts `--profile` and is launched sequentially so each batch and its analysis slice has a separate log. The default `smoke` profile is a quick GPU sanity sweep; `final` uses the publication grid and production averaging. The profiles write separate data subdirectories.

```bash
python scripts/run_paper_a_suite.py --profile smoke --dry-run
python scripts/run_paper_a_suite.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_a_suite.py --profile final --workers 1 --gpu-ids 0 --skip-existing
python scripts/run_paper_a_suite.py --profile final --manifest-only
python scripts/run_paper_a_suite.py --profile final --figures-only
```

`--jobs` accepts a comma-separated subset: `lambda_collapse`, `uniform_field`, `disorder_field`, or `stability_map`. `--no-figures`, `--no-skip-existing`, `--worker-stagger-seconds`, and `--keep-going` apply to the selected jobs. Dry runs enumerate exact combination counts, output subdirectories, and commands without creating manifests, logs, or subprocesses. Non-dry runs write command logs and `orchestration_status.jsonl` beneath `logs/paper_a/`.

The figure phase maps jobs to analysis slices as follows:

* `lambda_collapse`: one `psi_bias` × `Mironowicz_theta` figure bundle for each `nqubits_E`.
* `uniform_field`: one `Mironowicz_h0` × `Mironowicz_theta` bundle for each representative `psi_bias`.
* `disorder_field`: one `Mironowicz_alpha2` × `Mironowicz_theta` bundle for each representative `psi_bias`.
* `stability_map`: one symmetry-reduced `psi_bias` (0 to 0.5) × `Mironowicz_alpha2` bundle for each fixed `Mironowicz_theta`; its nonuniform field grid resolves the critical scale and retains the older strong-field range.

Every job stores the full sampled time series for mutual information, Z-Holevo information, and Z-discord. Paper-A runs reuse deterministic fragment sets within each realization, retain the 0.1 and 0.5 fragment quantiles, and store single-site conditional-branch fidelities. Batch analysis therefore reports same-time Holevo/discord, mean-curve and quantile-curve redundancy estimates, formation/occupancy/observed lifetime, and the typical single-site overlap exponent. The old polynomial QD slope remains in the export as a screening proxy. With `fragment_half_only=true`, failure to reach the Holevo threshold by half of the environment is reported as no redundant record rather than extrapolated beyond the simulated fragment range.

The focused submission batch is an explicit 13-point matched-alignment grid in
`lambda_target`, with two or three inequivalent `(psi_bias, Mironowicz_theta)`
geometries at each value. The final profile uses `nqubits_E=16`, 24
realizations, and up to 64 deterministic fragment samples.

```bash
python scripts/run_paper_a_submission.py --profile smoke --dry-run
python scripts/run_paper_a_submission.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_a_submission.py --profile final --dry-run
python scripts/run_paper_a_submission.py --profile final --workers 1 --gpu-ids 0
```

Generate the publication-oriented late-window diagnostics without launching simulations:

```bash
python analysis_scripts/paper_a_submission_panels.py --data-root data --output-dir data/figs/Paper_A_submission
```

The analyzer uses per-realization medians over `t=20..40` for Lambda/stability data and `t=30..60` for uniform/disorder fields. It exports normalized half-fragment Holevo and discord, conservative q10 redundancy, record formation/lifetime/occupancy, and both overlap rates. Here `kappa_typ = -mean(log(F))/2` and `kappa_ann = -log(mean(sqrt(F)))`, because stored branch fidelity is `F=B^2`. Confidence intervals resample realizations; stored fragment samples are retained for a later hierarchical sensitivity analysis but are not silently treated as independent realizations.

The compact release package contains its own saved plotting inputs. The detailed
analyzers require the corresponding raw v2 batch directories; those large
outputs are available from the authors upon reasonable request.

The final matched-strength field control is an explicit 26-case batch rather than another broad map. It compares a coherent uniform field with a zero-mean Gaussian field at the same field-strength-to-`H_SE_J` ratio for the aligned blind geometry, with a smaller longitudinal control set. Final runs use 24 realizations, 64 deterministic fragment samples, and persist raw fragment samples for hierarchical uncertainty checks.

```bash
python pySLbatch.py --config batch_configs.paper_a_field_control --profile final --manifest-only
python pySLbatch.py --config batch_configs.paper_a_field_control --profile final --workers 1 --gpu-ids 0 --skip-existing
```

After copying the completed batch locally, build the matched-profile, paired-contrast, commuting-axis, and fragment-scaling panels with:

```bash
python analysis_scripts/paper_a_field_profile_control.py \
  --batch-dir data/Paper_A_final_FieldProfileControl \
  --output-dir data/figs/Paper_A_field_profile_control
```

The optional higher-size Paper-A validation is deliberately narrow: one
moderate no-field record case and four aligned-field rescue/strong-field
controls at $N_E=20$.  Its smoke profile already allocates the final Hilbert
space, but evolves only one realization to $T=2$, so run it before committing
the GPU to the production batch:

```bash
python scripts/run_paper_a_higher_n_validation.py --profile smoke --dry-run
python scripts/run_paper_a_higher_n_validation.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_a_higher_n_validation.py --profile final --dry-run
python scripts/run_paper_a_higher_n_validation.py --profile final --workers 1 --gpu-ids 0
```

This is a full-state implementation cross-check.  Large-$N_E$ scaling in the
$H_{EE}=0$ model should primarily use the exact products of conditional
single-qubit branch overlaps instead of exponentially expensive state-vector
runs.

The Fig. 5(b) finite-pool extension is a separate fixed-time batch.  It adds
$N_E=20,24$ for the two distinct iid threshold levels, $\Lambda=0.75$ and
$1$, using eight realizations and 64 stored fragments per size at $t=36$.
These runs test the finite-environment quantile estimator against the exactly
$N_E$-independent iid benchmark; they are not a numerical derivation of that
benchmark.  The smoke profile allocates the largest $N_E=24$ state but evolves
only one realization to $T=0.02$.

```bash
python scripts/run_paper_a_fragment_threshold_finite_n.py --profile smoke --dry-run
python scripts/run_paper_a_fragment_threshold_finite_n.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_a_fragment_threshold_finite_n.py --profile final --manifest-only
python scripts/run_paper_a_fragment_threshold_finite_n.py --profile final --workers 1 --gpu-ids 0
```

The Paper-A timestep-convergence batch repeats three retained regimes with the
first four production seeds and $\Delta t=0.001$: the no-field
$\Lambda=1/2$ case, Gaussian-$0.5g$ rescue, and Gaussian-$30g$ suppression.
Its raw fragment samples can be paired with the existing $\Delta t=0.002$
matched-$\Lambda$ and field-profile batches:

```bash
python pySLbatch.py --config batch_configs.paper_a_timestep_convergence --profile smoke --workers 1 --gpu-ids 0 --skip-existing
python pySLbatch.py --config batch_configs.paper_a_timestep_convergence --profile final --manifest-only
python pySLbatch.py --config batch_configs.paper_a_timestep_convergence --profile final --workers 1 --gpu-ids 0 --skip-existing
```

The optional fixed-time high-size check uses one no-field
$N_E=24$, $\Lambda=1/2$ case at $t=36$, with eight paired seeds and 32
fragment samples per size.  Its smoke profile deliberately allocates the full
$N_E=24$ Hilbert space, but evolves only one realization to $T=0.02$ with four
fragments per size; run that memory check before the final batch:

```bash
python pySLbatch.py --config batch_configs.paper_a_high_n_fixed_time --profile smoke --workers 1 --gpu-ids 0 --skip-existing
python pySLbatch.py --config batch_configs.paper_a_high_n_fixed_time --profile final --manifest-only
python pySLbatch.py --config batch_configs.paper_a_high_n_fixed_time --profile final --workers 1 --gpu-ids 0 --skip-existing
```

After both review batches and the earlier higher-$N$ reference are available
under one data root, reproduce the compact convergence and fixed-time tables
with:

```bash
python analysis_scripts/paper_a_review_validation.py --data-root data --n20-batch-dir data/Paper_A_final_HigherNValidation --output-dir data/figs/Paper_A_review_validation
```

After the final batch and the matching $N_E=16$ production batches are
available locally, generate the paired validation bundle with:

```bash
python analysis_scripts/paper_a_higher_n_validation.py \
  --high-n-batch-dir data/Paper_A_final_HigherNValidation \
  --matched-lambda-dir /path/to/data/Paper_A_final_SubmissionMatchedLambda \
  --field-control-dir /path/to/data/Paper_A_final_FieldProfileControl \
  --output-dir data/figs/Paper_A_final_HigherNValidation
```

### Paper-B focused suite

The Paper-B suite is an intentionally bounded mechanism test, not a production sweep. It writes an identical CZ-class record until `staged_write_time=10`, removes every S-E term, and then evolves only under local E fields and E-E storage interactions. One command runs the batch and creates best-post-write and final-time analysis bundles:

```bash
python scripts/run_paper_b_suite.py --profile smoke --dry-run
python scripts/run_paper_b_suite.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_b_suite.py --profile focused --workers 1 --gpu-ids 0 --skip-existing
python scripts/run_paper_b_suite.py --profile focused --manifest-only
python scripts/run_paper_b_suite.py --profile focused --figures-only
```

`smoke` contains 16 combinations at `N_E=6`, one realization, `dT=0.02`, and two field scales. `focused` contains 48 combinations at `N_E=8,10`, three realizations, `dT=0.01`, and `W=0,0.2,0.6`. Both profiles compare chain `ZZ`, chain `ZX` with uniform/Gaussian/quasiperiodic fields, the historical composite-transverse interaction, and disjoint-`ZX` source/target field masks. The disjoint bonds are necessary because interior sites in the directed nearest-neighbor `ZX` chain are both sources and targets.

In addition to mutual information, Holevo information, discord, and redundancy lifetime, the suite stores:

* the pointer-conditioned storage-energy density gap and branch energy-density widths at the write/store boundary, which screen for the Cao-Nussinov conserved-density mechanism;
* mean and 10th-percentile single-site conditional-branch trace distance, plus its post-write lifetime and occupancy;
* an X-record-axis connected two-time correlator from one source site and spread summaries;
* the spatial weight and radius of `2 Im <O_i(t) O_j(0)>`, a cheap state-averaged commutator-expectation proxy.

The last diagnostic is not a squared-commutator OTOC and can vanish by cancellation. Treat it as a fast mechanism screen; reserve a full forward/backward OTOC for selected small-system points if the focused suite supports the protection claim.

New simulation runs are saved in a v2 file layout:
```text
data/<subdir>/
  run_index.jsonl
  runs/<run_id>/
    params.json
    metadata.json
    results.npz
```

Standalone `pySL.py` runs default to `data/nonbatch/`. Batch runs still use the explicit `subdir` from their config.

The v2 store uses a stable hash of canonical `params.json` for `run_id`, keeps analysis arrays in `results.npz`, and keeps a lightweight `run_index.jsonl` so analysis can find matching runs without unpickling every legacy file. Old `.rundata` files can still be read as a fallback. Set `"save_legacy_rundata": true` only when you explicitly need a new full-object legacy pickle; this can exceed RAM for large runs.

### Solver regression benchmarks

Small deterministic GPU baselines are tracked under `benchmarks/`. They cover a real Mironowicz Hamiltonian, the complex transverse Hamiltonian, correlator/discord diagnostics, and two-run averaging.

```bash
python benchmark_suite.py list
python benchmark_suite.py verify --profile strict
python benchmark_suite.py verify --profile scientific --report data/benchmark_scientific_report.json
```

The strict profile detects unintended changes to the current solver. The scientific profile is a broader first screen for intentional numerical-method changes and must be interpreted together with its per-field error report. Baseline replacement is explicit via `python benchmark_suite.py record --force`; see `benchmarks/README.md` for details.

Set `"store_psi": true` to save full wavefunction snapshots at each printed sample time. These are saved per realization, not averaged, under `data/<subdir>/runs/<run_id>/psi_snapshots/`; metadata is written to `psi_snapshots.json`. Use `StoredRunResult.get_stored_psi(t_idx, run_index=...)` after loading a v2 run to reconstruct a complex snapshot.

To migrate old `.rundata` files into the v2 layout without deleting the originals:
```bash
python scripts/migrate_rundata.py --subdir Mironowicz_rand_ThetaSweep
```

Preview a migration without writing v2 outputs:
```bash
python scripts/migrate_rundata.py --subdir Mironowicz_rand_ThetaSweep --dry-run
```

### Batch analysis
Analyze a completed batch from its manifest:
```bash
python pySL_batchAnalyze.py --subdir Mironowicz_rand_ThetaSweep
```

The analyzer uses the manifest's default `x` and `y` axes. Override them when needed:
```bash
python pySL_batchAnalyze.py --subdir Mironowicz_rand_ThetaSweep --x Mironowicz_theta --y Mironowicz_alpha2 --Tsample -1
```

If a manifest is not available yet, analyze from the original Python config:
```bash
python pySL_batchAnalyze.py --config batch_configs.mironowicz_theta_sweep
```

Figures are saved under `data/figs/<subdir>/`. Static analysis supports fixed-time Holevo and discord, Holevo redundancy, record lifetime/occupancy, local trace-distance, energy-density, correlator, and legacy QD-slope diagnostics. Restrict and order an export with `--metrics`; `holevo_discord_fhalf` creates a paired physical-information figure:

```bash
python pySL_batchAnalyze.py --subdir Paper_B_focused_FocusedWriteStore --Tsample 60 --setMinTime 20 --metrics holevo_discord_fhalf holevo_redundancy_q record_occupancy single_site_td_q10 qd_slope
```

Holevo and discord are reported in natural-log units. Treat fixed-time values and record-window occupancy/lifetime as the primary persistence diagnostics. `Tsample=-1`, QD slope, and the legacy redundancy score remain screening tools.

Run the bounded Paper-B suites, including their primary final/late-window figure bundles, with:

```bash
python scripts/run_paper_b_suite.py --profile smoke --workers 1 --gpu-ids 0
python scripts/run_paper_b_suite.py --profile focused --workers 1 --gpu-ids 0
python scripts/run_paper_b_suite.py --profile matched --workers 1 --gpu-ids 0
```

The `matched` profile compares uniform, exact-RMS Gaussian, exact-RMS quasiperiodic, and balanced binary fields for the chain `ZX` storage model at `W/K = 3, 5, 7`.

Create a fixed-time QD-slope animation over every saved sample time:
```bash
python pySL_batchAnimate.py --subdir Mironowicz_rand_ThetaSweep --x Mironowicz_theta --y Mironowicz_alpha2
```

The animation command is analysis-only and does not rerun simulations. It writes `qd_slope_time_cube.npz`, `qd_slope_time_animation.mp4`, and `animation_manifest.json` under `data/figs/<subdir>/<analysis_name>/`. Unlike static `--Tsample -1` plots, every animation frame uses `redundancy_slope_at_T(time_index)` at that fixed saved time; it never uses the best-time search. The default `setMinTime` is `0`, so static-analysis transient cutoffs stored in a manifest do not blank early animation frames.

Animate stored Z-basis discord at the fragment fraction closest to `F=1/2`:
```bash
python pySL_batchAnimate.py --subdir Mironowicz_rand_ThetaSweep --x Mironowicz_theta --y Mironowicz_alpha2 --metric discord_fhalf
```

Static batch analysis also writes `discord_fhalf.jpg` when the matched runs contain `Discord_Z_S_Ef_fractionsT_runAv`.

### Spectral analysis
Compute adjacent energy-level gap-ratio statistics over the same sweep grid as a batch config:
```bash
python pySL_spectralAnalyze.py --config batch_configs.mironowicz_mbl_sweep_TEST_small
```

The tool reuses the config's `sweep` and `analysis` axes, saves `data/<subdir>/spectral_gap_ratio.npz`, and shows the mean gap-ratio plot in the plot pane. If total `Sz` is conserved, `--sector auto` uses the middle charge sector. If `Sz` is not conserved, it falls back to full dense diagonalization. By default there is no hard size guard; add `--max-full-dim N` when you want one.

The default `--dtype auto` uses `float32` when the assembled Hamiltonian is real and `complex64` when it has nonzero imaginary entries. Use `--backend gpu` to run dense diagonalization through CuPy.

For non-conserved Hamiltonians, keep the total qubit count small or set an explicit guard:
```bash
python pySL_spectralAnalyze.py --config batch_configs/mironowicz_mbl_sweep_TEST_small --backend gpu --max-full-dim 32768
```


## Paper A reproducibility

The Paper-A release boundary and raw-data workflow are documented in
[REPRODUCING_PAPER_A.md](REPRODUCING_PAPER_A.md). Changes in this release are
listed in [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Authors and provenance

The Paper-A release and Quantum Darwinism extensions are maintained by
**Aleksander Lasek**. This repository descends from an earlier pySL codebase by
H. V. Lepage, A. A. Lasek, and C. H. W. Barnes; the Git history preserves
those earlier contributions.

## Citation

Use the metadata in [CITATION.cff](CITATION.cff) when citing this software.

## License

This software is distributed under the [BSD 3-Clause License](LICENSE).

## Publications
* [Entanglement generation via power-of-swap operations between dynamic electron-spin qubits (2020)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.052305)
* [Sound-driven single-electron transfer in a circuit of coupled quantum rails (2019)](https://www.nature.com/articles/s41467-019-12514-w)
* [Protocol for fermionic positive-operator-valued measures (2017)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.96.052305)


## Acknowledgments

* D. R. M. Arvidsson Shukur
* A. Andreev
* E. T. Owen
* J. Mosakowski

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 642688.
