# `params.example.json`

`params.example.json` is strict JSON, so it is intentionally comment-free and can be copied to `run_params/params.json` for a local run. This file documents each parameter in the same order.

| Field | Meaning |
| --- | --- |
| `clampISE` | Small clamp threshold used by mutual-information/redundancy calculations to avoid unstable near-zero behavior. |
| `AverageOverRunsN` | Number of disorder/initial-state realizations to average in one run. Mutual-information and entropy metrics are averaged; wavefunction snapshots are saved per realization when `store_psi` is enabled. |
| `QREmaxFragSize` | Largest contiguous environment fragment size used for SBS/QRE diagnostics. Must be no larger than `nqubits_E`. |
| `nqubits_S` | Number of system qubits. Current Mironowicz workflows usually use `1`. |
| `nqubits_E` | Number of environment qubits. Total qubits are `nqubits_S + nqubits_E`. |
| `psi_S_spec` | Initial system-state selector. `"x+"` means the +X superposition state. |
| `psi_E_spec` | Initial environment-state selector. `"rand"` draws a random environment state. |
| `psi_bias` | Bias value used when `psi_E_spec` is `"bias"`; kept at `0` here because `"rand"` is selected. |
| `dtypepbits` | Floating-point precision for simulation arrays. Supported values are `32` and `64`. |
| `dT` | Time-evolution step size. `T` and `printT` must be integer multiples of this value. |
| `T` | Final physical simulation time. |
| `printT` | Physical interval between recorded/printed diagnostics. |
| `H_SE_bonds` | System-environment bond pattern. `"S_to_all"` couples system qubits to all environment qubits. |
| `H_SE_Special` | Special Hamiltonian builder name. `"Mironowicz_rand_EE_transverse"` uses the random Mironowicz S-E/self-field model with explicit transverse E-E terms `0.5 * Z kron (I + X + Y)`. `"Mironowicz_rand_EE_ZX"` keeps only the simplified `Z kron X` E-E pair term. `"Mironowicz_rand_EE_ZZ_to_ZX"` uses `Z kron (cos(Mironowicz_ZZ_to_ZX_epsilon) Z + sin(Mironowicz_ZZ_to_ZX_epsilon) X)`. |
| `Mironowicz_epsilon` | Mironowicz model epsilon parameter. |
| `Mironowicz_epsilon2` | Secondary Mironowicz epsilon parameter. |
| `Mironowicz_ZZ_to_ZX_epsilon` | E-E interpolation angle for `Mironowicz_rand_EE_ZZ_to_ZX`. |
| `Mironowicz_alpha2` | Mironowicz environment-field/disorder scale used by the model builder. |
| `Mironowicz_theta` | Mironowicz theta angle in radians. JSON params may also use expression strings such as `"pi/2"`. |
| `Mironowicz_H_E` | Environment one-body operator choice for Mironowicz builders. `"Z"` selects sigma-z fields. |
| `H_SE_J` | Overall S-E coupling strength or standard deviation, depending on `H_SE_p` and the selected builder. |
| `seed` | Random seed. Use an integer for reproducible per-run seeds or `"TIME"` for time-derived seeds. Saved params also record the concrete `run_seeds` used. |
| `H_SE` | Generic S-E operator list, used by generic builders and kept here as a conventional fallback. Each entry is `[coefficient, system_operator, environment_operator]`. |
| `H_SE[0][0]` | Coefficient for the first generic S-E term. |
| `H_SE[0][1]` | System-qubit Pauli/operator label for the first generic S-E term. |
| `H_SE[0][2]` | Environment-qubit Pauli/operator label for the first generic S-E term. |
| `H_SE_p` | S-E coupling parameterization. `"Norm"` uses normalized/random coupling behavior in current builders. |
| `H_S` | System-only Hamiltonian terms. Empty list disables explicit system-only terms. |
| `H_S_J` | System-only coupling scale. `0` disables it for this example. |
| `H_S_p` | System-only coupling parameterization. |
| `H_E` | Environment-only one-body Hamiltonian terms for generic builders. Empty list disables generic terms. |
| `H_E_J` | Environment one-body coupling scale. |
| `H_E_p` | Environment one-body coupling parameterization. |
| `H_EE` | Generic E-E interaction list. Each entry is `[coefficient, left_operator, right_operator]`. |
| `H_EE[0][0]` | Coefficient for the first generic E-E term. |
| `H_EE[0][1]` | Operator label on the left environment qubit. |
| `H_EE[0][2]` | Operator label on the right environment qubit. |
| `H_EE_bonds` | Environment-environment bond pattern. `"1DNN"` means nearest-neighbor chain bonds. |
| `H_EE_J` | E-E coupling strength. `0.0` disables E-E coupling in this example. |
| `H_EE_p` | E-E coupling parameterization. `"Const"` uses the literal `H_EE_J` value. |
| `H_noise` | Noise Hamiltonian terms. Each entry is `[coefficient, operator]`. |
| `H_noise[0][0]` | Coefficient for the first noise term. |
| `H_noise[0][1]` | Operator label for the first noise term. |
| `p_noise_Gate` | Probability/rate parameter for stochastic gate noise. `0` disables gate noise. |
| `noiseT_Gate` | Gate-noise interval in time steps/units expected by the noise routine. Only relevant when `p_noise_Gate > 0`. |
| `noiseT` | Continuous/random Hamiltonian-noise interval. `0` disables this noise path. |
| `noiseStd` | Noise standard deviation when random Hamiltonian noise is enabled. |
| `store_psi` | Whether to save full wavefunction snapshots at each printed sample time. Snapshots are stored per realization and are not averaged. |
| `store_psi_on_disk` | Whether the current run's snapshot buffer should be disk-backed before it is moved into the saved run directory. Relevant only when `store_psi` is true and recommended for large systems. |
| `store_correlators` | Whether to compute/store two-time correlators. This adds memory and runtime cost. |
| `compute_discord` | Whether to compute Z-basis Holevo information and discord alongside `I(S:E)`. Defaults to `true`; set to `false` to recover the faster MI-only print path. |
| `H_precompute_mode` | Sparse Hamiltonian materialization mode. `"combined_cpu"` builds one combined CSR on CPU and transfers it to GPU for fast evolution; `"grouped_cpu"` extracts diagonal terms and combines non-diagonal terms into CSR groups; `"sum_terms"` avoids CPU combination but applies every sparse term separately each timestep and is mainly a fallback. |
| `H_precompute_group_size` | Number of non-diagonal local terms to combine per CSR group when `H_precompute_mode` is `"grouped_cpu"` or when `combined_cpu` falls back. Larger values usually improve timestep speed but increase CPU precompute memory. |
| `profile_evolution` | Enables detailed CUDA-event timing for timestep substeps in the final perf summary. This synchronizes GPU work and is intended for diagnosis, not production sweeps. |
| `profile_evolution_max_steps` | Maximum number of timesteps to profile with CUDA events when `profile_evolution` is true. Keeps diagnostic runs from becoming dominated by profiler synchronization. |
| `log_gpu_memory` | Whether to print a final GPU memory summary. Logs sampled CUDA device usage plus CuPy memory-pool high-water marks. |
| `plot_mode` | Plot display backend. `"plot_pane"` collects figures into the Tk plot pane; `"none"` suppresses display for batch runs; other supported values are validated in `src/input/validation.py`. |
| `save_figures` | Figure-saving policy. `"latest"` overwrites `data/nonbatch/latest_figs/` each standalone run; other supported values are `"off"`, `"run"`, and `"both"`. |
| `save_legacy_rundata` | Whether to also write the old full-object `.rundata` pickle. Defaults to `false`; v2 `results.npz` is the normal save format. |
