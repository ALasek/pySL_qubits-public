# Paper A low-field refinement

Config: `batch_configs.paper_a_low_field_refinement`.
This is a standard `pySLbatch.py` batch, with the same smoke/final profile
convention and v2 output layout as `paper_a_field_control`.

The final batch adds eight cases: h_rms/g = 0.05, 0.10, 0.15, and 0.20,
each with uniform and Gaussian Z fields. It uses N_E=16, p=0.5, theta=0,
g=0.1, no E-E coupling, T=60, dT=0.002, and printT=1. Physical h0 or W
values are 0.005, 0.010, 0.015, and 0.020. Seed 1337 expands to the same
24 realization seeds as the original field-control batch. Sampling retains
64 fragments per evaluated size, or all subsets where fewer exist, with
fixed fragment trajectories and stored sample identities/information.

The intentional protocol change is `clampISE=0.0`: the original 0.01-nat
cutoff must not hide low-field signals. This removes censoring, not
integration or floating-point error. Compare against the uncensored exact
reference for the original points. The original batch is not overwritten;
new outputs go to `data/Paper_A_final_LowFieldRefinement/`.

## Prepare or run

From the pySL_qubits repository root, on Linux or Windows:

```powershell
python pySLbatch.py --config batch_configs.paper_a_low_field_refinement --profile final --manifest-only
python pySLbatch.py --config batch_configs.paper_a_low_field_refinement --profile final --workers 1 --gpu-ids 0 --skip-existing
```

No environment-variable setup is needed. The first command only creates
the manifest; the second launches the GPU solver. Use `--profile smoke` for eight small cases
(N_E=6, one realization, T=8, dT=0.01, printT=2, four fragments).
Without `--profile`, the existing environment setting is respected and the
fallback remains smoke. The CLI option takes precedence and is inherited
by worker processes. Update `pySLbatch.py` on the server before using it.

## Analysis scope

Combine these points with the original zero-field and 0.25-and-above
reference points during analysis, preserving batch/run identities. Use
the [30,60] window, lead with mean Holevo information and mean-threshold
redundancy, and report q10 and the minimum-over-time q10 as robustness
checks. For nested bootstrapping, resample complete fragment trajectories
and take q10 before the time minimum. The `analysis` fields describe this
contract; creating or running the batch does not automatically produce the
manuscript's custom figures.

These points resolve the previously unsampled onset. They do not by
themselves locate a continuous optimum: that would also merit refinement
between 0.25 and 0.5 and an observation-window sensitivity check.

Exact CPU propagation is sufficient for this independent-witness model;
the standard GPU batch is provided for compatibility with previous
production runs. Preparing this config does not implement or execute a
new-case exact CPU driver.
