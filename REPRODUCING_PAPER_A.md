# Reproducing Paper A

`paper-a-v1.1.0` (2026-09-09) is the successor Paper-A software release. It
captures the current simulator, batch configurations, analysis code, and the
fixed-RMS Gaussian field mean/width study. The historical `paper-a-v1.0.0` release is not included in this public
repository.

## Compact public reproduction

Follow [paper_a/README.md](paper_a/README.md) to regenerate the released
figures and tables from the saved package on CPU. That workflow does not need
the raw production simulation outputs or the manuscript notes repository.

## Environment

The full simulation workflow requires an NVIDIA GPU visible to `nvidia-smi`.
Create an isolated environment and install exactly one existing CUDA-specific
requirements file:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# POSIX: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda12.txt  # CUDA 12
# or: python -m pip install -r requirements-cuda13.txt  # CUDA 13
```

Use NVIDIA's current [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) or [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) documentation if the host itself needs CUDA setup. These project instructions do not require removing or replacing an installed driver.

## Fig. 6(b) threshold estimator

At the specified time, construct the fragment-q10 Holevo curve separately for
each realization and find its first threshold fragment size. For paired
geometries, average the defined threshold sizes within each realization before
averaging the defined realization values. A realization without a crossing by
`m <= N_E/2` is undefined, excluded from the conditional mean, and reported
through its crossing count. Do not invert or average redundancies before this
calculation. Bootstrap only over realizations, keeping paired geometries and
stored fragment selections together.

The released threshold manifest records this order alongside the exact values.

## Detailed from-scratch analysis

The repository retains the GPU batch and analysis scripts. Raw production
outputs are not included because the time series and fragment-resolved samples
are large; they are available from the authors upon reasonable request. Place
the supplied batch directories under `data/` and preserve their manifests.

Run these commands from the repository root. The first command in each row
inspects the final plan without GPU evolution; `--manifest-only` writes only a
manifest. The second launches the final batch on GPU 0 and resumes completed
runs. Supply the profile directly with `--profile final`; no environment
variable is required.

| Batch | Inspect final plan | Run final batch |
| --- | --- | --- |
| Core suite | `python scripts/run_paper_a_suite.py --profile final --dry-run` | `python scripts/run_paper_a_suite.py --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Matched-Λ submission | `python scripts/run_paper_a_submission.py --profile final --dry-run` | `python scripts/run_paper_a_submission.py --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Field-profile control | `python pySLbatch.py --config batch_configs.paper_a_field_control --profile final --manifest-only` | `python pySLbatch.py --config batch_configs.paper_a_field_control --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Higher-ℕ validation | `python scripts/run_paper_a_higher_n_validation.py --profile final --dry-run` | `python scripts/run_paper_a_higher_n_validation.py --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Finite-ℕ threshold | `python scripts/run_paper_a_fragment_threshold_finite_n.py --profile final --dry-run` | `python scripts/run_paper_a_fragment_threshold_finite_n.py --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Intermediate-ℕ Fig. 6(b) threshold | `python scripts/run_paper_a_fragment_threshold_finite_n.py --intermediate --profile final --dry-run` | `python scripts/run_paper_a_fragment_threshold_finite_n.py --intermediate --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Λ = (2 + √3)/4 threshold completion | `python pySLbatch.py --config batch_configs.paper_a_fragment_threshold_lambda0933 --profile final --manifest-only` | `python pySLbatch.py --config batch_configs.paper_a_fragment_threshold_lambda0933 --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Low-field refinement | `python pySLbatch.py --config batch_configs.paper_a_low_field_refinement --profile final --manifest-only` | `python pySLbatch.py --config batch_configs.paper_a_low_field_refinement --profile final --workers 1 --gpu-ids 0 --skip-existing` |
| Fixed-RMS field mean/width | `python pySLbatch.py --config batch_configs.paper_a_field_mean_width --profile final --manifest-only` | `python pySLbatch.py --config batch_configs.paper_a_field_mean_width --profile final --workers 1 --gpu-ids 0 --skip-existing` |

The optional `--n26` mode is a separate capacity experiment and is not a
Paper-A release result.
