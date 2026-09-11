# QD-summary jobs 1-3 timestep recommendation

Generated from the RTX 5090 studies on 2026-07-14.

The acceptance limit for MI, Holevo, and discord is `1.4311425826130098e-4`, twice the MI error of the previously accepted no-EE `dT=0.025` case. Reported information errors use the second-order Richardson estimate

`estimated error = |metric(h) - metric(h_ref)| / (1 - (h_ref / h)^2)`.

| Job | theta | selected dT | first rejected dT | max norm dev | max D(rhoS) | max entropy err | estimated MI err | estimated Holevo err | estimated discord err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stab_pi4 | pi/4 | 1/1050 = 0.0009523809524 | 0.001 | 1.376867e-5 | 8.049906e-6 | 1.011249e-5 | 1.415557e-4 | 1.240999e-4 | 1.142572e-4 |
| Stab_pi8 | pi/8 | 1/1200 = 0.0008333333333 | 1/1100 = 0.0009090909091 | 1.049042e-5 | 1.046247e-5 | 1.757622e-5 | 1.132268e-4 | 8.169743e-5 | 9.809108e-5 |
| Stab_pi0 | 0 | 0.0005 | 0.000625 | 3.814697e-6 | 5.475468e-5 | 9.371444e-6 | 1.233116e-5 | 9.932698e-6 | 8.328725e-6 |

The exact-state screen used `alpha2=3`, all three deterministic production seeds, and biases `0`, `0.25`, and `0.5` over `T=30`. Full MI/Holevo/discord scans used the highest-spectral-radius seed, bias `0.5`, seven samples over `T=30`, and a finer reference timestep. Global state fidelity is retained as a diagnostic but is not an acceptance gate because accumulated many-body phase error is substantially more sensitive than the figure observables.
