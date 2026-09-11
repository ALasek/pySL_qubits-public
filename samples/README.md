# Samples

## Table of Contents
- [Samples](#samples)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Particle in a box](#particle-in-a-box)
    - [Output](#output)
  - [Double slit experiment](#double-slit-experiment)

## About
To use these samples, rename the input file to `parameters.py` and place in `src/input/parameters.py`.


## Particle in a box
**File name:** `parameters_particle-in-a-box-1D.py`

**Memory usage:** 1.2 MB

**Memory usage (with plots):** 3.0 MB

This sample will create an infinite potential well in the x dimension of length 200nm. The potential is set to zero inside the well and infinity outside the well.
A free electron is placed in the ground state and the time evolution is calculated.

### Output
Once the calculation is finished, a directory called `Particle_in_a_Box_1D` will be created and contain snapshots of the wave function at different time steps.

This directory also contains a copy of `InputParameters.txt` and the initial wave function for future reference.

The function `animate1D.py` can be called to create a simple animation of the real and imaginary components of the ground state wave function as they evolve in time.

<p align="center">
  <img src="https://www.dropbox.com/s/b15fv2c4f9d7x28/animation.gif?raw=1">
</p>

To observe a different initial wave function, the contribution of the first few eigenstates can be adjusted in `parameters.py`. Modifying the `InputStates` variable to the following
```
InputStates = 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
```
creates an equal superposition of the ground state and the first excited state. The resulting oscillations can be seen in the animation below.

<p align="center">
  <img src="https://www.dropbox.com/s/dpt3xrcdn39ooxz/animation1.gif?raw=1">
</p>

As defined in the input parameters, this simulation spans 400 picoseconds. Given the two first eigenvalues, we can calculate the period of oscillation of the wave function as <img src="https://render.githubusercontent.com/render/math?math=T = 2 \pi / (\lambda_1-\lambda_0)">. In natural units, the first five eigenvalues of this problem are calculated to be:
```
0.000120901495393
0.000483685405925
0.00108822679613
0.00193437782582
0.00302193243988
```
This gives a period of oscillation of approximately T = 149ps.





## Double slit experiment
**File name:** `parameters_double-slit-experiment.py`

**Memory usage:** 6.9 MB

**Memory usage (with plots):** 14.9 MB


In this example, an external wave function is imported. The file `wave_DSE1D.txt` must be located in the `/input/` directory.

This sample will create an infinite potential well in the x dimension of length 400nm. The potential is set to zero inside the well and infinity outside the well.

A free electron is placed in two Gaussian wave packets located at +/- 40nm and the time evolution is calculated. This calculation simulates a single electron that goes through two slits of an interferometer.

### Analysis
Once the calculation is finished, a directory called `Double_slit_experiment_1D` will be created and contain snapshots of the wave function at different time steps.

This directory also contains a copy of `InputParameters.txt` and the initial wave function for future reference.

The function `plot1Dtrace.py` can be called to create a trace plot of the wave function.

![1D - Double Slit Experiment](https://www.dropbox.com/s/91orqvwukzmbvqw/trace1D.png?raw=1)
