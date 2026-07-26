# Affective Ising Model with Temporal Inputs

Course project for Stochastic Modelling.

## Project overview

This project studies the paper **“The Affective Ising Model: A computational account of human affect dynamics.”** The model represents positive affect and negative affect as two interacting stochastic variables. Their dynamics take place on a free-energy landscape that can contain one or more stable affective states, called homebases.

The project has two main parts:

1. Reproduce and validate the central mathematical mechanism of the Affective Ising Model.
2. Extend the model with temporary negative external inputs and study transitions between affective homebases.

## Reproduction of the model

The reproduction includes:

- implementation of the free-energy function;
- implementation and numerical validation of its analytical gradient;
- calculation and normalization of the theoretical stationary density;
- simulation of the stochastic differential equation with the Euler–Maruyama method;
- visualization of the deterministic drift field; and
- comparison of the simulated stationary distribution with the theoretical distribution.

The stationary validation shows two symmetric high-probability regions: high positive/low negative affect and low positive/high negative affect. The empirical maxima obtained from simulation are close to the theoretical maxima.

## External-input extension

The original AIM includes time-dependent positive and negative inputs, but the paper does not analyse them because the available datasets do not contain sufficient information about external events.

This project studies a synthetic rectangular negative input \(B_2(t)\). The input modifies the model while the stochastic trajectory is being generated. The experiments compare:

- no external input;
- a weak negative input;
- a strong negative input;
- several input amplitudes to obtain an intensity-response curve; and
- acute, prolonged and fragmented inputs with the same total area.

The main output measures are:

- switching probability: the proportion of trajectories that finish in the negative-affect basin;
- recovery time after the input ends;
- persistent, temporary and absent recovery; and
- Wilson 95% confidence intervals for the Monte Carlo probabilities.

The equal-area experiment keeps the accumulated input constant while changing its temporal organization. This allows the project to test whether switching depends only on total input or also on how that input is distributed over time.

## Project structure

```text
.
├── notebooks/       # Reproduction, validation and extension experiments
├── presentation/    # Course presentation materials
├── references/      # Paper notes and reference metadata
├── results/
│   ├── figures/     # Generated plots
│   └── tables/      # Generated tabular results
└── src/aim/         # Reusable model, simulation, input and metric functions
```

The notebooks are organized as a computational diary:

- `01_free_energy_landscape.py`: free energy, gradient and drift field;
- `02_stationary_density.py`: theoretical stationary density;
- `03_euler_maruyama.py`: construction of the stochastic simulator;
- `04_stationary_validation.py`: empirical and theoretical validation;
- `05_inputs.py`: control, weak and strong negative inputs;
- `06_switching_and_recovery.py`: Monte Carlo switching and recovery;
- `07_trajectory_metrics.py`: trajectory measures;
- `08_input_intensity_sweep.py`: effect of input intensity;
- `09_equal_area_inputs.py`: equal-area temporal-input comparison.

## Current progress

- [x] Create the repository structure
- [x] Document the project scope and reproducibility workflow
- [x] Record the model equations from the main paper
- [x] Reproduce the free-energy function and gradient
- [x] Derive or reproduce the stationary density
- [x] Implement the Euler–Maruyama simulation
- [x] Validate simulated stationary distributions against theory
- [x] Add temporary negative external inputs
- [x] Estimate switching probability with Monte Carlo simulation
- [x] Measure recovery time
- [x] Add Wilson confidence intervals
- [x] Study the effect of input intensity
- [x] Compare acute, prolonged and fragmented equal-area inputs
- [ ] Prepare final figures, tables, and presentation

## Installation

Python 3.11 or later is recommended.

```bash
git clone https://github.com/rosaarosae/affective-ising-temporal-inputs.git
cd affective-ising-temporal-inputs
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproducibility

To keep analyses reproducible:

1. Create a fresh virtual environment and install dependencies from `requirements.txt`.
2. Set and record random seeds in every stochastic experiment.
3. Keep reusable computations in `src/aim/` and use notebooks as a readable record of each stage.
4. Record experiment parameters in the corresponding notebook or result metadata.
5. Save generated plots and tables under `results/figures/` and `results/tables/`.
6. Run notebooks from a clean kernel, from top to bottom, before reporting results.

Exact environment details can be captured when producing final results:

```bash
python --version
python -m pip freeze
```
