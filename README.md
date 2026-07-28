# Affective Ising Model with Temporal Inputs

Course project for Stochastic Modelling.

## Project overview

This project studies the model introduced in the paper **“The Affective Ising Model: A computational account of human affect dynamics.”**

The reproduction phase will cover:

- the free-energy function;
- its gradient;
- the stationary density; and
- an Euler–Maruyama simulation.

The validation phase will compare the stationary distributions obtained from simulation with their theoretical counterparts.

The planned extension will study temporary negative external inputs. The analysis will focus on switching probability and recovery time after an input is removed.

This repository currently contains documentation and project structure only. The mathematical equations and simulation implementation are intentionally left for future work.

## Project structure

```text
.
├── notebooks/       # Exploratory analysis and reproducible computational notebooks
├── presentation/    # Course presentation materials
├── references/      # Paper notes and reference metadata
├── results/
│   ├── figures/     # Generated plots
│   └── tables/      # Generated tabular results
└── src/             # Reusable Python source code
```

## Current progress

- [x] Create the repository structure
- [x] Document the project scope and reproducibility workflow
- [x] Record the model equations from the main paper
- [x] Reproduce the free-energy function and gradient
- [x] Derive or reproduce the stationary density
- [x] Implement the Euler–Maruyama simulation
- [x] Validate simulated stationary distributions against theory
- [x] Add temporary negative external inputs
- [x] Estimate switching probability
- [x] Measure recovery time
- [x] Prepare final figures, tables, and presentation

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
3. Keep reusable computations in `src/` and use notebooks for analysis and presentation.
4. Record experiment parameters in the corresponding notebook or result metadata.
5. Save generated plots and tables under `results/figures/` and `results/tables/`.
6. Run notebooks from a clean kernel, from top to bottom, before reporting results.

Exact environment details can be captured when producing final results:

```bash
python --version
python -m pip freeze
```

