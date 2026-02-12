# Multi-Class ARZ + RL Traffic Signal Control (West Africa)

A reproducible Python implementation of a **multi-class macroscopic traffic model** (motorcycles + cars) coupled with **Deep Reinforcement Learning (DQN)** for adaptive signal control on a signalized urban corridor.

This repository is the codebase associated with the JITS paper:

> **A Multi-Class Macroscopic Traffic Model with Reinforcement Learning Signal Control for Heterogeneous Urban Corridors in West Africa**

## Why this project matters

Most RL traffic-signal papers assume data-rich environments (detectors, calibrated microsimulators, expensive tooling). This implementation demonstrates a practical pathway for **data-scarce cities**:

- model heterogeneous traffic (motos + cars) with a multi-class ARZ formulation,
- train an adaptive controller on a standard CPU,
- evaluate across multiple demand regimes,
- generate publication-ready figures from the same pipeline.

## Key results (current run)

- Training budget: **60,000 steps** (~2,000 episodes)
- Hardware: **standard CPU**
- Training time: **~38 minutes**
- Overall reward improvement vs fixed timing: **+18.9%**
- Scenario breakdown:
  - Light: **-2.1%**
  - Moderate: **+17.1%**
  - Heavy: **+25.8%**
  - Saturated: **+31.4%**

## Repository structure

```text
python/
├─ params.py                  # Physical parameters, scenarios, DQN config
├─ solver.py                  # Multi-class ARZ finite-volume solver (LxF)
├─ environment.py             # Gymnasium environment + fixed-time baseline
├─ train.py                   # Two-phase training/evaluation pipeline
├─ generate_all_figures.py    # Generates paper/thesis figures
└─ data/results/              # Saved metrics, eval outputs, model artifacts
```

## Installation

### 1) Clone and enter folder

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

- **Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

- **Linux/macOS:**

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Full pipeline (recommended)

```bash
python train.py
```

Runs:
1. environment sanity check,
2. proof phase (10k),
3. full training phase (60k),
4. multi-scenario evaluation,
5. metrics export in `data/results/`.

### Optional modes

```bash
python train.py --proof-only
python train.py --full-only
```

### Generate all figures

```bash
python generate_all_figures.py
```

Outputs are written to:

- `../images/chapter3/fig_corridor_schematic.png`
- `../images/chapter3/fig_fundamental_diagram.png`
- `../images/chapter3/fig_scenario_profiles.png`
- `../images/chapter3/fig_training_curve.png`
- `../images/chapter3/fig_rl_comparison.png`

## Reproducibility notes

- Main corridor: 1.5 km, 3 signalized intersections
- Action space: `Discrete(8)` (joint phases for 3 signals)
- Observation space: 18-dimensional normalized state
- Numerical scheme: Local Lax-Friedrichs + forward Euler + CFL 0.5
- Scenarios: `light`, `moderate`, `heavy`, `saturated`
<!-- 
## Cite this work

If you use this code, please cite the associated paper (add DOI/link once public):

```bibtex
@article{hontinfinde_ahouanye_2026,
  title   = {A Multi-Class Macroscopic Traffic Model with Reinforcement Learning Signal Control for Heterogeneous Urban Corridors in West Africa},
  author  = {Hontinfinde, Régis Donald and Ahouanye, Elonm Josaphat},
  journal = {Journal of Intelligent Transportation Systems},
  year    = {2026}
}
``` -->

## Contact

- Régis Donald Hontinfinde — donald.hontinfinde@yahoo.com
- Elonm Josaphat Ahouanye — josaphatahouanye@gmail.com
